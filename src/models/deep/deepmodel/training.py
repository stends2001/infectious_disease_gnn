from typing import Dict, Any, Union, Optional, List, Tuple, TYPE_CHECKING, Literal
import pandas as pd
import torch 
from tqdm import tqdm
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...utils.loss.losshandler import LossHandler
from ...base.basemodel.statusmixin import ModelStatus
from ....dataloading.dataloaders import DeepDataLoaderManager, GraphDataLoaderManager
from ....utils.colors import traincolor, valcolor

if TYPE_CHECKING:
    from ....dataloading.epiconfig import EpiConfig
    from ..strategies.basestrategy import Strategy    

class DeepModelTrainMixin:
    """ 
    Mixin class that deals with the model's training.
    NOTE we have four stubs here, two of which are defined 
    in ModelStatusMixin and two in DeepModelPresentationMixin.
    These stubs follow the actual functions' signatures but
    are not called. Simply here for typing.    

    The main methods are `train()` and `show_monitoring_metrics()`.
    Besides the stubs, there's also two helper functions for the 
    monitoring-metrics plotting
    """    
    status_dict:        Dict[ModelStatus, bool]
    epiconfig:          'EpiConfig'
    config_info:        Dict[str, Any]    
    model:              torch.nn.Module
    dataloadermanager:  Union[DeepDataLoaderManager, GraphDataLoaderManager]
    strategy:           'Strategy'
    device:             torch.device
    optimizer:          Optimizer
    scheduler:          _LRScheduler    
    loss:               LossHandler
    min_delta:          float
    patience:           int 
    verbose:            int
    monitoring_metrics: pd.DataFrame
    n_epochs:           int

    def train(self):
        """ 
        Train DeepModel
        Uses set strategy extensively.
        """
        self._check_status(['model_hparams_set', 'global_hparams_set'])
  
        train_loader = self.dataloadermanager.dataloader_train 
        val_loader   = self.dataloadermanager.dataloader_val 

        verbose_loops, epoch_iter = self._return_verbose_iter()

        # ====== PRE-TRAINING ====== #
        self.model.train()
        best_val_loss       = float('inf')
        patience_counter    = 0
        best_model_state    = None

        list_val_loss       = []
        list_train_loss     = []
        list_patience       = []
        list_lr             = []

        L_train             = len(train_loader)
        L_val               = len(val_loader)

        if len(verbose_loops) > 0:
            self._return_verbose_line()

        # Each epoch is divided into:
        #   1. training phase
        #   2. validation phase
        #   3. update phase

        for epoch in epoch_iter:
            # for printing purposes
            repr_epoch = epoch + 1 

            # Reset state at epoch start
            self.strategy.reset_state_epoch()

            # ======================== TRAINING PHASE ========================
            total_loss = 0
            
            for snapshot in train_loader:
                snapshot = snapshot.to(self.device)

                # different models have different input and output in steps
                # taken care of using the strategy

                loss_train = self.strategy.training_step(
                    model       = self.model, 
                    snapshot    = snapshot, 
                    optimizer   = self.optimizer, 
                    loss_fn     = self.loss
                )
                
                total_loss += loss_train
            
            train_loss = total_loss / L_train
            list_train_loss.append(train_loss)

            # ======================== VALIDATION PHASE ========================
            self.model.eval()
            val_loss = 0
            
            # Reset state for validation
            self.strategy.reset_state_dataset()

            with torch.no_grad():
                for snapshot in val_loader:
                    snapshot = snapshot.to(self.device)

                    loss_val = self.strategy.validation_step(
                        model       = self.model, 
                        snapshot    = snapshot, 
                        loss_fn     = self.loss
                    )

                    val_loss += loss_val
            
            val_loss = val_loss / L_val
            list_val_loss.append(val_loss)
            
            # ======================== UPDATE PHASE ========================
            # the lr used in this epoch
            current_lr = self.optimizer.param_groups[0]['lr']
            list_lr.append(current_lr)
            
            self.model.train()

            # Check if validation loss improved
            val_improved = val_loss < (best_val_loss - self.min_delta)

            # if so => save best model
            if val_improved:
                best_val_loss   = val_loss
                patience_counter= 0
                best_model_state= self.model.state_dict().copy()
                list_patience.append(False)

            else:
                patience_counter += 1
                list_patience.append(True)

            if patience_counter >= self.patience:
            
                if self.verbose>=0:
                    print(f"Early stopping: Validation loss hasn't improved for {self.patience} epochs")

                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                        print(f"Restored model from best validation loss: {best_val_loss:.4f}")

                if epoch < self.n_epochs/10:
                    print(f'training was stopped before 10% of the set epochs. Inspect monitoring metrics.')

                break              

            # Step scheduler => scheduler.step requires val loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss) # type: ignore
            # for other schedulers, no arguments required
            else:
                self.scheduler.step()

            new_lr = self.optimizer.param_groups[0]['lr']

            if repr_epoch in verbose_loops:
                self._return_verbose_line(repr_epoch, train_loss, val_loss,
                                          "v" if val_improved else None,
                                          None if val_improved else f"{patience_counter}/{self.patience}",
                                          True if current_lr != new_lr else None
                                          )     
            
        self.monitoring_metrics = pd.DataFrame({'train_loss'    : list_train_loss,
                                                'val_loss'      : list_val_loss,
                                                'patience'      : list_patience,
                                                'learning_rate' : list_lr}).reset_index(names='epoch') # index starting from 0
        
        self.monitoring_metrics['epoch'] = self.monitoring_metrics['epoch'] + 1 # index + 1

        if self.verbose >=1:
            self.show_monitoring_metrics()

        self._update_status('trained')

    def show_monitoring_metrics(self) -> None:
        """Shows plot of trainloss, valloss, patience and learning rate per epoch."""
 
        if not hasattr(self, 'monitoring_metrics'):
            raise ValueError('no monitoring metrics found')
      
        fig, axes_array = plt.subplots(1, 3, figsize=(24, 4))
        axes: list[Axes]= list(axes_array.flatten())

        # lines: train_loss, val_loss and learning_rate
        sns.lineplot(data = self.monitoring_metrics, x='epoch', y='train_loss',   color=traincolor,   label='Train Loss',         ax=axes[0])
        sns.lineplot(data = self.monitoring_metrics, x='epoch', y='val_loss',     color=valcolor,     label='Validation Loss',    ax=axes[1])
        sns.lineplot(data = self.monitoring_metrics, x='epoch', y='learning_rate',color='black',      label='Learning Rate',      ax=axes[2])

        # Scatter patience epochs and corresponding values
        self._plot_scatter_patience_on_ax('train_loss',     axes[0])
        self._plot_scatter_patience_on_ax('val_loss',       axes[1])
        self._plot_scatter_patience_on_ax('learning_rate',  axes[2])                

        for ax in axes:
            self._format_ax(ax)

        axes[0].set_title('Training loss')      
        axes[1].set_title('Validation loss')    
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')

        fig.show()

    def _plot_scatter_patience_on_ax(self, y: Literal['train_loss','val_loss','learning_rate'], ax: Axes) -> None:
        """plots red cross for patience epoch on any ax"""
        patience_mask = self.monitoring_metrics['patience'] > 0

        ax.scatter(x        = self.monitoring_metrics['epoch'][patience_mask], 
                   y        = self.monitoring_metrics[y][patience_mask], 
                   color    ='red', 
                   marker   ='x', 
                   label    ='Patience Epochs')

    def _format_ax(self, ax: Axes) -> None:
        """basic ax format"""
        ax.grid()
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.legend()

    # ======== STUBS ======= #
    def _check_status(self, required_states: Union[List[ModelStatus], ModelStatus]) -> None: ...
    def _update_status(self, status: ModelStatus) -> None: ...
    def _return_verbose_iter(self) -> Tuple[list, Union[range, tqdm[int]]]: ...
    def _return_verbose_line(self, 
                             epoch:         Optional[int]  = None, 
                             train_loss:    Optional[float]= None, 
                             val_loss:      Optional[float]= None, 
                             new_best:      Optional[str]  = None, 
                             patience:      Optional[str]  = None, 
                             lr_updated:    Optional[bool] = None): ...
