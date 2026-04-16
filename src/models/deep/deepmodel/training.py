from typing import Dict, Any, Union, Optional, List, Tuple, TYPE_CHECKING
import pandas as pd
import torch 
from tqdm import tqdm
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ...issues import ModelStatusError
from ...utils.loss.losshandler import LossHandler
from ...base.statusmixin import ModelStatus
from ....dataloading.dataloaders import DeepDataLoaderManager, GraphDataLoaderManager
from ....utils.colors import traincolor, valcolor

if TYPE_CHECKING:
    from ....dataloading.epiconfig import EpiConfig
    from ..strategies.basestrategy import Strategy    

class DeepModelTrainMixin:
    """ 
    # TODO
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
    monitoring_metrics: Optional[pd.DataFrame]    

    # ======== STUBS ======= #
    def _check_status(self, required_states: Union[List[ModelStatus], ModelStatus]) -> None: ...
    def _update_status(self, status: ModelStatus) -> None: ...
    def _return_verbose_iter(self) -> Tuple[list, Union[range, tqdm]]: ...
    def _return_verbose_line(self, 
                             epoch:         Optional[int]  = None, 
                             train_loss:    Optional[float]= None, 
                             val_loss:      Optional[float]= None, 
                             new_best:      Optional[str]  = None, 
                             patience:      Optional[str]  = None, 
                             lr_updated:    Optional[bool] = None): ...

    def train(self):
        """ 
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
        #   training phase
        #   validation phase
        #   update phase

        for epoch in epoch_iter:
            # for printing purposes
            num_epoch = epoch + 1 

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

            if patience_counter >= self.patience and self.verbose>=0:
                print(f"Early stopping: Validation loss hasn't improved for {self.patience} epochs")

                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                    print(f"Restored model from best validation loss: {best_val_loss:.4f}")

                break              

            # Step scheduler => scheduler.step requires val loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss) # type: ignore
            # for other schedulers, no arguments required
            else:
                self.scheduler.step()

            new_lr = self.optimizer.param_groups[0]['lr']

            if num_epoch in verbose_loops:
                self._return_verbose_line(num_epoch, train_loss, val_loss,
                                          "v" if val_improved else None,
                                          None if val_improved else f"{patience_counter}/{self.patience}",
                                          True if current_lr != new_lr else None
                                          )     
            
        self.monitoring_metrics = pd.DataFrame({'train_loss'    : list_train_loss,
                                                'val_loss'      : list_val_loss,
                                                'patience'      : list_patience,
                                                'learning_rate' : list_lr}).reset_index(names='epoch')
        
        self.monitoring_metrics['epoch'] = self.monitoring_metrics['epoch'] + 1

        if self.verbose >=1:
            self.show_monitoring_metrics()

        self._update_status('trained')

    def show_monitoring_metrics(self):
        """Returns plot of trainloss, valloss, patience and learning rate per epoch."""
        if self.model is None:
            raise ModelStatusError(f'attribute model is None. Model was not correctly initiated')     
        
        if not isinstance(self.monitoring_metrics, pd.DataFrame):
            raise ValueError('no monitoring metrics found')
      
        fig, axes_array = plt.subplots(1, 3, figsize=(24, 4))
        axes: list[Axes]= list(axes_array.flatten())

        # lines: train_loss, val_loss and learning_rate
        sns.lineplot(data = self.monitoring_metrics, x='epoch', y='train_loss',   color=traincolor,   label='Train Loss',         ax=axes[0])
        sns.lineplot(data = self.monitoring_metrics, x='epoch', y='val_loss',     color=valcolor,     label='Validation Loss',    ax=axes[1])
        sns.lineplot(data = self.monitoring_metrics, x='epoch', y='learning_rate',color='black',      label='Learning Rate',      ax=axes[2])

        # Scatter patience epochs and corresponding values
        # Create a mask where patience > 0
        patience_mask = self.monitoring_metrics['patience'] > 0

        # Plot patience epochs as red 'x' markers
        axes[0].scatter(self.monitoring_metrics['epoch'][patience_mask], 
                        self.monitoring_metrics['train_loss'][patience_mask], 
                        color='red', marker='x', label='Patience Epochs')

        axes[1].scatter(self.monitoring_metrics['epoch'][patience_mask], 
                        self.monitoring_metrics['val_loss'][patience_mask], 
                        color='red', marker='x', label='Patience Epochs')

        axes[2].scatter(self.monitoring_metrics['epoch'][patience_mask], 
                        self.monitoring_metrics['learning_rate'][patience_mask], 
                        color='red', marker='x', label='Patience Epochs')   

        for ax in axes:
            ax.grid()
            ax.set_ylabel('loss')
            ax.set_xlabel('epoch')
            ax.legend()

        axes[0].set_title('Training loss')      
        axes[1].set_title('Validation loss')    
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')

        fig.show()
        return fig
