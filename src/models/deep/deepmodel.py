from typing import Union, Dict, Type, Literal, Optional, Any, Tuple, List, TypeVar, Generic

from wcwidth import wcswidth

import torch 
from torch import Tensor as Tensor
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from matplotlib.axes import Axes
import matplotlib.pyplot as plt 
import seaborn as sns

import pandas as pd 
import numpy as np

from tqdm import tqdm

from .modelmanager import ModelManager

from .strategies.basestrategy import Strategy

from ..utils.loss.losshandler import LossHandler
from ..base import BaseModel
from ..issues import DeviceWarning, InvalidLossError, InvalidOPtimizerError, InvalidSchedulerError, ModelStatusError

from ...dataloading import GraphDataLoaderManager, DeepDataLoaderManager
from ...utils import checkmark, traincolor, valcolor, align, section, check_dataset
 
from .issues import UnexpectedDataShape, InconsistentDataShape

class DeepModel(BaseModel[Union[GraphDataLoaderManager, DeepDataLoaderManager]]):

    _childclasses: Dict[str, Type["DeepModel"]] = {}
        
    def __init__(self, 
                 dataloadermanager:     Union[GraphDataLoaderManager, DeepDataLoaderManager], 
                 strategy:              Strategy,
                 name:                  str,          
                 verbose:               Literal[-1, 0, 1, 2] = -1):

        super().__init__(dataloadermanager = dataloadermanager, name = name, verbose = verbose)        

        self.model:     Optional[torch.nn.Module]           = None                  # to be initiated by childclass
        self.optimizer: Optional[optim.optimizer.Optimizer] = None                  # to be initiated by _get_optimizer
        self.scheduler: Optional[_LRScheduler]              = None                  # to be initiated by _get_scheduler
        self.model_manager                                  = ModelManager()
        self.monitoring_metrics                             = None
        self.evaluation_datasets                            = {}
        self._set_device()
        self._set_strategy(strategy)

    # ======= DUNDER ======= #

    # __init_subclass__ is run when a subclass is iniated
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        DeepModel._childclasses[cls.__name__.lower()] = cls
    
    # ======= MAIN METHODS =========== #
    def set_global_hparams(self, 
                           lr:              float           = 0.001,
                           n_epochs:        int             = 5,
                           patience:        int             = 15,
                           min_delta:       float           = 1e-4,                            
                           optimizer:       str             = 'adam',
                           loss:            str             = 'mse',                           
                           scheduler:       Optional[str]   = 'step',
                           # kwargs
                           optimizer_kwargs:Optional[Dict[str, Any]] = None,                           
                           scheduler_kwargs:Optional[Dict[str, Any]] = None,
                           loss_kwargs:     Optional[Dict[str, Any]] = None                            
                           ):
        """
        Prepares model for training using global hyperparameters
        
        Parameters
        ---------

        """
        self._check_state(['model_hparams_set'])

        global_params_config = {
            'lr'                : lr,
            'n_epochs'          : n_epochs,
            'patience'          : patience,
            'min_delta'         : min_delta,                       
            'optimizer'         : optimizer,
            'loss'              : loss,
            'scheduler'         : scheduler,

            'optimizer_kwargs'  : optimizer_kwargs,
            'scheduler_kwargs'  : scheduler_kwargs,
            'loss_kwargs'       : loss_kwargs
        }
        
        # ==== CONSTANTS ===== #
        self.global_hparams_set = True
        self.n_epochs           = n_epochs
        self.patience           = patience
        self.min_delta          = min_delta

        # ==== LOSS ==== #
        if loss == 'pinball':
            if loss_kwargs is None:
                loss_kwargs = {}

            if 'quantiles' not in loss_kwargs.keys():
                loss_kwargs['quantiles'] = self.epiconfig.quantiles

        self.loss       = LossHandler(loss, loss_kwargs = loss_kwargs)  

        # ==== OPTIMIZER ==== #
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer = self._get_optimizer(optimizer, lr, optimizer_kwargs)
        
        # ==== SCHEDULER ====== #
        if scheduler_kwargs is None:
            default_scheduler_kwargs = {
                'step':        {'step_size': 15, 'gamma': 0.8},
                'exponential': {'gamma': 0.95},
                'cosine':      {'T_max': 50},
                'cosine_warm': {'T_0': 10, 'T_mult': 2},                
                'plateau':     {'mode': 'min', 'factor': 0.5, 'patience': 10, 'verbose': True}
            }
            scheduler_kwargs = default_scheduler_kwargs.get(scheduler, {}) if scheduler else {}

        if scheduler:
            self.scheduler = self._get_scheduler(scheduler, self.optimizer, scheduler_kwargs)

        else:
            self.scheduler = None

        self._validate_global_hparams()
        self.config_info['global_hparams']  = global_params_config
        self._update_status('global_hparams_set')
        return self

    def train(self):
        """ 
        """
        self._check_state(['model_hparams_set', 'global_hparams_set'])

        if self.model is None:
            raise ModelStatusError(f'attribute model is None. Model was not correctly initiated')   
        
        if self.optimizer is None:
            raise ModelStatusError(f'attribute optimizer is None. global hparams were not correctly set.')   

        if self.scheduler is None:
            raise ModelStatusError(f'attribute scheduler is None. global hparams were not correctly set.')   
        
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

            if patience_counter >= self.patience:
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

    def debug(self):
        if self.model is None:
            raise ValueError('Please initiate a model')

        self._check_state(['model_hparams_set'])

        train_sample = self.dataloadermanager.dataloader_train[0].to(self.device)
        y_hat , report = self.strategy.debug(self.model, train_sample)

        y_hat = y_hat.detach().cpu()
        y     = train_sample.y.detach().cpu()
        report.validate()
        # if y_hat.shape != y.shape:
        #     raise DeepModelDebuggingError(f"incompatible prediction shape: y_hat [{y_hat.shape}], y [{y.shape}]")
        
    def save(self, filename: Optional[str] = None) -> None:
        """
        Save the trained model.
        """
        self._check_state(['trained'])

        self.model_manager.save(self, filename)
    
    @check_dataset()
    def forecast(self, dataset: Literal['train','val','test'] = 'test'):
        
        self._check_state(['model_hparams_set', 'global_hparams_set', 'trained'])

        if self.model is None:
            raise ModelStatusError(f'attribute model is None. Model was not correctly initiated')   

        self.model.eval()

        # stupdly keep pylance happy
        if dataset == 'train':
            dataloader = self.dataloadermanager.dataloader_train
        elif dataset == 'val':
            dataloader = self.dataloadermanager.dataloader_val 
        elif dataset == 'test':
            dataloader = self.dataloadermanager.dataloader_test
        else:
            # will never be raised given the check_dataset decorator
            raise ValueError(f'cant find dataset {dataset}')

        self.strategy.reset_state_dataset()
        iterator = tqdm(dataloader, desc=f"Forecasting {dataset}") if self.verbose >= 0 else dataloader

        raw_predictions: List[Tensor]   = []
        raw_targets: List[Tensor]       = []
        
        total_loss = 0

        expected_shape_yhat = [self.dataloadermanager.dataorchestrator.data_context.num_nodes, self.epiconfig.horizon_size, max(1,self.epiconfig._num_quantiles)]

        with torch.no_grad():
            for idx, snapshot in enumerate(iterator):
                snapshot = snapshot.to(self.device)
                y_hat, loss_val = self.strategy.forecast_step(
                    model=self.model, 
                    snapshot=snapshot, 
                    loss_fn=self.loss
                )
                total_loss += loss_val

                if not idx:
                    if list(y_hat.shape) != expected_shape_yhat:
                        raise UnexpectedDataShape(f'{list(y_hat.shape)}', f'{expected_shape_yhat}', "stacked yhat forecasting snapshot 0")

                raw_predictions.append(y_hat.detach().cpu())
                raw_targets.append(snapshot.y.detach().cpu())

        avg_loss = total_loss / len(dataloader)
        setattr(self, f'{dataset}_loss', avg_loss)

        # =========== SHAPE CHECK 1 ============= #
        # at this point, raw_predictions is a List of len [timestamps].
        # at each idx, there is a Tensor with shape [num_nodes, horizon_size, quantiles].
        # Since quantiles don't play a role in the target, those do not have that final dim
        # We're now removing the list-ness and stack that to a new dimension. The tensors therefore
        # get 3 (target) and 4 (predictions) dimensions.

        predictions_tensor  = torch.stack(raw_predictions)
        targets_tensor      = torch.stack(raw_targets)

        expected_shape_predictions  = [len(dataloader), self.dataloadermanager.dataorchestrator.data_context.num_nodes, self.epiconfig.horizon_size, max(1,self.epiconfig._num_quantiles)]
        expected_shape_targets      = [len(dataloader), self.dataloadermanager.dataorchestrator.data_context.num_nodes, self.epiconfig.horizon_size]        

        received_shape_predictions  = list(predictions_tensor.shape)
        received_shape_targets      = list(targets_tensor.shape)

        if expected_shape_predictions != received_shape_predictions:
            raise UnexpectedDataShape(f'{received_shape_predictions}', f'{expected_shape_predictions}', "stacked raw predictions")

        if expected_shape_targets != received_shape_targets:
            raise UnexpectedDataShape(f'{received_shape_targets}', f'{expected_shape_targets}', "stacked raw targets")

        num_timesteps, num_nodes, horizon_size, num_quantiles = predictions_tensor.shape

        # Get the quantile column names from the column registry
        # These are the names PredictionManager expects, e.g. ['q_0.1', 'q_0.5', 'q_0.9']
        # For num_quantiles == 0 (point forecast), this degenerates to ['pred']
        if self.epiconfig._num_quantiles == 0:
            pred_col_names = ['pred']                  
        else:
            pred_col_names = [c for c in self.predictions.column_registration.pred_columns if c != 'pred']    
        
        results = self._format_forecast_results(
            predictions     = predictions_tensor,
            targets         = targets_tensor,
            dataset         = dataset,
            num_timesteps   = num_timesteps,
            num_nodes       = num_nodes,
            horizon_size    = horizon_size,
            pred_col_names  = pred_col_names,
        )

        for hh in range(horizon_size):
            # Select the columns for this horizon: timestamp, id, all pred_cols, target
            horizon_cols = (
                [self.epiconfig.temporal_column, self.epiconfig.id_column]
                + [f'{col}_{hh}' for col in pred_col_names]
                + [f'target_{hh}']
            )
            horizon_data = results[horizon_cols].rename(
                columns={
                    **{f'{col}_{hh}': col for col in pred_col_names},
                    f'target_{hh}': 'target',
                }
            )

            if self.loss.loss_name in ['poisson', 'outbreakpoisson']:
                self.predictions.add_horizon_predictions(
                    dataset, horizon_data, hh,
                    additional_transformation=True,
                    transf='poisson_sampling',
                    transf_args={'sampling_mode': 'mean'}
                )
            else:
                self.predictions.add_horizon_predictions(dataset, horizon_data, hh)

        if self.verbose > 1:
            print(f"{dataset.capitalize()} loss: {avg_loss:.4f}")

        self._update_status('forecasted')
        return self

    @classmethod
    def load(cls, model_name: str) -> 'DeepModel':
        """
        Load a trained model.

        NOTE: classmethod 
        """
        manager = ModelManager()
        return manager.load(model_name, cls)

    # ======= TO BE IMPLEMENTED BY SUBCLASSES =========== #   
    def set_model_hparams(self):
        raise NotImplementedError("Subclass of DeepModel must implement set_model_hparams")

    # ======= HELPERS ======= #
    def _set_device(self):
        """sets attribute device"""
        self.device            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.device == 'cpu':
            w = DeviceWarning('device found is CPU')
            print(w)

    def _validate_global_hparams(self):        
        if self.epiconfig.quantiles:
            if self.loss.loss_name != 'pinball':
                raise ValueError('quantiles given to DeepModel, yet loss is not pinball.')

    def _get_optimizer(self, 
                       optimizer_name:  str, 
                       lr:              float, 
                       optimizer_kwargs:Dict[str, Any]) -> Optimizer:
        """Factory method to create and return optimizer"""
        self._check_state(['model_hparams_set'])

        if self.model is None:
            raise ModelStatusError(f'attribute model is None. Model was not correctly initiated')        

        # pylance struggles with torch typing?
        optimizer_map = {
            'adam':    optim.Adam,     # type: ignore
            'adamw':   optim.AdamW,    # type: ignore
            'sgd':     optim.SGD,      # type: ignore
            'rmsprop': optim.RMSprop,  # type: ignore
            'adagrad': optim.Adagrad,  # type: ignore
        }
        
        if optimizer_name.lower() not in optimizer_map:
            raise InvalidOPtimizerError(optimizer_name, list(optimizer_map.keys()))
        
        optimizer_class = optimizer_map[optimizer_name.lower()]

        return optimizer_class(self.model.parameters(), lr=lr, **optimizer_kwargs)

    def _get_scheduler(self, scheduler_name: str, optimizer: Optimizer, scheduler_kwargs: Dict[str, Any]) -> _LRScheduler:
        """Factory method to create and return scheduler"""
        
        self._check_state(['model_hparams_set'])
        
        scheduler_map = {
            'step':        torch.optim.lr_scheduler.StepLR,
            'exponential': torch.optim.lr_scheduler.ExponentialLR,
            'cosine':      torch.optim.lr_scheduler.CosineAnnealingLR,
            'cosine_warm': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,            
            'plateau':     torch.optim.lr_scheduler.ReduceLROnPlateau,
            'cyclic':      torch.optim.lr_scheduler.CyclicLR,
            'onecycle':    torch.optim.lr_scheduler.OneCycleLR,
            'multistep':   torch.optim.lr_scheduler.MultiStepLR,
            'lambda':      torch.optim.lr_scheduler.LambdaLR,
        }
        
        if scheduler_name.lower() not in scheduler_map:
            raise InvalidSchedulerError(scheduler_name, list(scheduler_map.keys()))
        
        scheduler_class = scheduler_map[scheduler_name.lower()]
        return scheduler_class(optimizer, **scheduler_kwargs)

    def _set_strategy(self, strategy: Strategy):
        """Allow subclasses to specify their strategy"""
        self.strategy = strategy

    def _return_verbose_iter(self) -> Tuple[list, Union[range, tqdm]]:
        # print dataloader snapshot
        if self.verbose>=2:
            print(f'Dataloader Snapshot: {self.dataloadermanager.dataloader_train [0]}')        

        # determine verbose - loops (which loops to return evaluation metric)
        if self.verbose >= 2:
            verbose_loops   = list(np.arange(1, self.n_epochs + 1))
            epoch_iter      = range(self.n_epochs)

        elif self.verbose >= 1:
            verbose_loops   = list(np.arange(1, self.n_epochs + 1, step=10))
            epoch_iter      = range(self.n_epochs)

        elif self.verbose < 0:
            verbose_loops   = []
            epoch_iter      = range(self.n_epochs)

        else:
            verbose_loops   = []
            epoch_iter      = tqdm(range(self.n_epochs), desc="Training epochs") # if no verbose, just a tqdm     

        return verbose_loops, epoch_iter   

    def _return_verbose_line(self, epoch=None, train_loss=None, val_loss=None, new_best=None, patience=None, lr_updated= None):
        columns = ["epoch", "train loss", "val loss", "new best", "patience"]
        columns = [col.upper() for col in columns]

        widths = [5, 10, 10, 8, 9]
        alignments = ["^", "^", "^", "^", "^"]

        def fmt(value, width, align):
            return f"{value:{align}{width}}"

        def make_row(values):
            return "| " + " | ".join(
                fmt(v, w, a) for v, w, a in zip(values, widths, alignments)
            ) + " |"

        # total table width = pipes + spaces + column widths
        total_width = sum(widths) + 3 * len(widths) + 1
        separator = "─" * total_width

        if any(x is not None for x in (epoch, train_loss, val_loss, new_best, patience, lr_updated)):
            row_values = [
                f"{epoch:03d}" if epoch is not None else "",
                f"{train_loss:.4f}" if train_loss is not None else "",
                f"{val_loss:.4f}" if val_loss is not None else "",
                f"{new_best}" if new_best is not None else "",
                f"{patience}" if patience else "",
            ]
            line = make_row(row_values)
            if lr_updated:
                line += " *"
            print(line)
        else:
            print(separator)
            print(make_row(columns))            

    def _format_forecast_results(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        dataset: str,
        num_timesteps: int,
        num_nodes: int,
        horizon_size: int,
        pred_col_names: list[str],
    ) -> pd.DataFrame:
        """
        Format predictions into a flat DataFrame aligned with correct timestamps.
        Handles both point forecasts (num_quantiles=1) and quantile forecasts.

        predictions shape: [num_timesteps, num_nodes, horizon_size, num_quantiles]
        targets shape:     [num_timesteps, num_nodes, horizon_size]
        """
        num_quantiles = len(pred_col_names)

        # Reshape: [num_sequences * num_nodes, horizon_size, num_quantiles]
        pred_reshaped = (
            predictions
            .view(num_timesteps * num_nodes, horizon_size, num_quantiles)
            .numpy()
        )
        # Reshape: [num_sequences * num_nodes, horizon_size]
        target_reshaped = targets.view(num_timesteps * num_nodes, horizon_size).numpy()

        # Index arrays — np.repeat/tile is correct here, no issue
        sequence_idx = np.repeat(np.arange(num_timesteps), num_nodes)
        node_idx     = np.tile(np.arange(num_nodes), num_timesteps)

        dataset_time_splits = self.dataloadermanager.time_splits[
            self.dataloadermanager.time_splits[dataset]
        ].reset_index(drop=True)  # ⚠️ drop=True to avoid keeping old integer index as a column

        timestamps = dataset_time_splits.loc[sequence_idx, self.epiconfig.temporal_column].values

        results = pd.DataFrame({
            self.epiconfig.temporal_column: timestamps,
            self.epiconfig.id_column: node_idx,
        })

        # One column per horizon per quantile: e.g. q_0.1_0, q_0.5_0, q_0.9_0, ...
        for hh in range(horizon_size):
            for qq, col_name in enumerate(pred_col_names):
                results[f'{col_name}_{hh}'] = pred_reshaped[:, hh, qq]
            results[f'target_{hh}'] = target_reshaped[:, hh]

        return results
    
    def __str__(self):
        # Calculate width
        all_keys = (
            ['model name', 'model class'] +
            list(self._state.keys()) +
            list(self.config_info.get('model_hparams', {}).keys()) +
            list(self.config_info.get('global_hparams', {}).keys())
        )
        width = max(len(k) for k in all_keys) if all_keys else 20
        
        # Build output
        lines = ['<DeepModel(']
        lines.append(align('model name', self.name, width))
        lines.append(align('model class', self.model_class, width))
        lines.append('')
        
        # Status section
        status_items = {k: "✓" if v else "✗" for k, v in self._state.items()}
        lines.extend(section('status', status_items, width))
        lines.append('')
        
        # Forecasts section
        lines.extend(section('forecasts', {'forecasted': list(self.evaluation_datasets.keys())}, width))
        lines.append('')
        
        # Model hparams
        model_hparams = dict(self.config_info.get('model_hparams', {}))
        model_hparams['strategy'] = self.strategy
        lines.extend(section('model hparams', model_hparams, width))
        lines.append('')
        
        # Global hparams
        lines.extend(section('global hparams', self.config_info.get('global_hparams', {}), width))
        
        lines.append(')>')
        
        return '\n'.join(lines)