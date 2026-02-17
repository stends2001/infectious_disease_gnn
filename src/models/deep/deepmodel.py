from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any, Literal, Type

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import pandas as pd
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from .modelmanager import ModelManager
from ..base import BaseModel
from ..utils.loss.losshandler import LossHandler
from ...dataloading.dataloaders.deepdataloaders.graphdataloader import GraphDataLoaderManager
from ...dataloading.dataloaders.deepdataloaders.deepdataloader import DeepDataLoaderManager
from ...utils import check_dataset
from ...utils.textformatting import warning_emoji, section, align, checkmark
from ...utils.colors import traincolor, valcolor
from .strategies.basestrategy import Strategy

from .debugging import DeepModelDebuggingError

class ConflictingDataLoaderManager(Exception):
    def __init__(self, model_name: BaseModel, suggested_strategy: str, dataloadermanager: str):
        super().__init__(f"Conflicting dataloader for {model_name}\nstrategy suggests {suggested_strategy} but dataloadermanager is of type {dataloadermanager}")

class ConflictingDataLoaderShape(Exception):
    def __init__(self, message: str):
        super().__init__(f"Conflicting shapes in dataloader\n{message}")    

class DeepModel(BaseModel, ABC):

    _childclasses: Dict[str, Type["DeepModel"]] = {}
    
    def __init__(self, 
                 dataloadermanager:     Union[GraphDataLoaderManager, DeepDataLoaderManager], 
                 strategy,
                 deepfamily:            Literal['vanilla','gnn'],
                 name:                  str,          
                 verbose:               Literal[-1, 0, 1, 2] = -1):

        super().__init__(dataloadermanager=dataloadermanager, name=name, verbose = verbose)        

        self.deepfamily        = deepfamily        
        self.dataloadermanager = dataloadermanager
        self.device            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.device == 'cpu':
            print(f'{warning_emoji} you are connected to cpu, not to gpu')

        self.model:     Optional[torch.nn.Module]           = None                  # to be initiated by childclass
        self.optimizer: Optional[optim.optimizer.Optimizer] = None                  # to be initiated by _get_optimizer
        self.scheduler: Optional[_LRScheduler]              = None                  # to be initiated by _get_scheduler
        self._set_strategy(strategy)
        self._validate_dataloader_class()
        self._validate_dataloader_shapes()

        self.model_manager = ModelManager()
        self.monitoring_metrics = None
        self.evaluation_datasets= {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        DeepModel._childclasses[cls.__name__.lower()] = cls

    def _validate_dataloader_class(self):
        # if strategy suggests vanilla deepmodel:
        if self.deepfamily == 'vanilla':
            if self.dataloadermanager.__class__.__name__ != 'DeepDataLoaderManager':
                raise ConflictingDataLoaderManager(self, 'deep-vanilla', self.dataloadermanager.__class__.__name__)
        elif self.deepfamily == 'gnn':
            if self.dataloadermanager.__class__.__name__ != 'GraphDataLoaderManager':
                raise ConflictingDataLoaderManager(self, 'deep-graph', self.dataloadermanager.__class__.__name__)    
        else:
            raise ValueError(f'Invalid input for attribute deepfamily found. Should be "vanilla" or "gnn" but received: {self.deepfamily}')        

    def _validate_dataloader_shapes(self):
        pass     
      
    def set_model_hparams(self):
        raise NotImplementedError("Child classes of DeepModel must implement set_model_hparams")

    def _get_optimizer(self, optimizer_name: str, lr: float, optimizer_kwargs: Dict[str, Any]) -> Optimizer:
        """Factory method to create and return optimizer"""
        self._check_state(['model_hparams_set'])

        if self.model is None:
            raise ValueError('Please initiate a model')        

        # pylance struggles with torch typing?
        optimizer_map = {
            'adam':    optim.Adam,     # type: ignore
            'adamw':   optim.AdamW,    # type: ignore
            'sgd':     optim.SGD,      # type: ignore
            'rmsprop': optim.RMSprop,  # type: ignore
            'adagrad': optim.Adagrad,  # type: ignore
        }
        
        if optimizer_name.lower() not in optimizer_map:
            raise ValueError(f"Optimizer '{optimizer_name}' not supported. Choose from: {list(optimizer_map.keys())}")
        
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
            raise ValueError(f"Scheduler '{scheduler_name}' not supported. Choose from: {list(scheduler_map.keys())}")
        
        scheduler_class = scheduler_map[scheduler_name.lower()]
        return scheduler_class(optimizer, **scheduler_kwargs)

    def _set_strategy(self, strategy: Strategy) -> None:
        """Allow subclasses to specify their strategy"""
        self.strategy = strategy

    def set_global_hparams(self, 
                            lr: float = 0.001,
                            n_epochs: int = 5,
                            patience: int = 15,
                            min_delta: float = 1e-4,                            
                            optimizer: str = 'adam',
                            optimizer_kwargs: Optional[Dict[str, Any]] = None,
                            scheduler: Optional[str] = 'step',
                            scheduler_kwargs: Optional[Dict[str, Any]] = None,
                            loss: str = 'mse',
                            loss_kwargs: Optional[Dict[str, Any]] = None                            
                            ):
        """Prepares model for training using global hyperparameters."""
        self._check_state(['model_hparams_set'])

        global_params_config = {
            'lr'                : lr,
            'n_epochs'          : n_epochs,
            'patience'          : patience,
            'min_delta'         : min_delta,                       
            'optimizer'         : optimizer,
            'optimizer_kwargs'  : optimizer_kwargs,
            'scheduler'         : scheduler,
            'scheduler_kwargs'  : scheduler_kwargs,
            'loss'              : loss,
            'loss_kwargs'       : loss_kwargs
        }
        
        self.global_hparams_set = True
        self.n_epochs           = n_epochs
        self.patience           = patience
        self.min_delta          = min_delta
        self.loss               = LossHandler(loss, loss_kwargs=loss_kwargs)  

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer = self._get_optimizer(optimizer, lr, optimizer_kwargs)
        
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

        self.config_info['global_hparams']  = global_params_config
        self._update_status('global_hparams_set')
        return self

    def show_monitoring_metrics(self):
        """Returns plot of trainloss, valloss, patience and learning rate per epoch."""
        if self.model is None:
            raise ValueError('Please initiate a model')
        
        if not isinstance(self.monitoring_metrics, pd.DataFrame):
            raise ValueError('no monitoring metrics found')

        fig, axes   = plt.subplots(1, 3, figsize=(24, 4))
        axes        = axes.flatten()

        # lines: train_loss, val_loss and learning_rate
        sns.lineplot(data=self.monitoring_metrics, x='epoch', y='train_loss',   color=traincolor,   label='Train Loss',         ax=axes[0])
        sns.lineplot(data=self.monitoring_metrics, x='epoch', y='val_loss',     color=valcolor,     label='Validation Loss',    ax=axes[1])
        sns.lineplot(data=self.monitoring_metrics, x='epoch', y='learning_rate',color='black',      label='Learning Rate',      ax=axes[2])

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

    def train(self):
        """ 
        """
        if self.model is None:
            raise ValueError('Please initiate a model')
        
        if self.optimizer is None:
            raise ValueError('no valid optimizer found')

        if self.scheduler is None:
            raise ValueError('no valid scheduler found')
        
        self._check_state(['model_hparams_set', 'global_hparams_set'])

        train_loader = self.dataloadermanager.dataloader_train 
        val_loader   = self.dataloadermanager.dataloader_val 

        # print dataloader snapshot
        if self.verbose>=2:
            print(f'Dataloader Snapshot: {train_loader[0]}')        

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

        self.model.train()
        best_val_loss       = float('inf')
        patience_counter    = 0
        best_model_state    = None

        list_val_loss       = []
        list_train_loss     = []
        list_patience       = []
        list_lr             = []

        L_train             = len(list(train_loader))
        L_val               = len(list(val_loader))
    
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
            # gives three digits as print no matter what
            verbose_statement_basis = f"Epoch {num_epoch:03d} train loss: {train_loss:.4f}, val loss: {val_loss:.4f}"
            
            # Check if validation loss improved
            val_improved = val_loss < (best_val_loss - self.min_delta)

            # if so => save best model
            if val_improved:
                best_val_loss   = val_loss
                patience_counter= 0
                best_model_state= self.model.state_dict().copy()
                
                verbose_statement = verbose_statement_basis + f" {checkmark} (new best)"
                    
                list_patience.append(False)

            else:
                patience_counter += 1
                verbose_statement = verbose_statement_basis + f" (patience: {patience_counter}/{self.patience})"
                list_patience.append(True)

            if patience_counter >= self.patience:
                print(f"Early stopping: Validation loss hasn't improved for {self.patience} epochs")

                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                    print(f"Restored model from best validation loss: {best_val_loss:.4f}")

                break              

            if num_epoch in verbose_loops:
                print(verbose_statement)

            # Step scheduler => scheduler.step requires val loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss) # type: ignore
            # for other schedulers, no arguments required
            else:
                self.scheduler.step()

            new_lr = self.optimizer.param_groups[0]['lr']

            if current_lr != new_lr and self.verbose >=1:
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) and not isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    print(f'lr has been updated from {current_lr:.2e} to {new_lr:.2e}')        
            
        self.monitoring_metrics = pd.DataFrame({'train_loss'    : list_train_loss,
                                                'val_loss'      : list_val_loss,
                                                'patience'      : list_patience,
                                                'learning_rate' : list_lr}).reset_index(names='epoch')
        
        self.monitoring_metrics['epoch'] = self.monitoring_metrics['epoch'] + 1

        if self.verbose >=1:
            self.show_monitoring_metrics()

        self._update_status('trained')

    @check_dataset()
    def forecast(self, dataset: Literal['train','val','test'] = 'test'):
        """
        Unified forecasting loop for deep learning models.
        
        Returns predictions aligned with the correct timestamps using the 
        pre-computed time_splits from the dataloader manager.
        """
        self._check_state(['model_hparams_set', 'global_hparams_set', 'trained'])

        if self.model is None:
            raise ValueError('Please initiate a model')

        self.model.eval()
        
        # Get the appropriate dataloader
        dataloader = getattr(self.dataloadermanager, f'dataloader_{dataset}')
        
        # Reset state before forecasting
        self.strategy.reset_state_dataset()
        
        iterator = tqdm(dataloader, desc=f"Forecasting {dataset}") if self.verbose >= 0 else dataloader

        all_predictions = []
        all_targets = []
        total_loss = 0
        
        with torch.no_grad():
            for snapshot in iterator:
                snapshot = snapshot.to(self.device)

                y_hat, loss_val = self.strategy.forecast_step(
                    model=self.model, 
                    snapshot=snapshot, 
                    loss_fn=self.loss
                )
                total_loss += loss_val

                # Collect predictions and targets
                all_predictions.append(y_hat.detach().cpu())
                all_targets.append(snapshot.y.detach().cpu())

        avg_loss = total_loss / len(dataloader)
        setattr(self, f'{dataset}_loss', avg_loss)

        # Stack all predictions and targets
        # Shape: [num_sequences, num_nodes, horizon_size]
        predictions_tensor = torch.stack(all_predictions)
        targets_tensor = torch.stack(all_targets)
        
        num_sequences, num_nodes, horizon_size = predictions_tensor.shape
        
        # Create the results dataframe
        results = self._format_forecast_results(
            predictions=predictions_tensor,
            targets=targets_tensor,
            dataset=dataset,
            num_sequences=num_sequences,
            num_nodes=num_nodes,
            horizon_size=horizon_size
        )
        
        # Store predictions for each horizon
        for hh in range(horizon_size):
            horizon_data = results[
                [self.epiconfig.temporal_column, self.epiconfig.id_column, f'pred_{hh}', f'target_{hh}']
            ].rename(columns={f'pred_{hh}': 'pred', f'target_{hh}': 'target'})
            
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

    def _format_forecast_results(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        dataset: str,
        num_sequences: int,
        num_nodes: int,
        horizon_size: int
    ) -> pd.DataFrame:
        """
        Format predictions and targets into a dataframe with correct timestamps.
        
        Uses the time_splits dataframe to align sequence indices with actual timestamps.
        """
        # Reshape to long format: [num_sequences * num_nodes, horizon_size]
        pred_reshaped = predictions.view(num_sequences * num_nodes, horizon_size).numpy()
        target_reshaped = targets.view(num_sequences * num_nodes, horizon_size).numpy()
        
        # Create index arrays
        sequence_idx = np.repeat(np.arange(num_sequences), num_nodes)
        node_idx = np.tile(np.arange(num_nodes), num_sequences)
        
        # Get the timestamps for this dataset
        dataset_time_splits = self.dataloadermanager.time_splits[
            self.dataloadermanager.time_splits[dataset]
        ].reset_index()
        
        # Sequence indices correspond to rows in the filtered time_splits
        # (after accounting for sequence_length lookback)
        sequence_length = self.dataloadermanager.dataorchestrator.config.sequence_length
        
        timestamp_indices = sequence_idx 
        
        # Map to actual timestamps
        timestamps = dataset_time_splits.loc[timestamp_indices, self.epiconfig.temporal_column].values
        
        # Build the results dataframe
        results = pd.DataFrame({
            self.epiconfig.temporal_column: timestamps,
            self.epiconfig.id_column: node_idx
        })
        
        # Add prediction columns
        for hh in range(horizon_size):
            results[f'pred_{hh}'] = pred_reshaped[:, hh]
            results[f'target_{hh}'] = target_reshaped[:, hh]
        
        return results

    def debug(self):
        if self.model is None:
            raise ValueError('Please initiate a model')

        self._check_state(['model_hparams_set'])

        train_sample = self.dataloadermanager.dataloader_train[0].to(self.device)
        y_hat , report = self.strategy.debug(self.model, train_sample)

        y_hat = y_hat.detach().cpu()
        y     = train_sample.y.detach().cpu()
        report.validate()
        if y_hat.shape != y.shape:
            raise DeepModelDebuggingError(f"incompatible prediction shape: y_hat [{y_hat.shape}], y [{y.shape}]")
        
    
    def save(self, filename: Optional[str] = None) -> None:
        """
        Save the trained model.
        """
        self._check_state(['trained'])

        self.model_manager.save(self, filename)
    
    @classmethod
    def load(cls, model_name: str) -> 'DeepModel':
        """
        Load a trained model.

        NOTE: classmethod 
        """
        manager = ModelManager()
        return manager.load(model_name, cls)

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