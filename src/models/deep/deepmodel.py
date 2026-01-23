from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict, Any, Literal

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import pandas as pd
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
# from matplotlib.figure.Figure import Figure
import seaborn as sns

from .modelmanager import ModelManager
from ..base import BaseModel
from .basestrategy import Strategy
from ..utils.loss.losshandler import LossHandler

from ...utils.textformatting import warning_emoji, section, align, checkmark
from ...utils.colors import traincolor, valcolor
from ...dataloading import GraphDataLoaderManager,  DeepDataLoaderManager
from ...plotting import ManagedFigure, convert_managedfigure

class ConflictingDataLoaderManager(Exception):
    def __init__(self, model_name: BaseModel, suggested_strategy: str, dataloadermanager: str):
        super().__init__(f"Conflicting dataloader for {model_name}\nstrategy suggests {suggested_strategy} but dataloadermanager is of type {dataloadermanager}")

class DeepModel(BaseModel, ABC):
    
    def __init__(self, 
                 dataloadermanager:     Union[GraphDataLoaderManager, DeepDataLoaderManager], 
                 strategy,
                 deepfamily:            Literal['vanilla','gnn'],
                 name:                  str,
                 model_color:           Optional[str] = None,                 
                 verbose:               Literal[-1, 0, 1, 2] = -1):

        super().__init__(dataloadermanager=dataloadermanager, name=name, model_color = model_color, verbose = verbose)        

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

        self.model_manager = ModelManager()
        self.monitoring_metrics = None
        self.evaluation_datasets= {}

    def _validate_dataloader_class(self):
        # if strategy suggests vanilla deepmodel:
        if self.deepfamily == 'vanilla':
            if self.dataloadermanager.__class__.__name__ != 'DeepDataLoaderManager':
                raise ConflictingDataLoaderManager(self.name, 'deep-vanilla', self.dataloadermanager.__class__.__name__)
        elif self.deepfamily == 'gnn':
            if self.dataloadermanager.__class__.__name__ != 'GraphDataLoaderManager':
                raise ConflictingDataLoaderManager(self.name, 'deep-graph', self.dataloadermanager.__class__.__name__)    
        else:
            raise ValueError(f'Invalid input for attribute deepfamily found. Should be "vanilla" or "gnn" but received: {self.deepfamily}')        
                

  # model hparams method to be written per model
    @abstractmethod
    def set_model_hparams(self, **kwargs):
        pass

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

    # @convert_managedfigure
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
        Unified training loop that works for both standard and recurrent models.
        
        Parameters
        ----------
        verbose: Literal[0,1,2] = 1:
            how often to return evaluation - updates during training.
            if 0 then only a tqdm is shown. for 1, an update is shown every 10 epochs.
            when 2, every epoch.
        dataloader_snapshot: bool = True
            whether or not to print the __str__ of the DeepDataLoader 
        show_loss: bool = True
            whether or not to plot train and val loss, as well as learning rate per epoch.

        See Also
        --------
        Strategies => src.models.deep.strategies            
        """
        if self.model is None:
            raise ValueError('Please initiate a model')
        
        if self.optimizer is None:
            raise ValueError('no valid optimizer found')

        if self.scheduler is None:
            raise ValueError('no valid scheduler found')
        
        self._check_state(['model_hparams_set', 'global_hparams_set'])
        dataloader_collection = self.dataloadermanager.dataloader_collection

        train_loader = dataloader_collection.train 
        val_loader   = dataloader_collection.val

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
            self.strategy.reset_state()

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
            self.strategy.reset_state()

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

    def forecast(self, dataset: Literal['train','val','test'] = 'test'):
        """
        Unified forecasting loop that works for deep learning models.

        See Also
        --------
        Strategies => src.models.deep.strategies
        basemodel._denorm_predictions()
        """
        self._check_state(['model_hparams_set', 'global_hparams_set', 'trained'])

        if self.model is None:
            raise ValueError('Please initiate a model')
        

        self.model.eval()
        
        predictions = []
        labels      = []

        hh                      = 0
        dataloader_collection   = self.dataloadermanager.dataloader_collection
        eval_df                 = self.dataloadermanager.dataorchestrator.data_final.data
        eval_df                 = eval_df[eval_df[dataset]]

        if dataset == 'train':
            dataloader = dataloader_collection.train
        elif dataset == 'val':
            dataloader = dataloader_collection.val
        elif dataset == 'test':
            dataloader = dataloader_collection.test       

        else:
            raise ValueError(f'dataset must be either "train", "val" or "test"')

        # Reset state before forecasting
        self.strategy.reset_state()
        
        iterator = tqdm(dataloader, desc=f"Forecasting {dataset}") if self.verbose >= 0 else dataloader

        loss = 0
        with torch.no_grad():
            for snapshot in iterator:
                snapshot = snapshot.to(self.device)

                y_hat, loss_val = self.strategy.forecast_step(
                    model   = self.model, 
                    snapshot= snapshot, 
                    loss_fn = self.loss
                )
                loss += loss_val

                # snapshot.y as well as y_hat are a torch.Tensor of shape (num_nodes, horizon_size)
                # labels and predictions are thus a List of these torch.Tensors, one for each timestep.
                labels.append(snapshot.y)
                predictions.append(y_hat)

        loss = loss / len(dataloader)

        setattr(self, f'{dataset}_loss', loss)

        # ======================== FORMAT PREDICTIONS ========================
        tensor_list_cpu                         =    [t.detach().cpu() for t in predictions]   # detach from device
        stacked                                 = torch.stack(tensor_list_cpu)              # concatenates the list of torch.Tensors into one torch.Tensor 
        num_timepoints, n_nodes, horizon_size   = stacked.shape                             # shape is (timestep, num_nodes, horizon_size)

        # check whether these are correct with config!

        reshaped                        = stacked.view(num_timepoints * n_nodes, horizon_size).numpy()  
        timepoints_idx                  = np.repeat(np.arange(num_timepoints), n_nodes)
        nodes                           = np.tile(np.arange(n_nodes), num_timepoints)
        # an object with the same shape as `reshaped` but with the idx to timestamps and node_ids
        index                           = pd.MultiIndex.from_arrays([timepoints_idx, nodes], names=['timestamp_idx', 'node'])

        prediction_columns              = [f'pred_{hh}' for hh in range(horizon_size)]
        prediction_df                   = pd.DataFrame(reshaped, 
                                                       index=index, 
                                                       columns=prediction_columns).reset_index(drop=False)

        # ======================== FORMAT TARGETS ========================
        tensor_list_cpu                 = [t.detach().cpu() for t in labels]
        stacked                         = torch.stack(tensor_list_cpu)   
        num_timepoints, n_nodes, horizon= stacked.shape

        reshaped                        = stacked.view(num_timepoints * n_nodes, horizon).numpy()  

        target_columns                  = [f'target_{hh}' for hh in range(horizon_size)]
        target_df                       = pd.DataFrame(reshaped, 
                                                       index=index,
                                                       columns=target_columns).reset_index(drop=False)
        
        pred_target         = pd.merge(target_df, prediction_df, on=['timestamp_idx','node'])

        idx_offset_train = len(self.dataloadermanager.time_splits[self.dataloadermanager.time_splits['train']])
        idx_offset_val   = len(self.dataloadermanager.time_splits[self.dataloadermanager.time_splits['val']])

        if dataset == 'test':
            timestamp_idx_offset = idx_offset_train + idx_offset_val
        elif dataset == 'val':
            timestamp_idx_offset = idx_offset_train 
        elif dataset == 'train':
            timestamp_idx_offset = 0
        else:
            raise ValueError(f'no valid dataset found')
        
        # matching index with TODAY, not with prediction horizon!
        pred_target['timestamp_idx']    = pred_target['timestamp_idx'] + timestamp_idx_offset
        timestamp_mapping               = self.dataloadermanager.time_splits.reset_index(drop = False)
        pred_target                     = pd.merge(timestamp_mapping[['index','timestamp']], pred_target, left_on = 'index', right_on = 'timestamp_idx').drop(columns = ['index','timestamp_idx'])

        for hh in range(horizon_size):
            horizon_data = pred_target[['timestamp','node',f'pred_{hh}',f'target_{hh}']].rename(columns = {f'pred_{hh}':'pred', f'target_{hh}':'target'})

            if self.loss.loss_name in ['poisson','outbreakpoisson']:
                print('additonal transformation')
                self.predictions.add_horizon_predictions(dataset, horizon_data, hh, additional_transformation=True, transf = 'poisson_sampling', transf_args={'sampling_mode': 'mean'})
            else:
                self.predictions.add_horizon_predictions(dataset, horizon_data, hh)
            if self.verbose > 1:
                print(f"{dataset.capitalize()} loss: {loss:.4f}")
        
        self._update_status('forecasted')
        return self  
        
    def save(self, filename: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Parameters
        ----------
        filename : Optional[str]
            Custom filename. If None, auto-generates one.
            
        Returns
        -------
        str : Path where model was saved
        
        Example
        -------
        >>> model.train()
        >>> model.save('my_best_model')
        """
        return self.model_manager.save(self, filename)
    
    @classmethod
    def load(cls, filepath: str) -> 'DeepModel':
        """
        Load a trained model.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model file
            
        Returns
        -------
        GraphNeuralNetwork : The loaded model
        
        Example
        -------
        >>> model = MyGNN.load('saved_models/my_best_model.pt')
        >>> model.forecast('test')
        """
        manager = ModelManager()
        return manager.load(filepath, cls)

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