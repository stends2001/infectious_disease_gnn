# Fix the imports at the top of your files
from typing import Optional, Dict, List, Literal, Any, Union, cast

from ._basemodel import BaseModel, GNNDataLoader, traincolor, valcolor, testcolor
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, ChebConv, GINConv
from torch_geometric_temporal.nn.recurrent import DCRNN, TGCN, A3TGCN
import pandas as pd
import numpy as np
from typing import Optional, Tuple, cast
from ..metrics.losses import spike_weighted_mse, mse, spike_timing_weighted_mse, temporal_smoothness_loss, spike_detection_loss, spatial_consistency_loss
import seaborn as sns

from ..dataloading.graphdataloader import GraphDataLoader

import torch.optim as optim
from torch.optim.optimizer import Optimizer
# from torch.optim import Optimizer

from torch.optim.lr_scheduler import _LRScheduler
from abc import ABC, abstractmethod

def _check_dataloader_validity(dataloader: 'GNNDataLoader') -> Tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]:

    if dataloader.dataloader_train is None or dataloader.dataloader_val is None or dataloader.dataloader_test is None:
        raise ValueError(f'dataloader invalid. No dataloaders found for train/val/test')

    train = cast(GraphDataLoader, dataloader.dataloader_train)
    val   = cast(GraphDataLoader, dataloader.dataloader_val)
    test  = cast(GraphDataLoader, dataloader.dataloader_test)

    return train, val, test


class DeepModel(BaseModel):
    """
    Childclass of ModelCore designed to work with GNNs
    Each GNN model inherits from here
    """    
    def __init__(self, dataloader: 'GNNDataLoader', name: Optional[str] = None):
        super().__init__(dataloader, name)    

        
        self.gnn_dataloader: 'GNNDataLoader' = dataloader

        self.train_loader, self.val_loader, self.test_loader = _check_dataloader_validity(dataloader)
        

        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[torch.nn.Module]     = None

        self.optimizer: Optional[optim.optimizer.Optimizer]    = None
        self.scheduler: Optional[_LRScheduler] = None

        self.model_hparams_set = False
        self.global_hparams_set= False
        self.model_set         = False

        self.prediction_horizon = dataloader.prediction_horizon

    def _get_optimizer(self, optimizer_name: str, lr: float, optimizer_kwargs: Dict[str, Any]) -> Optimizer:

        if self.model is None:
            raise ValueError('Please initiate a model')

        """Factory method to create and return optimizer"""
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
        if self.model is None:
            raise ValueError('Please initiate a model')
        scheduler_map = {
            'step':        torch.optim.lr_scheduler.StepLR,
            'exponential': torch.optim.lr_scheduler.ExponentialLR,
            'cosine':      torch.optim.lr_scheduler.CosineAnnealingLR,
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

    def _get_loss_function(self, loss_name: str):
        """Factory method to return loss function"""
        if self.model is None:
            raise ValueError('Please initiate a model')

        loss_map = {
            'spike_weighted_mse':   spike_weighted_mse,
            'mse':                  mse,
            'mae':                  torch.nn.L1Loss(),
            'huber':                torch.nn.HuberLoss(),
            'smooth_l1':            torch.nn.SmoothL1Loss(),
            'spatial_consistency':  spatial_consistency_loss,
            'temporal_smoothness' : temporal_smoothness_loss,
            'spike_detection' :     spike_detection_loss
        }
        
        if loss_name.lower() not in loss_map:
            raise ValueError(f"Loss '{loss_name}' not supported. Choose from: {list(loss_map.keys())}")
        
        return loss_map[loss_name.lower()]

    def set_global_hparams(self, 
                            lr: float                                  = 0.001,
                            optimizer: str                             = 'adam',
                            optimizer_kwargs: Optional[Dict[str, Any]] = None,
                            scheduler: Optional[str]                   = 'step',
                            scheduler_kwargs: Optional[Dict[str, Any]] = None,
                            n_epochs: int                              = 5,
                            patience: int                              = 15,
                            min_delta: float                           = 1e-4,
                            loss: Literal['spike_weighted_mse', 'mse',
                                          'mae', 'huber','smooth_L1',
                                          'spatial_consistency',
                                          'spike_detection',
                                          'temporal_smoothness']       = 'spike_weighted_mse'
                            ):
        
        """
        Prepares model for training using global hyperparameters. 
        Set model hyperparameters first!
        """
        if self.model is None:
            raise ValueError('Please initiate a model')

        if not self.model_hparams_set:
            raise ValueError(f'set model hyperparameters first! Use model.set_model_hparams()')

        self.global_hparams_set = True
        # Set training parameters
        self.n_epochs  = n_epochs
        self.patience  = patience
        self.min_delta = min_delta
        self.loss      = self._get_loss_function(loss)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        
        if scheduler_kwargs is None:
            # Default scheduler kwargs for common schedulers
            default_scheduler_kwargs = {
                'step':        {'step_size': 15, 'gamma': 0.8},
                'exponential': {'gamma': 0.95}
            }
            scheduler_kwargs = default_scheduler_kwargs.get(scheduler, {}) if scheduler else {}

        # Create optimizer
        self.optimizer = self._get_optimizer(optimizer, lr, optimizer_kwargs)
        
        # Create scheduler (optional)
        if scheduler:
            self.scheduler = self._get_scheduler(scheduler, self.optimizer, scheduler_kwargs)

        else:
            self.scheduler = None

        return self

    class DeepModel(ABC):
        @abstractmethod
        def set_model_hparams(self, *args, **kwargs):
            """Child classes must implement this with explicit arguments."""
            pass

    def train(self,
              verbose:             Literal[0,1,2] = 1,
              dataloader_snapshot: bool = True,
              show_loss:           bool = True
              ):

        """
        trains model following early stoping checkpoint approach:
        - train -> backward propagation
        - if validation loss improves by min_delta, the model is saved
        - if validation does not improve, we run start the patience iterator: no improvement by min_delta for patience-epochs,
          then stop training
        - either after patience is reached, or after n_epochs, the model with the best validation loss is saved.

        Parameters:
        -----------
        verbose : int = 1
            how frequently performance update is printed
        dataloader_snapshot: bool = True
            whether or not to show the first training dataloader snapshot
        show_loss: bool = True
            whether or not to show the curves of train/val loss per epoch

        See also:
        ---------
        plot_losses
            mehtod to visualise train/val loss per epoch
        """
        if self.model is None:
            raise ValueError('Please initiate a model')
        
        if self.optimizer is None:
            raise ValueError('no valid optimizer found')

        if self.scheduler is None:
            raise ValueError('no valid optimizer found')

        if not self.model_hparams_set:
            raise ValueError(f'set model hyperparameters first! Use model.set_model_hparams()')

        if not self.global_hparams_set:
            raise ValueError(f'set global hyperparameters first! Use model.set_global_hparams()')

        if dataloader_snapshot:
            print(f'Dataloader Snapshot: {self.train_loader[0]}')

        if verbose == 1:
            verbose_loops = list(np.arange(1, self.n_epochs + 1, step=10))
        elif verbose == 2:
            verbose_loops = list(np.arange(1, self.n_epochs + 1))
        else:
            verbose_loops = []

        self.model.train()
        best_val_loss    = float('inf')
        patience_counter = 0
        best_model_state = None

        list_val_loss  =[]
        list_train_loss=[]
        list_patience  =[]

        L_train    = len(list(self.train_loader))
        L_val      = len(list(self.val_loader))
        
        # each epoch is divided into a training phase and a validation phase
        for epoch in range(self.n_epochs):

            # Training phase
            total_loss = 0
            
            for snapshot in self.train_loader:
                snapshot = snapshot.to(self.device)
                self.optimizer.zero_grad()

                y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
                
                loss  = self.loss(y_hat, snapshot.y)
                
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            
            train_mse = total_loss / L_train
            list_train_loss.append(train_mse)
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for snapshot in self.val_loader:
                    snapshot = snapshot.to(self.device)
                    
                    y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
                    loss  = self.loss(y_hat, snapshot.y)
                    
                    val_loss += loss.item()
            
            val_mse = val_loss /L_val
            list_val_loss.append(val_mse)
            
            # Switch back to training mode for next epoch
            self.model.train()
            
            # Check if validation loss improved
            val_improved = val_mse < (best_val_loss - self.min_delta)
            
            # Validation improved -> save best model and reset patience
            if val_improved:
                best_val_loss    = val_mse
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                if epoch in verbose_loops:
                    print(f"Epoch {epoch} train loss: {train_mse:.4f}, val loss: {val_mse:.4f} ✓ (new best)")
                list_patience.append(False)
            # Validation didn't improve -> increment patience
            else:
                patience_counter += 1
                if epoch in verbose_loops:
                    print(f"Epoch {epoch} train loss: {train_mse:.4f}, val loss: {val_mse:.4f} (patience: {patience_counter}/{self.patience})")
                list_patience.append(True)
                # Early stopping if patience exceeded
                if patience_counter >= self.patience:
                    print(f"Early stopping: Validation loss hasn't improved for {self.patience} epochs")
                    # Restore best model
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                        print(f"Restored model from best validation loss: {best_val_loss:.4f}")
                    break
            
            # Step scheduler every epoch (or you could tie it to validation improvement)
            self.scheduler.step()
            self.train_losses   = list_train_loss
            self.val_losses     = list_val_loss
            self.epoch_patience = list_patience

        if show_loss:
            self.plot_losses()

    def plot_losses(self):
        """
        returns plot of train and val losses per epoch, after training model.
        """
        if self.model is None:
            raise ValueError('Please initiate a model')

        epochs          = np.arange(len(self.train_losses))
        patience_epochs = epochs[np.array(self.epoch_patience)]
        patience_train_losses = np.array(self.train_losses)[np.array(self.epoch_patience)]
        patience_val_losses   = np.array(self.val_losses)[np.array(self.epoch_patience)]

        fig, axes = plt.subplots(1,2, figsize = (18,4))
        axes = axes.flatten()

        sns.lineplot(self.train_losses,          color = traincolor, label = 'train loss', ax = axes[0])
        axes[0].scatter(patience_epochs, patience_train_losses, color='red', marker = 'x', label='Patience Epochs')
        sns.lineplot(self.val_losses,            color = valcolor,   label = 'val loss',   ax = axes[1])
        axes[1].scatter(patience_epochs, patience_val_losses, color='red',marker = 'x', label='Patience Epochs')   

        for ax in axes:
            ax.grid()
            ax.set_ylabel('loss')
            ax.set_xlabel('epoch')
            ax.legend()

        axes[0].set_title('Training loss')      
        axes[1].set_title('Validation loss')    
        
        return fig, axes

    def run_snapshot(self, index: int = 0, debug: bool = False):
        """
        Run a single snapshot through the model and print output for validation/debugging.
        """
        if self.model is None:
            raise ValueError('Please initiate a model')

        self.model.eval()  # Set model to eval mode

        # Get snapshot from dataloader
        snapshot = self.train_loader[index]
        snapshot = snapshot.to(self.device)

        with torch.no_grad():
            # Forward pass
            y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_weight, debug=debug)

        # Ground truth
        y_true = snapshot.y.to(self.device)

        # Compute error
        loss = F.mse_loss(y_hat, y_true).item()

        # Print summary
        if debug:
            print(f"\n🧪 Snapshot {index} validation summary:")
            print(f"➡️  Predicted shape: {y_hat.shape}")
            print(f"➡️  Ground truth shape: {y_true.shape}")

        print(f"✅ Snapshot ran successfully")

        return y_hat, y_true

    def forecast(self,
                 dataset: Literal['train','val','test']  = 'test'
                 ):
        """
        runs the testing dataloader. Prints the loss and sets attribute evluation_df
        """
        if self.model is None:
            raise ValueError('Please initiate a model')

        self.model.eval()
        loss = 0
        step = 0

        # Store for analysis
        predictions = []
        labels      = []

        eval_df = self.gnn_dataloader.data['final-horizon'][self.gnn_dataloader.data['final-horizon'][dataset]]

        if dataset == 'train':
            dataloader  = self.train_loader

        elif dataset == 'val':
            dataloader  = self.val_loader

        elif dataset == 'test':
            dataloader  = self.test_loader

        else:
            raise ValueError(f'dataset must be either "train", "val" or "test"')         

        for snapshot in dataloader:
            snapshot = snapshot.to(self.device)
            # Get predictions
            y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
            loss  = loss + self.loss(y_hat, snapshot.y)

            labels.append(snapshot.y)
            predictions.append(y_hat)
            step += 1

        loss = loss / (step+1)
        loss = float(loss)
        print("Test loss: {:.4f}".format(loss))
        self.test_loss = loss

        # Processing prediction format

        tensor_list_cpu       = [t.detach().cpu() for t in predictions]
        stacked               = torch.stack(tensor_list_cpu)   
        num_timepoints, n_nodes, horizon = stacked.shape
        reshaped = stacked.view(num_timepoints * n_nodes, horizon).numpy()  

        # Assume reshaped is your numpy array of shape [num_timepoints * n_nodes, horizon]
        num_timepoints, n_nodes, horizon = stacked.shape

        # Create MultiIndex for rows
        timepoints = np.repeat(np.arange(num_timepoints), n_nodes)  # repeats each timepoint n_nodes times
        nodes = np.tile(np.arange(n_nodes), num_timepoints)         # repeats nodes for all timepoints

        index = pd.MultiIndex.from_arrays([timepoints, nodes], names=['timestamp_idx', 'node'])

        # Create column names for horizon steps
        columns = [f"pred_h{h}" for h in range(horizon)]

        df_pred = pd.DataFrame(reshaped, index=index, columns=columns).reset_index(drop = False)

        # targets

        tensor_list_cpu       = [t.detach().cpu() for t in labels]
        stacked               = torch.stack(tensor_list_cpu)   
        num_timepoints, n_nodes, horizon = stacked.shape
        reshaped = stacked.view(num_timepoints * n_nodes, horizon).numpy()  

        # Assume reshaped is your numpy array of shape [num_timepoints * n_nodes, horizon]
        num_timepoints, n_nodes, horizon = stacked.shape

        # Create MultiIndex for rows
        timepoints = np.repeat(np.arange(num_timepoints), n_nodes)  # repeats each timepoint n_nodes times
        nodes = np.tile(np.arange(n_nodes), num_timepoints)         # repeats nodes for all timepoints

        index = pd.MultiIndex.from_arrays([timepoints, nodes], names=['timestamp_idx', 'node'])

        # Create column names for horizon steps
        columns = [f"incidence_h{h}" for h in range(horizon)]

        df_target = pd.DataFrame(reshaped, index=index, columns=columns).reset_index(drop = False)

        merged = pd.merge(df_pred, df_target, on = ['timestamp_idx','node'])
        merged['timestamp_idx'] = merged['timestamp_idx'] + (self.gnn_dataloader.periods - 1)

        timestamp_map                  = eval_df[['timestamp']].drop_duplicates().reset_index(drop = True).reset_index(drop=False).rename(columns={'index': 'timestamp_idx'})

        merged['timestamp'] = merged['timestamp_idx'].map(dict(zip(timestamp_map['timestamp_idx'], timestamp_map['timestamp'])))

        formatted_eval = eval_df.merge(
            merged[['timestamp', 'node', 'pred_h0', 'pred_h1', 'pred_h2', 'pred_h3']],
            on=['timestamp', 'node'],
            how='left'
        )
        self.evaluation_datasets[dataset] = formatted_eval
        return self