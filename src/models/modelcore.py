import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Dict, List, Literal, Any

import torch


from ..dataloading import EpiDataLoader
from ..metrics.spike_weighted_mse import spike_weighted_mse, mse
from ..dataloading.normalization import reverse_zscore_scaling

class ModelCore:

    """
    Parent class for all models

    DeepLearningModelCore builds off of this by inheritance

    TODO: de-normalize predictions    
    """

    def __init__(self, 
                 dataloader: EpiDataLoader, 
                 name:       Optional[str] = None):
        self.dataloader = dataloader

        self.name       = name if name else "unknown"

        # inherit dataloader's metadata
        self.target_column   = dataloader.target_column
        self.temporal_column = dataloader.temporal_column
        self.id_column       = dataloader.id_column

    def forecast(self):
        """supposed to create the attribute `evaluation_df`"""
        raise NotImplementedError("Each model must implement its own forecast method.")
    
    def _process_predictions(self):
        agg_target = self.evaluation_df.groupby(self.temporal_column)[self.target_column].sum().reset_index()
        agg_preds  = self.evaluation_df.groupby(self.temporal_column)['preds'].sum().reset_index()

        self.aggregated_evaluation_df = pd.merge(agg_target, agg_preds, on=self.temporal_column)
        return self

    
    def show_forecasts(self,
                       id: int = 0):
        """
        previews split and normalized data for a specific node, by default token 0.
        """
        
        # create aggregated predictions
        self._process_predictions()

        traincolor = '#4a90d9'
        valcolor   = "#1b9e77"
        testcolor  = '#d94e4e'

        XYt_train = self.dataloader.XYt_train[self.dataloader.XYt_train[self.id_column] == id]
        XYt_val   = self.dataloader.XYt_val[self.dataloader.XYt_val[self.id_column] == id]
        XYt_test  = self.dataloader.XYt_test[self.dataloader.XYt_test[self.id_column] == id]

        time_axis_train     = XYt_train[self.temporal_column]
        time_axis_val       = XYt_val[self.temporal_column]
        time_axis_test      = XYt_test[self.temporal_column]

        cases_train         = XYt_train[self.target_column]
        cases_val           = XYt_val[self.target_column]
        cases_test          = XYt_test[self.target_column]
        

        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        # plot 1: history of selected ID
        ax1.plot(time_axis_train, cases_train, color = traincolor,  label = 'train')
        ax1.plot(time_axis_val,   cases_val,   color = valcolor,    label = 'val')
        ax1.plot(time_axis_test,  cases_test,  color = testcolor,   label = 'test')
        ax1.set_title(f'history {self.id_column}: {id}')
        ax1.grid()
        ax1.legend()

        # plot 2: predictions vs groundtruth of selected ID
        sns.lineplot(data=self.evaluation_df[self.evaluation_df[self.id_column] == id], x=self.temporal_column, y=self.target_column, color=testcolor, marker = "o", ax=ax2)
        sns.lineplot(data=self.evaluation_df[self.evaluation_df[self.id_column] == id], x=self.temporal_column, y='preds', color=self.model_color, marker = "x",    markeredgecolor='black',  ax=ax2)
        ax2.set_title(f'predictions {self.id_column}: {id}')
        ax2.set_xlabel("")            
        ax2.grid()

        # plot 3: predictions vs groundtruth of aggregation -> over all nodes
        sns.lineplot(data=self.aggregated_evaluation_df, x=self.temporal_column, y=self.target_column, color=testcolor, marker = "o", ax=ax3, label='target')
        sns.lineplot(data=self.aggregated_evaluation_df, x=self.temporal_column, y='preds', color=self.model_color, marker = "x",    markeredgecolor='black', ax=ax3, label='preds')
        ax3.set_title(f"predictions nationally, aggregated {self.name}")
        ax3.grid()
        ax3.legend()
        ax3.autoscale(enable=True)

        plt.tight_layout()
        return self
    
class DeepLearningModelCore(ModelCore):
    """
    Childclass of ModelCore designed to work with GNNs
    Each GNN model inherits from here
    """    
    def __init__(self, dataloader, name: Optional[str] = None):
        super().__init__(dataloader, name)        
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model     = None
        self.optimizer = None
        self.scheduler = None

        self.model_hparams_set = False
        self.global_hparams_set= False

    def _get_optimizer(self, optimizer_name: str, lr: float, optimizer_kwargs: Dict[str, Any]):
        """Factory method to create and return optimizer"""
        optimizer_map = {
            'adam':   torch.optim.Adam,
            'adamw':  torch.optim.AdamW,
            'sgd':    torch.optim.SGD,
            'rmsprop':torch.optim.RMSprop,
            'adagrad':torch.optim.Adagrad,
        }
        
        if optimizer_name.lower() not in optimizer_map:
            raise ValueError(f"Optimizer '{optimizer_name}' not supported. Choose from: {list(optimizer_map.keys())}")
        
        optimizer_class = optimizer_map[optimizer_name.lower()]

        return optimizer_class(self.model.parameters(), lr=lr, **optimizer_kwargs)

    def _get_scheduler(self, scheduler_name: str, optimizer: torch.optim.Optimizer, scheduler_kwargs: Dict[str, Any]):
        """Factory method to create and return scheduler"""
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
        loss_map = {
            'spike_weighted_mse': spike_weighted_mse,
            'mse':                mse,
            'mae':                torch.nn.L1Loss(),
            'huber':              torch.nn.HuberLoss(),
            'smooth_l1':          torch.nn.SmoothL1Loss(),
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
                            loss: str                                  = 'spike_weighted_mse'
                            ):
        
        """
        Prepares model for training using global hyperparameters. 
        Set model hyperparameters first!
        """

        if not self.model_hparams_set:
            raise ValueError(f'set model hyperparameters first! Use model.set_model_hparams()')

        self.global_hparams_set = 'set'
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

    def set_model_hparams(self, **kwargs):
        raise NotImplementedError("deep learning models must have this method")

    def train(self,
              verbose: int = 1,
              dataloader_snapshot: bool = True
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
        """
        if not self.model_hparams_set:
            raise ValueError(f'set model hyperparameters first! Use model.set_model_hparams()')

        if not self.global_hparams_set:
            raise ValueError(f'set global hyperparameters first! Use model.set_global_hparams()')

        if dataloader_snapshot:
            print(f'Dataloader Snapshot: {self.dataloader.dataset_train[0]}')

        self.model.train()
        best_val_loss    = float('inf')
        patience_counter = 0
        best_model_state = None

        L_train    = len(list(self.dataloader.dataset_train))
        L_val      = len(list(self.dataloader.dataset_val))
        
        # each epoch is divided into a training phase and a validation phase
        for epoch in range(self.n_epochs):

            # Training phase
            total_loss = 0
            
            for snapshot in self.dataloader.dataset_train:
                snapshot = snapshot.to(self.device)
                self.optimizer.zero_grad()

                y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
                
                loss  = self.loss(y_hat, snapshot.y)
                
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            
            train_mse = total_loss / L_train
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for snapshot in self.dataloader.dataset_val:
                    snapshot = snapshot.to(self.device)
                    
                    y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
                    loss  = self.loss(y_hat, snapshot.y)
                    
                    val_loss += loss.item()
            
            val_mse = val_loss /L_val
            
            # Switch back to training mode for next epoch
            self.model.train()
            
            # Check if validation loss improved
            val_improved = val_mse < (best_val_loss - self.min_delta)
            
            # Validation improved -> save best model and reset patience
            if val_improved:
                best_val_loss    = val_mse
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                if epoch // verbose:
                    print(f"Epoch {epoch} train loss: {train_mse:.4f}, val loss: {val_mse:.4f} ✓ (new best)")
            # Validation didn't improve -> increment patience
            else:
                patience_counter += 1
                if epoch // verbose:
                    print(f"Epoch {epoch} train loss: {train_mse:.4f}, val loss: {val_mse:.4f} (patience: {patience_counter}/{self.patience})")
                
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

    def forecast(self,  
                 ):
        """
        runs the testing dataloader. Prints the loss and sets attribute evluation_df
        """
        self.model.eval()
        loss = 0
        step = 0

        # Store for analysis
        predictions = []
        labels      = []

        # formatting predictions:
        eval_df     = self.dataloader.test_df
 
        for snapshot in self.dataloader.dataset_test:
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

        tensor_list_cpu       = [t.detach().cpu() for t in predictions]
        stacked               = torch.stack(tensor_list_cpu)            # shape [timepoints, n_nodes]
        df                    = pd.DataFrame(stacked.numpy())  

        # reform predictions dataframe into a long df with timestep_indices to merge with timepoints
        long_df_predictions   = df.reset_index().melt(id_vars='index', 
                                                      var_name=self.id_column, 
                                                      value_name='preds').rename(columns = {'index':f'{self.temporal_column}_idx'})
        
        # the predicted timestamps are all of those, without the first periods
        evaluation_timestamps = eval_df[self.temporal_column].unique()[self.dataloader.periods:]

        timestamps = {}

        for idx, tss in enumerate(evaluation_timestamps):
            timestamps[idx] = tss

        cutperiods                               = eval_df[eval_df[self.temporal_column].isin(evaluation_timestamps)] 
        long_df_predictions[self.temporal_column]= long_df_predictions[f'{self.temporal_column}_idx'].map(timestamps)
        self.evaluation_df                       = pd.merge(cutperiods,long_df_predictions, on = [self.temporal_column,self.id_column])  

        return self              