import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Dict, List, Literal, Any

import torch


from ..dataloading import EpiDataLoader
from ..metrics.losses import spike_weighted_mse, mse, spike_detection_loss, temporal_smoothness_loss, spatial_consistency_loss
from ..dataloading.normalization import reverse_zscore_scaling, reverse_log

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

        self.traincolor = '#4a90d9'
        self.valcolor   = "#1b9e77"
        self.testcolor  = '#d94e4e'        

    def forecast(self):
        """supposed to create the attribute `evaluation_df`"""
        raise NotImplementedError("Each model must implement its own forecast method.")
    
    def _process_predictions(self):

        train_normalized = self.dataloader.XYt_train
        train_denorm     = reverse_zscore_scaling(train_normalized, params = self.dataloader.norm_params['params'])
        train_denorm     = reverse_log(train_denorm, self.dataloader.logged_params)

        val_normalized   = self.dataloader.XYt_val
        val_denorm       = reverse_zscore_scaling(val_normalized, params = self.dataloader.norm_params['params'])
        val_denorm       = reverse_log(val_denorm, self.dataloader.logged_params)

        test_normalized  = self.dataloader.XYt_test
        test_denorm      = reverse_zscore_scaling(test_normalized, params = self.dataloader.norm_params['params'])
        test_denorm      = reverse_log(test_denorm, self.dataloader.logged_params)

        preds_normalized = self.evaluation_df
        preds_denorm      = reverse_zscore_scaling(preds_normalized, params = self.dataloader.norm_params['params'])
        preds_denorm      = reverse_log(preds_denorm, self.dataloader.logged_params)

        agg_target = preds_normalized.groupby(self.temporal_column)[self.target_column].sum().reset_index()
        agg_preds  = preds_normalized.groupby(self.temporal_column)['preds'].sum().reset_index()
        self.aggregated_evaluation_df = pd.merge(agg_target, agg_preds, on=self.temporal_column)

        agg_target_denorm = preds_denorm.groupby(self.temporal_column)[self.target_column].sum().reset_index()
        agg_preds_denorm  = preds_denorm.groupby(self.temporal_column)['preds'].sum().reset_index()
        self.aggregated_evaluation_df_denorm = pd.merge(agg_target_denorm, agg_preds_denorm, on=self.temporal_column)        
        
        self.XYt_train_denorm =train_denorm
        self.XYt_val_denorm   =val_denorm
        self.XYt_test_denorm  =test_denorm
        self.preds_denorm     =preds_denorm

        return self

    
    def show_forecasts(self,
                       id: int = 0,
                       norm: bool = False):
        """
        previews split and normalized data for a specific node, by default token 0.
        """
        
        # create aggregated predictions
        self._process_predictions()

        traincolor = '#4a90d9'
        valcolor   = "#1b9e77"
        testcolor  = '#d94e4e'

        if norm:
            XYt_train = self.dataloader.XYt_train[self.dataloader.XYt_train[self.id_column] == id]
            XYt_val   = self.dataloader.XYt_val[self.dataloader.XYt_val[self.id_column] == id]
            XYt_test  = self.dataloader.XYt_test[self.dataloader.XYt_test[self.id_column] == id]
            
            preds = self.evaluation_df[self.preds_denorm[self.id_column] == id]
            aggr = self.aggregated_evaluation_df


        else:
            XYt_train = self.XYt_train_denorm[self.XYt_train_denorm[self.id_column] == id]
            XYt_val   = self.XYt_val_denorm[self.XYt_val_denorm[self.id_column] == id]
            XYt_test  = self.XYt_test_denorm[self.XYt_test_denorm[self.id_column] == id]

            preds = self.preds_denorm[self.preds_denorm[self.id_column] == id]
            aggr  = self.aggregated_evaluation_df_denorm            


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
        sns.lineplot(data=preds, x=self.temporal_column, y=self.target_column, color=testcolor, marker = "o", ax=ax2)
        sns.lineplot(data=preds, x=self.temporal_column, y='preds', color=self.model_color, marker = "x",    markeredgecolor='black',  ax=ax2)
        ax2.set_title(f'predictions {self.id_column}: {id}')
        ax2.set_xlabel("")            
        ax2.grid()

        # plot 3: predictions vs groundtruth of aggregation -> over all nodes
        sns.lineplot(data=aggr, x=self.temporal_column, y=self.target_column, color=testcolor, marker = "o", ax=ax3, label='target')
        sns.lineplot(data=aggr, x=self.temporal_column, y='preds', color=self.model_color, marker = "x",    markeredgecolor='black', ax=ax3, label='preds')
        ax3.set_title(f"predictions nationally, aggregated incidences {self.name}")
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

        self.prediction_horizon = dataloader.prediction_horizon

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
            'spatial_consistency': spatial_consistency_loss,
            'temporal_smoothness' : temporal_smoothness_loss,
            'spike_detection' : spike_detection_loss
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
              dataloader_snapshot: bool = True,
              show_loss: bool = True
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

        list_val_loss = []
        list_train_loss=[]

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
            list_train_loss.append(train_mse)
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
            self.train_losses = list_train_loss
            self.val_losses   = list_val_loss

        if show_loss:
            self.plot_losses()

    def plot_losses(self):
        """
        returns plot of train and val losses per epoch, after training model.
        """
        fig, axes = plt.subplots(2,1, figsize = (9,5))
        axes = axes.flatten()

        sns.lineplot(self.train_losses, color = self.traincolor, label = 'train loss', ax = axes[0])
        sns.lineplot(self.val_losses, color = self.valcolor, label = 'val loss',   ax = axes[1])

        for ax in axes:
            ax.grid()
            ax.set_ylabel('loss')
            ax.legend()

        axes[0].set_title('Loss over epochs')
        axes[1].set_xlabel('epoch')        
        
        return fig, axes

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
        prediction_horizon = self.dataloader.prediction_horizon
 
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
        self.test_loss = loss

        # tensor_list_cpu       = [t.detach().cpu() for t in predictions]
        # stacked               = torch.stack(tensor_list_cpu)            # shape [timepoints, n_nodes]
        # df                    = pd.DataFrame(stacked.numpy())  

        # # reform predictions dataframe into a long df with timestep_indices to merge with timepoints
        # long_df_predictions   = df.reset_index().melt(id_vars='index', 
        #                                               var_name=self.id_column, 
        #                                               value_name='preds').rename(columns = {'index':f'{self.temporal_column}_idx'})
        
        # # the predicted timestamps are all of those, without the first periods
        # evaluation_timestamps = eval_df[self.temporal_column].unique()[self.dataloader.periods:]

        # timestamps = {}

        # for idx, tss in enumerate(evaluation_timestamps):
        #     timestamps[idx] = tss

        # cutperiods                               = eval_df[eval_df[self.temporal_column].isin(evaluation_timestamps)] 
        # long_df_predictions[self.temporal_column]= long_df_predictions[f'{self.temporal_column}_idx'].map(timestamps)
        # self.evaluation_df                       = pd.merge(cutperiods,long_df_predictions, on = [self.temporal_column,self.id_column])  
        
        
        # Handle multi-step vs single-step predictions
        if prediction_horizon == 1:
            # Single-step prediction (original behavior)
            tensor_list_cpu = [t.detach().cpu() for t in predictions]
            stacked = torch.stack(tensor_list_cpu)  # shape [timepoints, n_nodes]
            df = pd.DataFrame(stacked.numpy())
            
            # reform predictions dataframe into a long df with timestep_indices to merge with timepoints
            long_df_predictions = df.reset_index().melt(id_vars='index', 
                                                       var_name=self.id_column, 
                                                       value_name='preds').rename(columns = {'index':f'{self.temporal_column}_idx'})
            
            # the predicted timestamps are all of those, without the first periods
            evaluation_timestamps = eval_df[self.temporal_column].unique()[self.dataloader.periods:]
            
            timestamps = {}
            for idx, tss in enumerate(evaluation_timestamps):
                timestamps[idx] = tss
            
            cutperiods = eval_df[eval_df[self.temporal_column].isin(evaluation_timestamps)] 
            long_df_predictions[self.temporal_column] = long_df_predictions[f'{self.temporal_column}_idx'].map(timestamps)
            self.evaluation_df = pd.merge(cutperiods, long_df_predictions, on = [self.temporal_column, self.id_column])
            
        else:
            # Multi-step prediction
            self._process_multi_step_predictions(predictions, labels, eval_df, prediction_horizon)

        return self

    def _process_multi_step_predictions(self, predictions, labels, eval_df, prediction_horizon):
        """
        Process multi-step predictions into evaluation dataframe
        """
        # Convert predictions to numpy
        pred_tensors = [t.detach().cpu() for t in predictions]  # List of [n_nodes, prediction_horizon]
        label_tensors = [t.detach().cpu() for t in labels]      # List of [prediction_horizon, n_nodes]
        
        
        # Stack predictions: [n_samples, n_nodes, prediction_horizon]
        stacked_preds = torch.stack(pred_tensors)  # [n_samples, n_nodes, prediction_horizon]
        stacked_labels = torch.stack(label_tensors)  # [n_samples, prediction_horizon, n_nodes]
        
        # Get unique timestamps for evaluation
        all_timestamps = eval_df[self.temporal_column].unique()
        evaluation_timestamps = all_timestamps[self.dataloader.periods:]
        
        # Create evaluation dataframe
        eval_rows = []
        
        for sample_idx in range(stacked_preds.shape[0]):
            # Get predictions for this sample: [n_nodes, prediction_horizon]
            sample_preds = stacked_preds[sample_idx]  # [n_nodes, prediction_horizon]
            # Get labels for this sample: [prediction_horizon, n_nodes]
            sample_labels = stacked_labels[sample_idx]  # [prediction_horizon, n_nodes]
            
            # For multi-step prediction, we predict multiple future steps
            for step in range(prediction_horizon):
                # Calculate the actual timestamp for this prediction step
                if sample_idx + step < len(evaluation_timestamps):
                    pred_timestamp = evaluation_timestamps[sample_idx + step]
                else:
                    # Handle edge case where we run out of timestamps
                    continue
                
                # Get predictions for this step: [n_nodes]
                step_preds = sample_preds[:, step]  # [n_nodes]
                # Get labels for this step: [n_nodes]
                step_labels = sample_labels[:, step]  # [n_nodes]
                
                # For each node
                for node_idx in range(step_preds.shape[0]):
                    eval_rows.append({
                        self.temporal_column: pred_timestamp,
                        self.id_column: node_idx,
                        'preds': step_preds[node_idx].item(),
                        self.target_column: step_labels[node_idx].item(),
                        'prediction_step': step + 1  # 1-indexed step
                    })
        
        self.evaluation_df = pd.DataFrame(eval_rows)
        
        # Also create aggregated version (sum across all nodes for each timestamp)
        agg_eval = self.evaluation_df.groupby([self.temporal_column, 'prediction_step']).agg({
            self.target_column: 'sum',
            'preds': 'sum'
        }).reset_index()
        
        self.aggregated_evaluation_df = agg_eval
        return self