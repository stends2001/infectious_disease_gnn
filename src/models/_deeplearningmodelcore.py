



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
        if not self.model_hparams_set:
            raise ValueError(f'set model hyperparameters first! Use model.set_model_hparams()')

        if not self.global_hparams_set:
            raise ValueError(f'set global hyperparameters first! Use model.set_global_hparams()')

        if dataloader_snapshot:
            print(f'Dataloader Snapshot: {self.dataloader.dataset_train[0]}')

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
        epochs          = np.arange(len(self.train_losses))
        patience_epochs = epochs[np.array(self.epoch_patience)]
        patience_train_losses = np.array(self.train_losses)[np.array(self.epoch_patience)]
        patience_val_losses   = np.array(self.val_losses)[np.array(self.epoch_patience)]

        fig, axes = plt.subplots(1,2, figsize = (18,4))
        axes = axes.flatten()

        sns.lineplot(self.train_losses,          color = self.traincolor, label = 'train loss', ax = axes[0])
        axes[0].scatter(patience_epochs, patience_train_losses, color='red', marker = 'x', label='Patience Epochs')
        sns.lineplot(self.val_losses,            color = self.valcolor,   label = 'val loss',   ax = axes[1])
        axes[1].scatter(patience_epochs, patience_val_losses, color='red',marker = 'x', label='Patience Epochs')   

        for ax in axes:
            ax.grid()
            ax.set_ylabel('loss')
            ax.set_xlabel('epoch')
            ax.legend()

        axes[0].set_title('Training loss')      
        axes[1].set_title('Validation loss')    
        
        return fig, axes

    def forecast(self,
                 dataset: Literal['train','val','test']  = 'test'
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

        if dataset == 'test':
        # formatting predictions:
            eval_df     = self.dataloader.test_df
            dataloader  = self.dataloader.dataset_test

        if dataset == 'val':
        # formatting predictions:
            eval_df     = self.dataloader.val_df
            dataloader  = self.dataloader.dataset_val           

        if dataset == 'train':
        # formatting predictions:
            eval_df     = self.dataloader.train_df
            dataloader  = self.dataloader.dataset_train  

        prediction_horizon = self.dataloader.prediction_horizon
 
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