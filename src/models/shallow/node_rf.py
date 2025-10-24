from ..base.basemodel import BaseModel

from typing import Optional, Union, List, Dict, Literal
from ...dataloading.shallowdataloader import ShallowDataLoader
from sklearn.ensemble import RandomForestRegressor

from ...utils.textformatting import section, align

from sklearn.metrics import mean_absolute_error, mean_squared_error

import pandas as pd
import numpy as np

class NodeRFModel(BaseModel):

    """

    """

    def __init__(self, 
                 dataloader     : 'ShallowDataLoader',
                 name           : Optional[str] = None):
        super().__init__(dataloader, name)
        
        if not self.name:
            self.name = f'NodeRFModel'
        
        # dataloader metadata
        self.dataloader         = dataloader
        self.sequence_length    = dataloader.sequence_length
        self.horizon_size       = dataloader.horizon_size
        self.models             = {}
        self.evaluation_datasets = {}
        
        self._state = {
            'model_initialized' : False,
            'trained'           : False,
            'forecasted'        : False,
        }
        
        self.evaluation_datasets    = {}
        self.train_losses           = []
        self.val_losses             = []
       
    def _check_state(self, required_states: list) -> None:
        """Validate that required setup steps have been completed."""
        missing = [s for s in required_states if not self._state.get(s, False)]
        if missing:
            raise ValueError(
                f"Missing required setup steps: {', '.join(missing)}. "
                f"Call the corresponding methods first."
            )

    def set_model_hparams(self,
                        max_depth: int = 6,
                        min_samples_split: int = 4,
                        min_samples_leaf: int = 4,
                        bootstrap: bool = True,
                        n_estimators: int = 100,
                        random_state: int = 42,
                        **kwargs):
        """

        """
        self._check_state(['global_hparams_set'])

        # Protect global hparams from being overwritten here
        forbidden_keys = {'learning_rate', 'n_estimators', 'early_stopping_rounds'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in forbidden_keys}

        # Prepare the model hyperparameters dictionary
        self.model_hparams = {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap,
            'n_estimators': n_estimators,
            'random_state': random_state,
            **filtered_kwargs
        }

        self.config_info['model_hparams'] = self.model_hparams
        self._state['model_initialized'] = True

        # Initialize models per horizon
        self.models: Dict[str, RandomForestRegressor] = {}
        for hh in self.dataloader.target_horizons:
            # Initialize a model for each horizon
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
            
            # Update model with hyperparameters
            model.set_params(**self.model_hparams)  # Update with all hyperparameters like max_depth, subsample, etc.
            
            # Store the model for each horizon
            self.models[hh] = model

        print(f"✓ Initialized {self.horizon_size} RF model")

    def set_global_hparams(self, 
                        lr: float = 0.001,
                        n_epochs: int = 5,
                        patience: int = 15,
                        min_delta: float = 1e-4,
                        loss: Literal['rmse'] = 'rmse'):
        """
        Set global training hyperparameters (used across models).
        """

        self.global_hparams = {
            'lr': lr,
            'n_epochs': n_epochs,
            'patience': patience,
            'min_delta': min_delta
        }

        self.lr = lr
        self.n_epochs = n_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.loss    = loss

        self.config_info['global_hparams'] = self.global_hparams
        self._state['global_hparams_set'] = True

    def train(self, verbose=2, show_loss: bool = True):
        """
        Train and validate models for each horizon.
        """
        self._check_state(['model_initialized', 'global_hparams_set'])

        for hh in self.dataloader.target_horizons:

            # Retrieve the training and validation data for this horizon
            X_train, y_train = self.dataloader.dataloader_train[hh].values()
            X_val, y_val = self.dataloader.dataloader_val[hh].values()

            model = self.models[hh]

            # Train the model on the training data
            model.fit(X_train, y_train.values.ravel())

            # After training the model, validate it
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Compute training and validation loss (e.g., RMSE, MAE)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            val_mae = mean_absolute_error(y_val, y_val_pred)

            if verbose > 1:
                print(f"Validation loss for horizon {hh}: RMSE = {val_rmse:.4f}, MAE = {val_mae:.4f}")

            # Optionally, track best model and early stopping
            best_loss = float('inf')
            patience_counter = 0

            # Track if validation loss improves
            if val_rmse < best_loss - self.min_delta:
                best_loss = val_rmse
                patience_counter = 0  # reset patience if there is improvement
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping for horizon {hh} due to lack of improvement.")
                break  # stop training if validation doesn't improve

            # Optionally, display or log the training progress and losses
            if show_loss:
                print(f"Training RMSE: {train_rmse:.4f} - Validation RMSE: {val_rmse:.4f}")

            # You can save the best model here if needed
            if val_rmse < best_loss:
                best_loss = val_rmse
                # Save the model for this horizon
                # self.save_model_for_horizon(hh, model)

        print("Training complete!")
    
    def forecast(self, dataset: Literal['train','val','test'] = 'test'):

        if dataset == 'train':
            dataloader = self.dataloader.dataloader_train
        elif dataset == 'val':
            dataloader = self.dataloader.dataloader_val 
        elif dataset == 'test':
            dataloader = self.dataloader.dataloader_test                  
        
        horizon_prediction_dict = {'transformed':{}, 'nontransformed':{}}
        
        for ii, hh in enumerate(self.dataloader.target_horizons):
            # Retrieve the training and validation data for this horizon
            X, y = dataloader[hh].values()



            model = self.models[hh]
            preds = model.predict(X)
            
            # rmse = np.sqrt(mean_squared_error(y, preds))
            # print(rmse)

            evaluation_df               = self.dataloader.dataloader_main[hh][self.dataloader.dataloader_main[hh][dataset]]
            evaluation_df.loc[:,'incidence']  = y.values
            evaluation_df.loc[:,'pred']       = preds

            horizon_prediction_dict['transformed'][f'horizon_{ii}']     = evaluation_df
            horizon_prediction_dict['nontransformed'][f'horizon_{ii}']  = self._denorm_predictions(evaluation_df)

        self.evaluation_datasets[dataset]   = horizon_prediction_dict
        self._state['forecasted']           = True
        return self        
   
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
        lines = ['<NodeRFModel(']
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
        lines.extend(section('model hparams', model_hparams, width))
        lines.append('')
        
        # Global hparams
        lines.extend(section('global hparams', self.config_info.get('global_hparams', {}), width))
        
        lines.append(')>')
        
        return '\n'.join(lines)