from ..base import BaseModel, PredictionCollection
from typing import Optional, Union, List, Dict, Literal
from ...dataloading import ShallowDataLoaderManager
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

from ...utils.textformatting import section, align

from sklearn.metrics import mean_absolute_error, mean_squared_error

import pandas as pd
import numpy as np

class NodeRFModel(BaseModel):

    """
    Examples:
    --------
    rf = NodeRFModel(shallowdata)
    rf.set_global_hparams()
    rf.set_model_hparams(n_estimators=1)
    rf.train(verbose=0)
    rf.forecast('test')
    rf.forecast('train')
    rf.forecast('val')    
    """

    def __init__(self, 
                 dataloadermanager: ShallowDataLoaderManager, 
                 name:              Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = f'NodeRFModel'       

        super().__init__(dataloadermanager, name, verbose)

        self.dataloadermanager: ShallowDataLoaderManager= dataloadermanager
        
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
                          class_weight: Optional[Union[str, dict]] = None,
                          **kwargs):
        """

        """
        self._check_state(['global_hparams_set'])

        # Prepare the model hyperparameters dictionary
        self.model_hparams = {
            'max_depth':            max_depth,
            'min_samples_split':    min_samples_split,
            'min_samples_leaf':     min_samples_leaf,
            'bootstrap':            bootstrap,
            'n_estimators':         n_estimators,
            'random_state':         random_state,
        }

        # Add class_weight only for classification
        if self.prediction_mode == 'classification' and class_weight is not None:
            self.model_hparams['class_weight'] = class_weight        

        self.config_info['model_hparams']    = self.model_hparams
        self._update_status('model_hparams_set')

        # Initialize models per horizon
        self.models: Dict[str, Union[RandomForestRegressor, RandomForestClassifier]] = {}

        for hh in range(self.dataloadermanager.dataorchestrator.config.horizon_size):
            if self.prediction_mode == 'regression':
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1) # Initialize a model for each horizon
            if self.prediction_mode == 'classification':
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1) # Initialize a model for each horizon                
            model.set_params(**self.model_hparams)  # Update with all hyperparameters like max_depth, subsample, etc.
        
            self.models[f"horizon_{hh}"] = model

    def set_global_hparams(self, 
                           loss: Literal['rmse','accuracy','f1'] = 'rmse'):
        """
        Set global training hyperparameters (used across models).
        """

        self.global_hparams = {
            'loss':           loss,

        }

        self.loss       = loss

        self.config_info['global_hparams'] = self.global_hparams
        self._update_status('global_hparams_set')

    def train(self):
        """
        Train and validate models for each horizon.
        """
        self._check_state(['model_initialized', 'global_hparams_set'])

        for hh in range(self.dataloadermanager.dataorchestrator.config.horizon_size):

            horizon_name =f"horizon_{hh}"

            # Retrieve the training and validation data for this horizon
            dataloader_collection   = self.dataloadermanager.dataloader_collections[horizon_name]
            X_train, y_train        = dataloader_collection.train.X, dataloader_collection.train.y
            X_val,   y_val          = dataloader_collection.val.X,   dataloader_collection.val.y

            model = self.models[horizon_name]

            # Train the model on the training data
            model.fit(X_train, y_train.squeeze(1))

            # After training the model, validate it
            y_train_pred    = model.predict(X_train)
            y_val_pred      = model.predict(X_val)

            # Compute metrics based on task type
            if self.prediction_mode == 'classification':
                train_acc = accuracy_score(y_train, y_train_pred)
                train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
                val_acc = accuracy_score(y_val, y_val_pred)
                val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
                
                if self.verbose > 1:
                    print(f"Training metrics for horizon {hh}: Accuracy = {train_acc:.4f}, F1 = {train_f1:.4f}")
                    print(f"Validation metrics for horizon {hh}: Accuracy = {val_acc:.4f}, F1 = {val_f1:.4f}")
            else:
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                train_mae = mean_absolute_error(y_train, y_train_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                val_mae = mean_absolute_error(y_val, y_val_pred)
                
                if self.verbose > 1:
                    print(f"Training loss for horizon {hh}: RMSE = {train_rmse:.4f}, MAE = {train_mae:.4f}")
                    print(f"Validation loss for horizon {hh}: RMSE = {val_rmse:.4f}, MAE = {val_mae:.4f}")
        
        self._update_status('trained')

   
    def forecast(self, dataset: Literal['train','val','test'] = 'test'):
        """
        Forecast for set dataset
        """
        
        for hh in range(self.dataloadermanager.dataorchestrator.config.horizon_size):
            horizon_name            = f'horizon_{hh}'
            timesteps_ahead         = int(hh + self.dataloadermanager.dataorchestrator.config.horizon_leadtime)
            timeshift               = f"{timesteps_ahead}W"
            dataloader_collection   = self.dataloadermanager.dataloader_collections[horizon_name]

            if dataset == 'train':
                X, y = dataloader_collection.train.X, dataloader_collection.train.y
            elif dataset == 'val':
                X, y = dataloader_collection.val.X, dataloader_collection.val.y                
            elif dataset == 'test':
                X, y = dataloader_collection.test.X, dataloader_collection.test.y      
            else:
                raise ValueError('please provide a valid dataset: "train"/"val"/"test"')       

            # This is the df with all features and all targets (all incidence_ahead...)
            Xy_main      = dataloader_collection.main.copy()
            Xy_dataset   = Xy_main[Xy_main[dataset]].reset_index(drop=True)
            evaluation_df= Xy_dataset

            model = self.models[horizon_name]

            # NEW: Handle classification differently
            if self.prediction_mode == 'classification':
                # Get probability predictions for the positive class
                preds_proba = model.predict_proba(X)[:, 1]  # P(class=1)
                # preds = (preds_proba > self.classification_threshold).astype(int)  # Binary predictions
                
                evaluation_df = Xy_dataset.copy().rename(columns={f'incidence_ahead{timesteps_ahead}': 'target'})
                evaluation_df['pred'] = preds_proba
                # evaluation_df['pred_proba'] = preds_proba  # Store probabilities too
            else:
                preds = model.predict(X)
                evaluation_df = Xy_dataset.copy().rename(columns={f'incidence_ahead{timesteps_ahead}': 'target'})
                evaluation_df['pred'] = preds

            self.predictions.add_horizon_predictions(dataset, evaluation_df, hh)

        self._update_status('forecasted')    
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
        # TODO
        # lines.extend(section('forecasts', {'forecasted': list(self.evaluation_datasets.keys())}, width))
        lines.append('')
        
        # Model hparams
        model_hparams = dict(self.config_info.get('model_hparams', {}))
        lines.extend(section('model hparams', model_hparams, width))
        lines.append('')
        
        # Global hparams
        lines.extend(section('global hparams', self.config_info.get('global_hparams', {}), width))
        
        lines.append(')>')
        
        return '\n'.join(lines)