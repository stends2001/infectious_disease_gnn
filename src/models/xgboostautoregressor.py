from ._shallowmodel import ShallowModel
from typing import Union, Optional, Dict, Literal
import xgboost as xgb
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..utils.constants import paired_colors
class XGBoostAutoRegressor(ShallowModel):
    """
    XGBoost-based autoregressive model for time series forecasting.
    
    Parameters:
    -----------
    dataloader : EpiDataLoader
        Prepared data with train/val/test splits
    name : str, optional
        Custom name for the model

    Examples:
    ---------
    xgbm = XGBoostAutoRegressor(epidata)

    xgbm.set_global_hparams()

    xgbm.set_model_hparams()

    xgbm.train()

    xgbm.forecast('test')
    xgbm.forecast('val')
    xgbm.forecast('train')
    xgbm.show_forecasts('test',[0,69,204])
    xgbm.show_forecasts_maps('test',36,'constant')    
    """
    
    def __init__(self, dataloader, name: Optional[str] = None, node_label: bool = True):
        self.node_label = node_label
        super().__init__(dataloader, name=name)
        
        # Set default name and styling
        self.name = name or 'XGBoostAutoRegressor'
        self.model_color = paired_colors[3]
        self.model = None
        
        # Initialize DMatrix objects for efficient XGBoost training
        self._prepare_dmatrices()
    
    def _prepare_dmatrices(self):
        """Prepare XGBoost DMatrix objects for train/val/test sets."""
        try:
            self.dtrain = xgb.DMatrix(
                self.dataloaders['train']['X'], 
                label=self.dataloaders['train']['y']
            )
            self.dval = xgb.DMatrix(
                self.dataloaders['val']['X'], 
                label=self.dataloaders['val']['y']
            )
            self.dtest = xgb.DMatrix(
                self.dataloaders['test']['X'], 
                label=self.dataloaders['test']['y']
            )
        except KeyError as e:
            raise ValueError(f"Missing required dataloader split: {e}")
    
    def set_global_hparams(self,
                           learning_rate: float = 0.05,
                           num_boost_round: int = 500,
                           early_stopping_rounds: int = 20,
                           seed: int = 42,
                           verbosity: int = 1,
                           **kwargs) -> 'XGBoostAutoRegressor':
        """
        Set global hyperparameters for training process.
        
        Parameters:
        -----------
        learning_rate : float
            Step size shrinkage to prevent overfitting
        num_boost_round : int
            Number of boosting rounds
        early_stopping_rounds : int
            Early stopping patience
        seed : int
            Random seed for reproducibility
        verbosity : int
            Verbosity level (0=silent, 1=warning, 2=info, 3=debug)
        **kwargs : dict
            Additional global parameters
        """
        self.global_hparams = {
            'learning_rate': learning_rate,
            'num_boost_round': num_boost_round,
            'early_stopping_rounds': early_stopping_rounds,
            'seed': seed,
            'verbosity': verbosity,
            **kwargs
        }
        return self
    
    def set_model_hparams(self,
                          max_depth: int = 6,
                          subsample: float = 0.8,
                          colsample_bytree: float = 0.8,
                          colsample_bylevel: float = 1.0,
                          colsample_bynode: float = 1.0,
                          reg_alpha: float = 0,
                          reg_lambda: float = 1,
                          min_child_weight: int = 1,
                          gamma: float = 0,
                          **kwargs) -> 'XGBoostAutoRegressor':
        """
        Set model architecture hyperparameters.
        
        Parameters:
        -----------
        max_depth : int
            Maximum tree depth
        subsample : float
            Fraction of samples used for training each tree
        colsample_bytree : float
            Fraction of features used for training each tree
        colsample_bylevel : float
            Fraction of features used for each level
        colsample_bynode : float
            Fraction of features used for each split
        reg_alpha : float
            L1 regularization term
        reg_lambda : float
            L2 regularization term
        min_child_weight : int
            Minimum sum of instance weight needed in a child
        gamma : float
            Minimum loss reduction required for split
        **kwargs : dict
            Additional model parameters
        """
        self.model_hparams = {
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': colsample_bylevel,
            'colsample_bynode': colsample_bynode,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            **kwargs
        }
        return self
    
    def _build_params(self,
                      objective: str = "reg:squarederror",
                      eval_metric: str = "rmse", 
                      tree_method: str = "hist") -> Dict:
        """
        Build XGBoost parameters from hyperparameters.
        
        Parameters:
        -----------
        objective : str
            Learning objective
        eval_metric : str
            Evaluation metric
        tree_method : str
            Tree construction algorithm
            
        Returns:
        --------
        dict : Combined parameters for XGBoost
        """
        # Set defaults if not already set
        if not hasattr(self, 'global_hparams'):
            self.set_global_hparams()
        if not hasattr(self, 'model_hparams'):
            self.set_model_hparams()
        
        # Combine all parameters
        params = {
            'objective': objective,
            'eval_metric': eval_metric,
            'tree_method': tree_method,
            **{k: v for k, v in self.global_hparams.items() 
               if k not in ['num_boost_round', 'early_stopping_rounds']},
            **self.model_hparams
        }
        
        return params
    
    def train(self, 
              verbose_eval: Union[bool, int] = 20,
              objective: str = "reg:squarederror",
              eval_metric: str = "rmse",
              tree_method: str = "hist") -> 'XGBoostAutoRegressor':
        """
        Train the XGBoost model using set hyperparameters.
        
        Parameters:
        -----------
        verbose_eval : bool or int
            Verbosity for training progress
        objective : str
            Learning objective
        eval_metric : str
            Evaluation metric
        tree_method : str
            Tree construction algorithm
            
        Returns:
        --------
        self : XGBoostAutoRegressor
        """
        # Build parameters from hyperparameters
        self.params = self._build_params(objective, eval_metric, tree_method)
        
        # Define evaluation sets
        evals = [(self.dtrain, 'train'), (self.dval, 'val')]
        
        # Extract training parameters
        if not hasattr(self, 'global_hparams'):
            self.set_global_hparams()
            
        num_boost_round = self.global_hparams.get('num_boost_round', 500)
        early_stopping_rounds = self.global_hparams.get('early_stopping_rounds', 20)
        
        # Train the model
        self.model = xgb.train(
            params=self.params,
            dtrain=self.dtrain,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            evals=evals,
            verbose_eval=verbose_eval
        )
        
        return self
    
    def forecast(self, 
                 dataset: Literal['train', 'val', 'test'] = 'test') -> 'XGBoostAutoRegressor':
        """
        Generate forecasts for specified dataset.
        
        Parameters:
        -----------
        dataset : str
            Which dataset to forecast on ('train', 'val', 'test')
            
        Returns:
        --------
        self : XGBoostAutoRegressor
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Select appropriate DMatrix
        dmatrix_map = {
            'train': self.dtrain,
            'val': self.dval, 
            'test': self.dtest
        }
        
        if dataset not in dmatrix_map:
            raise ValueError(f"Invalid dataset: {dataset}. Must be one of {list(dmatrix_map.keys())}")
        
        # Generate predictions
        dmatrix     = dmatrix_map[dataset]
        predictions = self.model.predict(dmatrix)
        
        self._format_predictions(dataset, predictions)
        
        return self
    
    def _format_predictions(self, dataset, predictions):
        data_input   = self.dataloaders[dataset]['X']

        if self.node_label:
            data_input.drop(labels = [self.dataloader.id_column], axis = 1, inplace=True)

        data_context = self.dataloaders[dataset]['c']
        data_target  = self.dataloaders[dataset]['y']
        data_preds   = pd.DataFrame(predictions, index=data_context.index, columns=['pred'])
        evaluation_df= pd.concat([data_context, data_input, data_target, data_preds], axis=1)

        self.evaluation_datasets[dataset] = evaluation_df

    def get_feature_importance(self, importance_type: str = 'weight') -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Parameters:
        -----------
        importance_type : str
            Type of importance ('weight', 'gain', 'cover')
            
        Returns:
        --------
        pd.DataFrame
            Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance = self.model.get_score(importance_type=importance_type)
        
        return pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)