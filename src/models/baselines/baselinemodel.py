from typing import Optional, Literal, List, Self
import numpy as np
import pandas as pd
from abc import abstractmethod

from ..base import BaseModel
from ...dataloading import BaseLineDataLoaderManager 
from ...dataloading.epidataorchestration.normalization import apply_minmax_scaling, apply_zscore_scaling
from ...utils.textformatting import warning_emoji

class BaseLineModel(BaseModel[BaseLineDataLoaderManager]):
    """
    Parent class for all BaseLineModels, as well as a subclass of BaseModel.

    The main difference between BaseLineModel - instances and other models, is that
    the former work off of non-transformed data, while the latter require
    transformed data. Since the PredictionManager still expects the predictions to
    be in the transformed scale, baselinemodels need to transform their predictions
    into the same scale.

    The BaseLineModel has initiated the following 'intermediate' class functions: 
    `train()`, `set_model_hparams()`, `set_global_hparams()`, that return a 
    print-statement that a baselinemodel doesn't require these. When necessary, 
    the child class must implement these themselves. 

    Note
    ----
    Each model requires a `forecast()` function, listed here as abstractmethod
    

    See Also
    --------
    BaseModel
    """
    def __init__(self, 
                 dataloadermanager: BaseLineDataLoaderManager,                     
                 name:              Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        self._expected_dataloadermanager = 'BaseLineDataLoaderManager'
        
        super().__init__(dataloadermanager = dataloadermanager, name = name, verbose = verbose)
        self._setup_transformations()   

    @abstractmethod
    def forecast(self, dataset: Literal['train','val','test'] = 'test') -> Self:
        pass

    # ====== NONSENSE METHODS ====== #
    def train(self, *args, **kwargs):
        print("This BaseLineModel doesn't train")

    def set_global_hparams(self, *args, **kwargs):
        print("This BaseLineModel doesn't have global hyper parameters")

    def set_model_hparams(self, *args, **kwargs):
        print("This BaseLineModel doesn't have model hyper parameters") 

    def save_model(self, *args, **kwargs):
        print(f'{warning_emoji} Baseline models cant be saved.')
    
    # ======= HIDDEN METHODS ======= 
    def _setup_transformations(self):
        """
        setup factory method for transformation methods.
        """
        self.transformation_funcs = {
            'minmax': apply_minmax_scaling,
            'zscore': apply_zscore_scaling,
            'log'   : self._apply_log
        }

    def _apply_log(self, df: pd.DataFrame, cols: List[str], eps: float) -> pd.DataFrame:
        """ 
        apply log

        Parameters
        ----------
        df: pd.DataFrame
            the dataframe in which column {cols} will be logged
        cols: List[str]
            the list of column names to be logged
        eps: float
            small constant to make logging safe

        Returns
        -------
        df: pd.DataFrame
            df with logged columns
        """
        for col in cols:
            df[col] = np.log(df[col] + eps)
        return df
 
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        apply normalization, by calling methods in self.transformation_funcs

        Parameters
        ----------
        df: pd.DataFrame
            the dataframe to be normalized

        Returns
        -------
        df_transformed: pd.DataFrame
            the transformed dataframe
        """
        normalization_method    = self.dataloadermanager.dataorchestrator.config.normalization_method
        df_transformed          = df.copy()
           
        if self.dataloadermanager.dataorchestrator.config.target_column == 'incidence':
            col_entry_target        = self.column_registration.get_by_name('target')
            transformation_dict     = col_entry_target.transformation_params

            if 'log' in transformation_dict:
                df_transformed = self.transformation_funcs['log'](df   = df_transformed, 
                                                                  cols = (['target'] + self.pred_cols), 
                                                                  eps  = transformation_dict['log'])

            if normalization_method:
                df_transformed = self.transformation_funcs[normalization_method](val_df = df_transformed, 
                                                                                 params = {col: transformation_dict['normalization'] for col in (['target'] + self.pred_cols)}, 
                                                                                 columns= (['target'] + self.pred_cols))
                        
        return df_transformed           