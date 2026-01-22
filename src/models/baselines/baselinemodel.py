from typing import Optional, Literal, List
from abc import abstractmethod
import numpy as np
import pandas as pd
from src.dataloading.dataorchestration.column_registry import ColEntryMissingTransformationError

from ..base import BaseModel
from ...dataloading import BaseLineDataLoaderManager 
from ...dataloading.dataorchestration.normalization import apply_minmax_scaling, apply_zscore_scaling


class BaseLineModel(BaseModel):

    def __init__(self, 
                 dataloadermanager: BaseLineDataLoaderManager,       
                 model_color:        str,                           
                 name:              Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = f'Unnamed BaseModel'

        super().__init__(dataloadermanager=dataloadermanager, name= name, model_color="#d8d8d8c3", verbose=verbose)

        self._setup_transformations()

    def train(self):
        print("This baseline model doesn't train")

    def set_global_hparams(self):
        print("This baseline model doesn't require global hparams")

    def set_model_hparams(self):
        print("This baseline model doesn't require model hparams")     

    def _setup_transformations(self):
        self.transformations = {
            'minmax': apply_minmax_scaling,
            'zscore': apply_zscore_scaling,
            'log'   : self._apply_log
        }

    def _apply_log(self,df, cols: List[str], eps):
        for col in cols:
            df[col] = np.log(df[col] + eps)
        return df
 
    def _normalize(self, df: pd.DataFrame):
        """
        """
        normalization_method    = self.dataloadermanager.dataorchestrator.config.normalization_method
        
        df_norm               = df.copy()       
        if self.dataloadermanager.dataorchestrator.config.target_column == 'incidence':
            col_entry_target        = self.column_registration.get_by_name('target')
            transformation_dict     = col_entry_target.transformation

            if transformation_dict is None:
                raise ColEntryMissingTransformationError(entryname = 'target')

            if 'log' in transformation_dict:
                df_norm = self.transformations['log'](df_norm, cols=['target','pred'], eps=transformation_dict['log'])

            df_norm = self.transformations[normalization_method](df_norm, params = {col: transformation_dict['normalization'] for col in ['target','pred']}, columns = ['target','pred'])
            
        return df_norm           

    @abstractmethod
    def forecast():
        pass
