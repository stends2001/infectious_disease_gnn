from typing import Optional, Literal, List
import numpy as np
import pandas as pd

from ..base import BaseModel
from ...dataloading import BaseLineDataLoaderManager 
from ...dataloading.epidataorchestration.normalization import apply_minmax_scaling, apply_zscore_scaling
from ...utils.textformatting import warning_emoji

class BaseLineModel(BaseModel):

    def __init__(self, 
                 dataloadermanager: BaseLineDataLoaderManager,                     
                 name:              Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = f'Unnamed BaseModel'

        if dataloadermanager.__class__.__name__ != 'BaseLineDataLoaderManager':
            raise ValueError(f'Unexpected dataloadermanager type. Expected BaseLineDataLoaderManager but got {dataloadermanager.__class__.__name__}')

        super().__init__(dataloadermanager=dataloadermanager, name= name, verbose=verbose)

        if self.epiconfig._num_quantiles == 0:
            pred_cols= ['pred']
        else:
            pred_cols= [c for c in self.column_registration.pred_columns if c != 'pred']
        self.pred_cols = pred_cols

        self._setup_transformations()

    def train(self, dataset: Literal['train','val','test'] = 'test'):
        print("This baseline model doesn't train")

    def forecast(self):
        raise NotImplementedError("Child classes of BaseLineModel must implement train-method")

    def set_global_hparams(self):
        print("This baseline model doesn't require global hparams")

    def set_model_hparams(self):
        print("This baseline model doesn't require model hparams")     

    def save_model(self):
        print(f'{warning_emoji} Baseline models cant be saved.')
    
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
            transformation_dict     = col_entry_target.transformation_params

            if 'log' in transformation_dict:
                df_norm = self.transformations['log'](df_norm, cols=(['target'] + self.pred_cols), eps=transformation_dict['log'])

            if normalization_method:
                df_norm = self.transformations[normalization_method](df_norm, params = {col: transformation_dict['normalization'] for col in (['target'] + self.pred_cols)}, columns = (['target'] + self.pred_cols))
                        
        return df_norm           