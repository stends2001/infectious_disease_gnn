from typing import Optional, Literal, List, Self
import numpy as np
import pandas as pd
from abc import abstractmethod

from ..base import BaseModel
from ...dataloading.databuilders import BaseLineDataBuilder 
from ...dataloading.epidataorchestration.utils.normalization import apply_log, apply_zscore, apply_minmax
from ...utils.textformatting import warning_emoji

class BaseLineModel(BaseModel[BaseLineDataBuilder]):
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
                 dataloadermanager: BaseLineDataBuilder,                     
                 name:              str,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        self._expected_dataloadermanager = 'BaseLineDataBuilder'
        
        super().__init__(dataloadermanager = dataloadermanager, name = name, verbose = verbose)

    @abstractmethod
    def forecast(self, dataset: Literal['train','val','test'] = 'test') -> None:
        pass

    # ====== NONSENSE METHODS ====== #
    def train(self, *args, **kwargs) -> None:
        print("This BaseLineModel doesn't train")

    def set_global_hparams(self, *args, **kwargs) -> None:
        print("This BaseLineModel doesn't have global hyper parameters")

    def set_model_hparams(self, *args, **kwargs) -> None:
        print("This BaseLineModel doesn't have model hyper parameters") 

    def save_model(self, *args, **kwargs) -> None:
        print(f'{warning_emoji} Baseline models cant be saved.')
    
    # ======= HIDDEN METHODS ======= 
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization to baseline model predictions, bringing them
        into the transformed scale expected by PredictionManager.
        """
        if self.dataloadermanager.dataorchestrator.config.target_column != 'incidence':
            return df.copy()

        col_entry = self.column_registration.get_entry_by_name('target')
        params    = col_entry._transformation_params

        if params is None:
            return df.copy()

        columns       = ['target'] + self.pred_cols
        df_transformed = df.copy()

        for col in columns:
            if col not in df_transformed.columns:
                continue
            if params.log is not None:
                df_transformed = apply_log(df_transformed, col, params.log)
            if params.zscore is not None:
                df_transformed = apply_zscore(df_transformed, col, params.zscore)
            elif params.minmax is not None:
                df_transformed = apply_minmax(df_transformed, col, params.minmax)

        return df_transformed