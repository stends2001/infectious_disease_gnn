from typing import Dict, List
import pandas as pd
from dataclasses import field

from src.issues import Error

class InvalidPredictionCompilation(Error):
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)   

class InvalidMetricsCompilation(Error):
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)   

class EvaluationPredictionsCompilation:

    """ 
    _data: Dict[DATASET: Dict[HORIZON: Dict["predictions": pd.DataFrame, "metrics": pd.DataFrame]]]


    A class to store and manage predictions compilations (i.e. one dataset with preds of all models)
    and associated metrics, over multiple datasets (train/val/test) and horizons.

    Per combination of dataset and horizon, there is a long table as follows in
    _predictions_compilations 
    ________________________________________________________
    | node | pred-column | model1.name | ... | modelN.name |

    and _metric_compilations
    ___________________________________________________
    | node | metric | model1.name | ... | modelN.name |


    Methods
    -------
    add_horizon

    get_compilation

    Properties
    ----------
    compilation    



    Note
    ----
    The structure of self._data looks as follows:
    {
        'train': {'horizon_0' : {'predictions' : df, 'metrics' : df}},    
        'val'  : {'horizon_0' : {'predictions' : df, 'metrics' : df},
                  'horizon_1' : {'predictions' : df, 'metrics' : df}}                 
        'test' : {'horizon_0' : {'predictions' : df, 'metrics' : df},
                  'horizon_1' : {'predictions' : df, 'metrics' : df},
                  'horizon_2' : {'predictions' : df, 'metrics' : df}},
    }    
    """

    def __init__(self, model_names: List[str]):
        self.model_names    = model_names
        self._data:                     Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}

    # ======== DATA ====== #
    def add_data(self, predictions: pd.DataFrame, metrics: pd.DataFrame, horizon: int, dataset: str):
        """Adds data"""
        horizon_str = f"horizon_{horizon}"
        
        if dataset not in self._data:
            self._data[dataset] = {} 
            
        self._data[dataset][horizon_str] = {'predictions':  predictions,
                                            'metrics'    :  metrics}

    def get_data(self, horizon: int, dataset: str) -> Dict[str, pd.DataFrame]:
        """Get the data for a specified horizon and predictions_compilation."""
        horizon_str = f"horizon_{horizon}"

        if dataset not in self._data:
            raise ValueError('data not found')
        
        return self._data[dataset][horizon_str]         
    
    @property
    def datasets(self) -> List[str]:
        """
        
        """
        return list(self._data.keys())

    @property
    def horizons(self) -> Dict[str, List[str]]:
        """
        
        """        
        dataset_horizons = {}
        for dataset in self.datasets:
            horizons                    = list(self._data[dataset].keys())
            dataset_horizons[dataset]   = horizons
        return dataset_horizons

    def __repr__(self) -> str:
        representation = f"<{self.__class__.__name__}(" 
        
        representation += f"{self.horizons}" if len(self.horizons) else "empty"
        
        representation += ")>"
        return representation     


