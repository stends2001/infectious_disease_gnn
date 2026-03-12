from typing import Dict, List, Literal
import pandas as pd

from src.issues import Error 

class EvaluationPredictionsCompilation:
    """ 
    Stores all predictions and metrics within the evaluator. The main attribute is
    self._data. This dictionary stores data the following way:
    _data:  Dict[DATASET: 
                Dict[HORIZON: 
                    Dict["predictions"  : pd.DataFrame, 
                         "metrics"      : pd.DataFrame]]]

    Per combination of dataset and horizon, there is a long table as follows in
    Dict[DATASET][HORIZON]['predictions']
    _____________________________________________________
    | timestamp | node | target | model | pred-cols ... |

    and _metric_compilations
    ___________________________________
    | node | model | metric-cols ... |

    Methods
    -------
    - add_data()
    - get_data()

    Properties
    ----------
    - datasets
    - horizons    

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

    # ======== DATA WORKING ====== #
    def add_data(self, predictions: pd.DataFrame, metrics: pd.DataFrame, horizon: int, dataset: Literal['train','val','test']):
        """
        Adds data to self._data to [dataset][horizon]

        Parameters
        ----------
        predictions: pd.DataFrame
            predictions dataframe with columns TODO
        metrics: pd.DataFrame
            metrics dataframe with columns TODO
        horizon: int
            integer of horizon of prediction
        dataset: Literal['train','val','test']
            dataset of prediction
        """
        horizon_str = f"horizon_{horizon}"
        
        if dataset not in self._data:
            self._data[dataset] = {} 
            
        self._data[dataset][horizon_str] = {'predictions': predictions, 'metrics' : metrics}

    def get_data(self, horizon: int, dataset: str) -> Dict[str, pd.DataFrame]:
        """
        Gets data from self._data, from [dataset][horizon]

        Parameters
        ----------
        horizon: int
            integer of horizon of prediction
        dataset: Literal['train','val','test']
            dataset of prediction

        Returns
        -------
        Dict[str, pd.DataFrame]
            keys:
            - 'predictions'
            - 'metrics'
            values:
            - predictions_df: columns TODO
            - metrics_df: columns TODO
        """
        horizon_str = f"horizon_{horizon}"

        if dataset not in self._data:
            raise ValueError('data not found')
        
        return self._data[dataset][horizon_str]         
    
    @property
    def datasets(self) -> List[str]:
        """returns a list of datasets inside the main data dictionary"""
        return list(self._data.keys())

    @property
    def horizons(self) -> Dict[str, List[str]]:
        """returns a list of horizons inside the main data dictionary. Outputted per dataset"""  
        dataset_horizons = {}
        for dataset in self.datasets:
            horizons                    = list(self._data[dataset].keys())
            dataset_horizons[dataset]   = horizons
        return dataset_horizons

    def __repr__(self) -> str:
        """repr of horizons-property basically"""
        representation = f"<{self.__class__.__name__}(" 
        
        representation += f"{self.horizons}" if len(self.horizons) else "empty"
        
        representation += ")>"
        return representation     