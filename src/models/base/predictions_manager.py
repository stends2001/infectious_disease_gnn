import pandas as pd 
import geopandas as gpd
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Union
from ...dataloading.dataorchestration.normalization import reverse_log, reverse_minmax_scaling, reverse_zscore_scaling
from ...dataloading.dataorchestration.epiconfig import EpiConfig
from ...dataloading.dataorchestration.column_registry import ColumnRegistration, ColEntryMissingTransformationError

from ..utils.loss.poissonloss import convert_poisson_predictions

# ============= Predictions Manager =============
class PredictionManager:
    """
    Manages predictions in a centralized class.

    Parameters
    ----------
    epiconfig: EpiConfig
        the dataorchestrator's configuration object

    column_registration: ColumnRegistration
        the dataorchestrator's columnregistration object

    Attributes
    ----------
    train, val, test: PredictionCollection
        each of these is a collection of prediction - dataframes per horizon

    Examples
    --------

    See Also
    --------
    PredictionCollection

    """

    def __init__(self, epiconfig: EpiConfig, column_registration: ColumnRegistration):

        self.train = PredictionCollection()
        self.val   = PredictionCollection()
        self.test  = PredictionCollection()
        
        self.epiconfig          = epiconfig
        self.column_registration= column_registration
        self._setup_reverse_transformations()

    def _setup_reverse_transformations(self):
        self.reverse_transformations = {
            'minmax': reverse_minmax_scaling,
            'zscore': reverse_zscore_scaling,
            'log'   : reverse_log
        }

    def add_horizon_predictions(self, dataset: Literal['train','val','test'], horizon_df: pd.DataFrame, horizon: int, additional_transformation: bool = False, transf: Optional[str] = None, transf_args: Optional[Dict[str,Union[str,float]]] = None):
        """
        add the predictions - dataframe of a certain horizon for a set dataset

        Parameters
        ----------
        dataset: Literal['train','val','test']
            gets saved seperately into respective attribute
        horizon_df: pd.DataFrame
            dataframe that looks like: | timestamp | node | pred | target | 
        horizon: int
            the integer of horizon, based on which the timesteps ahead (= horizon + horizon_leadtime) is calculated

        See Also
        --------
        self._denorm_predictions()
        """
        # get required columns
        df              = self._get_prediction_df_cols(horizon_df)   
        # validate columns
        df_validated    = self._validate_columns(df)
        timesteps_ahead = int(horizon + self.epiconfig.horizon_leadtime)
        
        if dataset == 'train':
            prediction_collection = self.train 

        elif dataset == 'val':
            prediction_collection = self.val

        elif dataset == 'test':
            prediction_collection = self.test 

        else:
            raise ValueError(f'{dataset} invalid dataset. Supply "train", "val" or "test"')      

        df = self._apply_prediction_timeshift(df_validated, f'{timesteps_ahead}W')

        if additional_transformation:

            if transf == 'poisson_sampling':

                if 'sampling_mode' in  transf_args.keys():

                    sampling_mode = transf_args['sampling_mode'] 
                    df['pred'] = convert_poisson_predictions(df['pred'], sampling_mode)

                else:
                    raise ValueError('When using poisson_sampling as transformation, please supplly an argument for "sampling_mode" (options: "mean", "sample", "mode")')

            else:
                raise ValueError(f'currently only "poisson_sampling" supported as argument for additional_transformation')

        # add both original (nontransformed) and transformed prediction - data
        prediction_collection.add(df.copy(), horizon=horizon, is_original=False)
        prediction_collection.add(self._denorm_predictions(df), horizon=horizon, is_original=True)

    def get_preds(self, dataset: Literal['train','val','test']) -> 'PredictionCollection':
        """
        get predictions for either train/val/test
        """

        if dataset == 'train':
            prediction_collection = self.train 

        elif dataset == 'val':
            prediction_collection = self.val

        elif dataset == 'test':
            prediction_collection = self.test 

        if prediction_collection._contains_data():
            return prediction_collection
        
        else:
            raise ValueError(f'No predictions found for {dataset}')

    def _denorm_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse transformations (normalization and log) for predictions and target columns.
        
        Parameters
        -----------
        df : pd.DataFrame
            DataFrame that looks like: | timestamp | node | pred | target
            
        Returns
        --------
        pd.DataFrame
            DataFrame with reversed transformations applied
        """
        normalization_method    = self.epiconfig.normalization_method
        
        df_denorm               = df.copy()       
        if self.epiconfig.target_column == 'incidence':
            col_entry_target        = self.column_registration.get_by_name('target')
            transformation_dict     = col_entry_target.transformation

            if transformation_dict is None:
                raise ColEntryMissingTransformationError(entryname = 'target')
            
            for col in ['target','pred']:
                df_denorm = self.reverse_transformations[normalization_method](df_denorm, transformation_dict['normalization'], column = col)
                
                if 'log' in transformation_dict:
                    df_denorm = self.reverse_transformations['log'](df_denorm, transformation_dict['log'], column = col)        
        
        return df_denorm

    def _apply_prediction_timeshift(self, df: pd.DataFrame, timeshift: str) -> pd.DataFrame:
        """shift timestamp by required timeshift"""
        df['timestamp']         = df['timestamp'] + pd.to_timedelta(timeshift) # type: ignore
        return df
    
    def _get_prediction_df_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """returns only the four columns required, as long as they're present"""
        return df[['timestamp','node','target','pred']].copy()        
    
    def _validate_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """checks the presence of required (and only the required) columns"""
        essential_columns = ['timestamp','node','pred','target']
        for cc in essential_columns:
            if cc not in data.columns.tolist():
                raise ValueError(f'{cc} not found in prediction df')
            
        for cc in data.columns:
            if cc not in essential_columns:
                raise ValueError(f'{cc} found in prediction df')
                        
        return data

    def __repr__(self) -> str:
        status_return = ""

        if self.train._contains_data():
            status_return += "train, "
        if self.val._contains_data():
            status_return += "val, "
        if self.test._contains_data():
            status_return += "test, "  

        if len(status_return) == 0:
            status_return += "No predictions found"
        else:
            # remove final space and comma 
            status_return = f"PredictionCollection at {status_return[:-2]}"                     

        return f"<PredictionManager({status_return})>"

# ============= DATA CONTAINERS =============
@dataclass
class PredictionCollection:
    """
    Stores predictions across horizons for one datast (train/val/test)
    
    Accessible by:
    self.get_original(0)

    """
    _data: dict[tuple[int, bool], pd.DataFrame] = field(default_factory=dict)       # a new dictionary is created for each class' instance
    
    def add(self, data: pd.DataFrame, horizon: int, is_original: bool = False):
        """
        Add predictions

        Parameters
        ----------
        data: pd.DataFrame

        horizon: int

        is_original: bool
            if False, then transformed scale, if True then nontransformed        
        """
        self._data[(horizon, is_original)] = data
    
    def get_transformed(self, horizon: int) -> pd.DataFrame:
        return self._data[(horizon, False)]
    
    def get_original(self, horizon: int) -> pd.DataFrame:
        return self._data[(horizon, True)]

    @property
    def horizons(self) -> list[int]:
        return sorted(set(h for h, _ in self._data.keys()))
    
    def _contains_data(self) -> bool:
        if len(self.horizons) > 0:
            return True 
        else:
            return False

    def __repr__(self) -> str:
        if self._contains_data():
            return f"<PredictionCollection(predictions for horizons {self.horizons})>"
        else:
            return f"<PredictionCollection(no predictions)>"