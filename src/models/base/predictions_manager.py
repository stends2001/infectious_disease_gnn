import pandas as pd 
import geopandas as gpd
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Union
from ...dataloading.epidataorchestration.normalization import reverse_log, reverse_minmax_scaling, reverse_zscore_scaling
from ...dataloading.epidataorchestration.epidataorchestrator import EpiDataOrchestrator
from ...dataloading.epidataorchestration.epiconfig import EpiConfig
from ...dataloading.epidataorchestration.column_registry import ColumnRegistration
from ...dataloading.epidataorchestration.issues import ColEntryMissingTransformationError
from ...dataloading.epidataorchestration.temporal_summary import EpiDataTemporalSummary, TemporalError

from ...utils import check_dataset
from ..utils.loss.poissonloss import convert_poisson_predictions
from datetime import timedelta
from dateutil.relativedelta import relativedelta

class InvalidPredictionsError(Exception):

    def __init__(self, message: str):
        super().__init__(f'Invalid Predictions {message}')

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

    def __init__(self, data_orchestrator: EpiDataOrchestrator, column_registration: ColumnRegistration, temporal_summary: EpiDataTemporalSummary):

        self.train = PredictionCollection()
        self.val   = PredictionCollection()
        self.test  = PredictionCollection()
        
        self.epidata_orchestrator = data_orchestrator
        self.epiconfig          = data_orchestrator.config
        self.column_registration= column_registration
        self.temporal_summary   = temporal_summary
        self._setup_reverse_transformations()

    def _setup_reverse_transformations(self):
        self.reverse_transformations = {
            'minmax': reverse_minmax_scaling,
            'zscore': reverse_zscore_scaling,
            'log'   : reverse_log
        }

    def _shift_prediction_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        dfc                                 = df.copy()

        if self.epiconfig.temporal_frequency == 'd':
            delta = timedelta(days=self.epiconfig.horizon_leadtime)
        elif self.epiconfig.temporal_frequency == 'w':
            delta = timedelta(weeks=self.epiconfig.horizon_leadtime)
        elif self.epiconfig.temporal_frequency == 'm':
            delta = relativedelta(months=self.epiconfig.horizon_leadtime)
        else:
            raise TemporalError(f"Unknown frequency: {self.epiconfig.temporal_frequency}")

        timestamps: pd.Series[pd.Timestamp] = dfc[self.epiconfig.temporal_column]
        dfc[self.epiconfig.temporal_column]  = timestamps + delta
        return dfc

    @check_dataset()
    def add_horizon_predictions(self, dataset: Literal['train','val','test'], horizon_df: pd.DataFrame, horizon: int, additional_transformation: bool = False, transf: Optional[str] = None, transf_args: Optional[Dict[str,Union[str,float]]] = None):

        df              = self._get_prediction_df_cols(horizon_df)   
        df_validated    = self._validate_columns(df)

        # shift timestamps to mathc the PREDICTION timestamps
        df_shifted_t    = self._shift_prediction_timestamp(df_validated)
        df_filtered     = self._filter_by_dataset_timerange(df_shifted_t, dataset)
        
        self._validate_predictions(df_filtered, dataset)

        timesteps_ahead = int(horizon + self.epiconfig.horizon_leadtime)
        timesteps_unit  = self.epiconfig.temporal_frequency
        
        if dataset == 'train':
            prediction_collection = self.train 

        elif dataset == 'val':
            prediction_collection = self.val

        elif dataset == 'test':
            prediction_collection = self.test 

        else:
            raise ValueError(f'{dataset} invalid dataset. Supply "train", "val" or "test"')      

        # Apply additional transformations if needed
        if additional_transformation:
            if transf == 'poisson_sampling':
                if 'sampling_mode' in transf_args.keys():
                    sampling_mode = transf_args['sampling_mode'] 
                    df_filtered['pred'] = convert_poisson_predictions(df_filtered['pred'], sampling_mode)
                else:
                    raise ValueError('When using poisson_sampling, please supply "sampling_mode"')
            else:
                raise ValueError(f'Only "poisson_sampling" supported for additional_transformation')

        # Add both normalized and denormalized versions
        prediction_collection.add(df_filtered.copy(), horizon=horizon, is_original=False)
        prediction_collection.add(self._denorm_predictions(df_filtered), horizon=horizon, is_original=True)

    @check_dataset()
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
        normalization_method = self.epiconfig.normalization_method

        if normalization_method == 'none':
            return df
        
        df_denorm = df.copy()

        if self.epiconfig.predict_difference:
            df_denorm = self._get_anchor_for_merge(df_denorm)
        
        if self.epiconfig.target_column == 'incidence':
            col_entry_target    = self.column_registration.get_by_name('target')
            transformation_dict = col_entry_target.transformation

            if transformation_dict is None:
                raise ColEntryMissingTransformationError(entryname='target')
            
            for col in ['target', 'pred']:
                df_denorm = self.reverse_transformations[normalization_method](
                    df_denorm, transformation_dict['normalization'], column=col
                )
                
                if 'log' in transformation_dict:
                    df_denorm = self.reverse_transformations['log'](
                        df_denorm, transformation_dict['log'], column=col
                    )

            if self.epiconfig.predict_difference and 'delta' in transformation_dict:
                anchor_col = transformation_dict['delta']['anchor_col']
                for col in ['target', 'pred']:
                    df_denorm[col] = df_denorm[col] + df_denorm[anchor_col]
                df_denorm = df_denorm.drop(columns=[anchor_col])
        
        return df_denorm

    def _apply_prediction_timeshift(self, df: pd.DataFrame, timeshift: str) -> pd.DataFrame:
        """shift timestamp by required timeshift"""
        df['timestamp']         = df['timestamp'] + pd.to_timedelta(timeshift) # type: ignore
        return df
    
    def _get_prediction_df_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [self.epiconfig.temporal_column, self.epiconfig.id_column, 'target', 'pred']
        return df[cols].copy()    
        
    def _validate_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        essential_columns = [self.epiconfig.temporal_column, self.epiconfig.id_column, 'target', 'pred']
        
        for cc in essential_columns:
            if cc not in data.columns.tolist():
                raise ValueError(f'{cc} not found in prediction df')
            
        for cc in data.columns:
            if cc not in essential_columns:
                raise ValueError(f'{cc} found in prediction df but not expected')
                        
        return data
    
    def _get_anchor_for_merge(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetches the anchor column from data_final.data and merges it onto df
        by timestamp and node id, so _denorm_predictions can use it for delta reversal.
        """
        anchor_col = f'{self.epiconfig.target_column}_anchor'
        merge_keys = [self.epiconfig.temporal_column, self.epiconfig.id_column]

        source_df = self.epidata_orchestrator.data_final.data[merge_keys + [anchor_col]]
        
        return df.merge(source_df, on=merge_keys, how='left')

    @check_dataset()
    def _filter_by_dataset_timerange(self, df: pd.DataFrame, dataset: Literal['train','val','test']) -> pd.DataFrame:
        if dataset == 'train':
            min_date = self.temporal_summary.min_date
            max_date = self.temporal_summary.split_trainval
        elif dataset == 'val':
            min_date = self.temporal_summary.split_trainval
            max_date = self.temporal_summary.split_valtest
        elif dataset == 'test':
            min_date = self.temporal_summary.split_valtest
            max_date = self.temporal_summary.max_date_extended

        dfc = df.copy()
        mask = (dfc['timestamp'] >= min_date) & (dfc['timestamp'] < max_date)
        filtered = dfc[mask].reset_index(drop=True)        
        return filtered

    @check_dataset()
    def _validate_predictions(self, df: pd.DataFrame, dataset):
        unique_timestamp_ids = pd.to_datetime(
            df[self.epiconfig.temporal_column].unique()
        )
        expected_timerange  = self.temporal_summary.get_daterange_dataset(dataset, reference = 'target')
        
        rng                 = pd.date_range(start=expected_timerange[0], end=expected_timerange[1], freq=pd.infer_freq(sorted(unique_timestamp_ids)))
        if len(set(rng) - set(unique_timestamp_ids))>0:
            missing_timesteps = set(rng) - set(unique_timestamp_ids)
            raise InvalidPredictionsError(f'Prediction timestamps arent complete. Missing: {missing_timesteps}')

        if len(set(unique_timestamp_ids)-set(rng))>0:
            leftover_timesteps = set(unique_timestamp_ids)-set(rng)
            raise InvalidPredictionsError(f'Prediction timestamps are too numerous. Leftover: {leftover_timesteps}')

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