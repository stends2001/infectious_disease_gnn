import pandas as pd 
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Union
from datetime import timedelta
from dateutil.relativedelta import relativedelta

from ..issues import InvalidPredictionsError, MissingPredictionsError
from ...dataloading.epidataorchestration.normalization import reverse_log, reverse_minmax_scaling, reverse_zscore_scaling
from ...dataloading.epidataorchestration.epidataorchestrator import EpiDataOrchestrator
from ...dataloading.epidataorchestration.column_registry import ColumnRegistration
from ...dataloading.epidataorchestration.temporal_summary import EpiDataTemporalSummary, TemporalError
from ...utils import check_dataset

from ..utils.loss.poissonloss import convert_poisson_predictions

# ============= Predictions Manager =============
class PredictionManager:
    """
    Manages predictions in a centralized class.
    Each model has an instance, at `model.predictions`

    Within this class, each of ['train','val','test'] is a collection of prediction - dataframes per horizon

    Parameters
    ----------
    epiconfig: EpiConfig
        the dataorchestrator's configuration object
    column_registration: ColumnRegistration
        the dataorchestrator's columnregistration object
    temporal_summary: EpiDataTemporalSummary
        the temporal summary of the epidataorchestrator

    Note
    ----
    The predictions and the targets are shifted here,
    meaning that with respect to the final dataorchestrator's df,
    the timestamp/target is shifted.

    Methods
    -------
        External functions
    - `add_horizon_predictions()`
        adds data to PredictionManager: calls all necessary helper functions internally
    - `get_preds()`
        gets data from PredictionManager

        Helper functions
    - `_return_dataset()`
    - `_setup_reverse_transformations()`
    - `_setup_required_columns()`
    - `_shift_prediction_timestamp()`
    - `_denorm_predictions()`
    - `_validate_columns()`
    - `_get_anchor_for_merge()`
    - `_filter_by_dataset_timerange()`
    - `_validate_predictions_temporally()`
    - `_aggregate_predictions_spatially()`

    Examples
    --------

    See Also
    --------
    PredictionCollection

    """

    def __init__(self, 
                 data_orchestrator:     EpiDataOrchestrator, 
                 column_registration:   ColumnRegistration,
                 temporal_summary:      EpiDataTemporalSummary):

        self.train = PredictionCollection()
        self.val   = PredictionCollection()
        self.test  = PredictionCollection()
        
        self.epidata_orchestrator   = data_orchestrator
        self.epiconfig              = data_orchestrator.config
        self.column_registration    = column_registration
        self.temporal_summary       = temporal_summary

        # call helper functions
        self._setup_reverse_transformations()
        self._setup_required_columns()

    def add_horizon_predictions(self, 
                                dataset:                    Literal['train','val','test'], 
                                horizon_df:                 pd.DataFrame, 
                                horizon:                    int, 
                                additional_transformation:  bool = False, 
                                transf:                     Optional[str] = None, 
                                transf_args:                Optional[Dict[str,Union[str,float]]] = None):
        """
        Adds a prediction df to the prediction manager
        Internally validates columns, temporal-axis, and everything else.

        TODO: validate that the horizon df doesn't exist yet
        """
 
        df_validated    = self._validate_columns(horizon_df)

        # shift timestamps to match the PREDICTION timestamps
        df_shifted_t    = self._shift_prediction_timestamp(df_validated)
        df_filtered     = self._filter_by_dataset_timerange(df_shifted_t, dataset)
        
        self._validate_predictions_temporally(df_filtered, dataset)
                
        # get prediction_collection
        prediction_collection = self._return_dataset(dataset)

        # ===================================================================================
        # TODO
        # this piece needs to be rewritten. Probably put into 
        # col registry somehow
        # Apply additional transformations if needed
        if additional_transformation:
            if transf == 'poisson_sampling':
                if 'sampling_mode' in transf_args.keys():
                    sampling_mode = transf_args['sampling_mode'] 
                    df_filtered[self.pred_cols] = convert_poisson_predictions(df_filtered[self.pred_cols], sampling_mode)
                else:
                    raise ValueError('When using poisson_sampling, please supply "sampling_mode"')
            else:
                raise ValueError(f'Only "poisson_sampling" supported for additional_transformation')
        # ===================================================================================

        # Add both normalized and denormalized versions, as well as a national aggregate for the denormalized
        df_transformed      = df_filtered.copy()
        df_nontransformed   = self._denorm_predictions(df_transformed.copy())
        df_aggregated       = self._aggregate_predictions_spatially(df_nontransformed.copy())
        
        # transformed - regional
        prediction_collection.add(df_transformed, 
                                  horizon               = horizon, 
                                  is_original           = False, 
                                  spatially_aggregated  = False)
        # non-transformed - regional
        prediction_collection.add(df_nontransformed, 
                                  horizon               = horizon, 
                                  is_original           = True, 
                                  spatially_aggregated  = False)
        # non-transformed - aggregated
        prediction_collection.add(df_aggregated,
                                  horizon               = horizon, 
                                  is_original           = True, 
                                  spatially_aggregated  = True)

    def get_preds(self, dataset: Literal['train','val','test']) -> 'PredictionCollection':
        """
        get predictions from any of the datasets
        """
        return self._return_dataset(dataset)

    @check_dataset()
    def _return_dataset(self, dataset: Literal['train','val','test']) -> 'PredictionCollection':
        """
        Returns the PredictionCollection at attribute train/val/test
        if invalid dataset, error is raised from decorator
        """
        if dataset == 'train':
            return self.train
        elif dataset == 'val':
            return self.val 
        elif dataset == 'test':
            return self.test 

    def _setup_reverse_transformations(self):
        """sets the reverse transformation functions: from normalized data -> non-normalized data"""
        self.reverse_transformations = {
            'minmax': reverse_minmax_scaling,
            'zscore': reverse_zscore_scaling,
            'log'   : reverse_log
        }

    def _setup_required_columns(self):
        """
        sets the required columns in the model-specific prediction dfs.
        The columns present should be:
        - temporal_column (epiconfig.temporal_column)
        - node_column (epiconfig.id_column)
        - 'target'
        - prediction columns:
            - if num_qunatiles = 0 => only 'pred'
            - else all registered pred columns except for 'pred'
        """        
        cols        = [self.epiconfig.temporal_column, self.epiconfig.id_column, 'target']

        if self.epiconfig._num_quantiles == 0:
            pred_cols= ['pred']
        else:
            pred_cols= [c for c in self.column_registration.pred_columns if c != 'pred']

        self.pred_cols          = pred_cols
        self.required_columns   = cols + pred_cols

    def _shift_prediction_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        In the DataOrchestrator the target has been shifted. 
        In other words, the combination of values for columns
        | 'timestamp' | 'target' | has a temporal discrepancy,
        where 'timestamp' refers to t0; the point in time of the
        most recent observation (related to features, not target).

        In order to plot and analyze correctly, the prediction
        managers deal with the 'correct' temporal axis: target/pred
        are looked at from the correct point in time. Thus, we shift
        the target by {epiconfig.horizon_leadtime} timesteps, of the 
        proper temporal_frequency.
        """
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

    def _denorm_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Denormalizes the normalized predictions in df using the parameters
        from the column registry

        TODO: make this cases-friendly
        """
        normalization_method = self.epiconfig.normalization_method
        
        df_denorm = df.copy()

        # when predicting difference: get anchor
        if self.epiconfig.predict_difference:
            df_denorm = self._get_anchor_for_merge(df_denorm)            
        
        if normalization_method:

            col_entry_target    = self.column_registration.get_by_name('target')
            transformation_dict = col_entry_target.transformation_params            

            # normalize all columns that need to be
            for col in self.column_registration.pred_columns + ['target']:
                if col in df.columns:
                    df_denorm = self.reverse_transformations[normalization_method](
                        df_denorm, transformation_dict['normalization'], column=col
                    )
                # de-log all columns that need to be
                if 'log' in transformation_dict:
                    df_denorm = self.reverse_transformations['log'](
                        df_denorm, transformation_dict['log'], column=col
                    )

            if self.epiconfig.predict_difference and 'delta' in transformation_dict:
                anchor_col = transformation_dict['delta']['anchor_col']
                for col in self.column_registration.pred_columns + ['target']:
                    if col in df.columns:
                        df_denorm[col] = df_denorm[col] + df_denorm[anchor_col]
                df_denorm = df_denorm.drop(columns=[anchor_col])
        
        return df_denorm
        
    def _validate_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the df with all the required columns
        Also validates that these are all present
        """
        essential_columns = self.required_columns
        
        for cc in essential_columns:
            if cc not in data.columns.tolist():
                raise InvalidPredictionsError(f'column {cc} not found in prediction df')
            
        for cc in data.columns:
            if cc not in essential_columns:
                raise InvalidPredictionsError(f'column {cc} found in prediction df but not expected')
                        
        return data[self.required_columns].copy()  
    
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
        """
        Filters out data from outside the expected period,
        using the dates specified in TemporalSummary
        """
        if dataset == 'train':
            min_date = self.temporal_summary.min_date
            max_date = self.temporal_summary.split_trainval
        elif dataset == 'val':
            min_date = self.temporal_summary.split_trainval
            max_date = self.temporal_summary.split_valtest
        elif dataset == 'test':
            min_date = self.temporal_summary.split_valtest
            max_date = self.temporal_summary.max_date_extended

        dfc     = df.copy()
        mask    = (dfc[self.epiconfig.temporal_column] >= min_date) & (dfc[self.epiconfig.temporal_column] < max_date)
        filtered= dfc[mask].reset_index(drop=True)        
        return filtered

    @check_dataset()
    def _validate_predictions_temporally(self, df: pd.DataFrame, dataset: Literal['train','val','test']):
        """
        Validates the temporal axis that is expected
        If this is not the case, InvalidPredictionsError is thrown
        """
        dt_index = pd.DatetimeIndex(
            pd.to_datetime(df[self.epiconfig.temporal_column])
        ).drop_duplicates().sort_values()

        expected_timerange = self.temporal_summary.get_daterange_dataset(
            dataset,
            reference='target'
        )

        freq = pd.infer_freq(dt_index)
        if freq is None:
            raise InvalidPredictionsError(
                "Could not infer frequency from prediction timestamps."
            )

        range_expected = pd.date_range(
            start=expected_timerange[0],
            end=expected_timerange[1],
            freq=freq
        )

        missing = range_expected.difference(dt_index)
        if not missing.empty:
            raise InvalidPredictionsError(
                f"Prediction timestamps are incomplete. Missing: {missing.tolist()}"
            )

        leftover = dt_index.difference(range_expected)
        if not leftover.empty:
            raise InvalidPredictionsError(
                f"Prediction timestamps are too numerous. Leftover: {leftover.tolist()}"
            )

    def _aggregate_predictions_spatially(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates regional predictions to a national level.
        
        For 'cases': simple sum across regions per timestep.
        For 'incidence': population-weighted aggregation —
            converts to cases, sums nationally, divides by total population.
        """
        dfc = df.copy()
        columns_to_aggregate = self.pred_cols + ['target']
        temporal_col = self.epiconfig.temporal_column

        if self.epiconfig.target_column == 'cases':
            aggregated = (
                dfc
                .groupby(temporal_col)[columns_to_aggregate]
                .sum()
                .reset_index()
            )

        elif self.epiconfig.target_column == 'incidence':
            population_data = self.epidata_orchestrator.data_context.population_size

            dfc['year'] = dfc[temporal_col].dt.year
            dfc = dfc.merge(population_data, on=[self.epiconfig.id_column, 'year'])

            # Convert incidence rates -> raw cases per region
            rate_denominator = self.epiconfig.incidence_scalar
            for col in columns_to_aggregate:
                dfc[f'{col}_cases'] = dfc[col] * dfc['population_size'] / rate_denominator

            # Sum cases and population nationally per timestep
            case_cols = [f'{col}_cases' for col in columns_to_aggregate]
            national = (
                dfc
                .groupby(temporal_col)[case_cols + ['population_size']]
                .sum()
                .reset_index()
            )

            # Convert back to national incidence rate
            for col in columns_to_aggregate:
                national[col] = national[f'{col}_cases'] / national['population_size'] * rate_denominator

            aggregated = national[[temporal_col] + columns_to_aggregate]

        else:
            raise ValueError(f"Unsupported target_column '{self.epiconfig.target_column}' for spatial aggregation.")

        return aggregated

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
    Predictions are stored under three variables:
    - horizon:              int
    - is_original:          bool
    - spatially_aggregated: bool
    
    Accessible by:
    self.get_original(0)

    """
    _data: dict[tuple[int, bool, bool], pd.DataFrame] = field(default_factory=dict)       # a new dictionary is created for each class' instance
    
    def add(self, data: pd.DataFrame, horizon: int, is_original: bool = False, spatially_aggregated: bool = False):
        """
        Add predictions

        Parameters
        ----------
        data: pd.DataFrame

        horizon: int

        is_original: bool
            if False, then transformed scale, if True then nontransformed        
        """
        self._data[(horizon, is_original, spatially_aggregated)] = data
    
    def get(self, horizon: int, is_original: bool, spatially_aggregated: bool) -> pd.DataFrame:

        key = (horizon, is_original, spatially_aggregated)
        if key not in self._data:
            raise MissingPredictionsError(f"No predictions found for horizon={horizon}, is_original={is_original}, spatially_aggregated={spatially_aggregated}. Available: {list(self._data.keys())}")
        return self._data[key]

    @property
    def horizons(self) -> list[int]:
        """return a list of horizon integers for which predictions are found"""
        return sorted(set(h for h, _, _ in self._data.keys()))
    
    def _contains_data(self) -> bool:
        """return bool for whether or not predictions exist"""
        return bool(self.horizons)
    
    def __repr__(self) -> str:
        if self._contains_data():
            return f"<PredictionCollection(predictions for horizons {self.horizons})>"
        else:
            return f"<PredictionCollection(no predictions)>"