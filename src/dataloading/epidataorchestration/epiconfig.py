from dataclasses import dataclass, field
from src.utils.helpers import get_data_env
from typing import Literal, Optional, Tuple, List
from pathlib import Path
import pandas as pd

from ...utils.textformatting import warning_emoji
from ...utils.exceptions import WissdatenMountingError

class EpiConfigError(Exception):
    def __init__(self, explanation: str):
        statement = "Epiconfig couldn't be loaded" + "\n" + explanation
        super().__init__(statement)

@dataclass
class EpiConfig:
    """ 
    Note
    ----
    the dates listed in this class are solely strings.
    for proper dealing with timestamps, please refer to TemporalSummary   
    """

    # ============= REQUIRED =============
    disease:                str
    
    # ============= TEMPORAL =============
    min_date:               str = '2011-01-01'
    max_date:               str = '2020-06-01'
    split_trainval:         str = '2018-06-01'
    split_valtest:          str = '2019-06-01'
    temporal_frequency:     Literal['m','w','d']= 'w'
    
    # ============= GEOGRAPHY =============
    nuts_level:             Literal['nuts1', 'nuts2', 'nuts3'] = 'nuts3'
    split_berlin:           bool = True
    
    # ============= TASK CONFIG =============
    horizon_size:           int = 1
    horizon_leadtime:       int = 1
    sequence_length:        int = 1
    lag_num:                int = 1
    num_quantiles:          int = 1
    prediction_mode:        Literal['regression','classification'] = 'regression'
    predict_difference:     bool = False
    
    # ============= FEATURES =============
    time_index_d:           bool = False
    time_index_w:           bool = True
    time_index_m:           bool = False
    lag_column:             str  = 'incidence'

    feature_population_size:bool = False      
    feature_popdens:        bool = False
    feature_gisd:           bool = False
    feature_popage:         bool = False
    
    # ============= NORMALIZATION =============
    normalization_method:   Literal['minmax', 'zscore', 'none'] = 'zscore'
    log_transform:          Optional[List[str]] = None
    log_shift:              float = 1.0        
            
    # ============= COLUMN NAMES =============
    temporal_column:        str = 'timestamp'
    target_column:          str = 'incidence'
    id_column:              str = 'nuts_node'
    pred_column:            str = 'pred'
    
    # ============= OTHER =============
    incidence_scalar:       int = 10_000    
    verbose:                Literal[0,1,2] = 0

    def __post_init__(self):

        # Validate input
        self._validate_horizon_inputs()
        self._validate_datapaths()
        self._validate_current_limitations()

    # ============= VALIDATION FUNCTIONS ==============
    def _validate_horizon_inputs(self):
        # Validate horizon config
        if self.horizon_size < 1:
            raise EpiConfigError(f"horizon_size must be >= 1, got {self.horizon_size}")
        if self.horizon_leadtime < 1:
            raise EpiConfigError(f"horizon_leadtime must be >= 1, got {self.horizon_leadtime}")
        if self.sequence_length < 1:
            raise EpiConfigError(f"sequence_length must be >= 1, got {self.sequence_length}")
        if self.lag_num < 1:
            raise EpiConfigError(f"number of lags must be >= 1, got {self.lag_num}")  
        
    def _validate_datapaths(self):

        if not Path(get_data_env()).exists():
            raise WissdatenMountingError(get_data_env())

        path_checks = [
            ("Disease data",            self.get_disease_path),
            ("Population data",         self.get_population_path),
            ("Shape data",              self.get_nuts_shapefile_path),
            ("Nuts names",              self.get_nuts_harmonization_path),
            ("Berlin population data",  self.get_population_berlin_districts_path),
        ]
        for name, path_func in path_checks:
            path = path_func()
            if not path.exists():
                raise FileNotFoundError(f"{name} not found: {path}")
            
    def _validate_current_limitations(self):
        # validate task
        if self.target_column == 'incidence' and self.prediction_mode == 'classification':
            raise EpiConfigError(f'Invalid combination of target as incidence and prediction mode as classification. Please adjust')
        
        if self.num_quantiles < 1:
            raise EpiConfigError(f'Number of predicted quantiles must be at least 1')
        
        if self.num_quantiles != 1 and self.prediction_mode != 'regression':
            raise EpiConfigError(f'when prediction task is not regression, num_quantiles will not be taken into account. Please adjust num_quantiles back to 1')
        
        # time indices
        if self.time_index_d and self.disease != 'covid_daily':
            raise EpiConfigError(f'time_index_d is only relevant to disease covid_daily. Please adjust')

        # temporal frequency
        if self.temporal_frequency not in ['m','w','d']:
            print(f'{warning_emoji} Currently temporal frequency limited to ["m","w","d"]: {self.temporal_frequency} is invalid and will be reset to "w"')
            self.temporal_frequency = "w" 

        if self.predict_difference and self.horizon_leadtime > 1:
            raise EpiConfigError(
                f"predict_difference=True is only supported with horizon_leadtime=1. "
                f"Got horizon_leadtime={self.horizon_leadtime}. "
                f"For multi-step forecasting with deltas, set horizon_leadtime=1 and use horizon_size > 1 instead."
            )            

        # features
        if self.feature_popdens:
            if self.nuts_level != 'nuts3':
                raise EpiConfigError('currently population-density-feature data only exists for nuts3. Please remove this feature, or switch to nuts3.')
            
            if self.split_berlin:
                raise EpiConfigError('currently population-density-feature data only exists for Berlin as entirety, not split. Please adjust.')

        if self.feature_gisd:
            if self.nuts_level not in ['nuts2','nuts3']:
                raise EpiConfigError('GISD data only available for nuts2 or nuts3. Please adjust')
            if self.split_berlin:
                raise EpiConfigError('currently GISD data only exists for Berlin as entirety, not split. Please adjust.')        

        if self.feature_popage:
            if self.nuts_level != 'nuts3':
                raise EpiConfigError('population age only available when nuts is nuts3')
            
        if self.feature_population_size:
            if not self.feature_popage:
                raise EpiConfigError('currently only popsize supported if popage is included as well')

    # ============= COMPUTED PROPERTIES =============
    @property
    def split_columns(self) -> list[str]:
        """Names of split indicator columns."""
        return ['train', 'val', 'test']
    
    @property
    def data_path(self) -> Path:
        """Path object for data directory."""
        return Path(get_data_env())
    
    # ============== Paths returning ===================

    def get_disease_path(self) -> Path:
        """Path to disease CSV file."""
        return self.data_path / 'processed/germany/epidemiology/casedata/survstat' / f'{self.disease}.csv'
    
    def get_population_path(self) -> Path:
        """Path to population CSV file."""
        return self.data_path / 'processed/germany/sociodemography/population_size_03.csv'
    
    def get_population_density_path(self) -> Path:
        """Path to population density CSV file."""
        return self.data_path / f'processed/germany/sociodemography/population_density_{self.nuts_level}.csv'        

    def get_gisd_path(self) -> Path:
        """Path to gisd CSV file."""
        return self.data_path / f'processed/germany/sociodemography/gisd_{self.nuts_level}.csv'        

    def get_population_age_path(self) -> Path:
        """Path to population age CSV file."""
        return self.data_path / f'processed/germany/sociodemography/population_age_{self.nuts_level}.csv'        

    def get_population_berlin_districts_path(self) -> Path:
        """Path to berlin - districts population (2024) CSV file."""
        return self.data_path / 'processed/germany/sociodemography/population_size_berlin_districts_03.csv'    
    
    def get_nuts_shapefile_path(self) -> Path:
        """Path to shapefile."""
        return self.data_path / f'processed/germany/geospatial/shapefiles/shape_{self.nuts_level}.shp'
    
    def get_nuts_harmonization_path(self) -> Path:
        """Path to NUTS names file."""
        return self.data_path / 'processed/germany/geospatial/harmonization/nuts.tsv'
    
    # ============= SUMMARY METHODS =============
    
    def summary(self) -> str:
        """Return formatted summary of configuration."""
        return f"""
"""
    
    def __str__(self) -> str:
        return self.summary()
    
    def __repr__(self) -> str:
        return (f"EpiConfig(disease={self.disease!r}, "
                f"nuts_level={self.nuts_level!r}, "
                f"min_date={self.min_date!r}, "
                f"max_date={self.max_date!r}, "                
                f"horizon_size={self.horizon_size}, "
                f"sequence_length={self.sequence_length})")
