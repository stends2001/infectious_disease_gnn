from dataclasses import dataclass, field
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
    Highly standardized configuration dataclass to be fed into the DataOrchestrator.
    The str function is very very detailed.
    
    Parameters
    ----------
    # ============= REQUIRED =============
    disease:                str
    data_env_dir:           str
    
    # ============= DATES =============
    date_range:             Tuple[str, str] = ('2001-01-01', '2025-01-01')
    temporal_frequency:     Literal['m','w','d']= 'w'
    
    # ============= GEOGRAPHY =============
    nuts_level:             Literal['nuts1', 'nuts2', 'nuts3'] = 'nuts3'
    split_berlin:           bool = True
    
    # ============= TASK CONFIG =============
    horizon_size:           int = 1
    horizon_leadtime:       int = 1
    sequence_length:        int = 1
    lag_num:                int = 1
    prediction_mode: Literal['regression','classification'] = 'regression'
    
    # ============= FEATURES =============
    include_population:     bool = False
    include_gisd:           bool = False    
    lag_column              str  = 'incidence
    weekly_time_index:      bool = True
    daily_time_index:       bool = false
    log_transform:          Optional[List[str]] = None
    log_shift:              float = 1.0    
    
    # ============= NORMALIZATION =============
    normalization_method:   Literal['minmax', 'zscore', 'none'] = 'zscore'

    
    # ============= SPLITTING =============
    split_trainval:         str = '2018-06-01'
    split_valtest:          str = '2019-06-01'
        

    # ============= COLUMN NAMES =============
    temporal_column:        str = 'timestamp'
    target_column:          str = 'incidence'
    id_column:              str = 'node'
    pred_column:            str = 'pred'
    
    # ============= OTHER =============
    incidence_scalar:       int = 10_000    
    verbose:                Literal[0,1,2] = 0 (higher is more output)

    Downstream:
    ----------
    Based on this configuration dataclass, the DataOrchestrator prepares a dataset.

    Examples:
    --------
    >>> config = EpiConfig(
    >>>     disease='influenza',
    >>>     data_env_dir=get_data_env(),
    >>>     date_range=('2017-05-15', '2019-06-01'),
    >>>     horizon_size=2,
    >>>     sequence_length=1,
    >>>     horizon_leadtime=3,
    >>>     lag_num=1,
    >>>     nuts_level='nuts3',
    >>>     log_transform=['incidence'],
    >>>     split_berlin=False,
    >>>     include_population=True
    >>> )    
    """


    # ============= REQUIRED =============
    disease:                str
    data_env_dir:           str
    
    # ============= DATES =============
    date_range:             Tuple[str, str] = ('2001-01-01', '2025-01-01')
    temporal_frequency:     Literal['m','w','d']= 'w'
    
    # ============= GEOGRAPHY =============
    nuts_level:             Literal['nuts1', 'nuts2', 'nuts3'] = 'nuts3'
    split_berlin:           bool = True
    
    # ============= TASK CONFIG =============
    horizon_size:           int = 1
    horizon_leadtime:       int = 1
    sequence_length:        int = 1
    lag_num:                int = 1
    prediction_mode:        Literal['regression','classification'] = 'regression'
    
    # ============= FEATURES =============
    include_population:     bool = False
    include_gisd:           bool = False
    weekly_time_index:      bool = True
    daily_time_index:       bool = False
    monthly_time_index:     bool = False
    lag_column:             str  = 'incidence'
    log_transform:          Optional[List[str]] = None
    log_shift:              float = 1.0    
    
    # ============= NORMALIZATION =============
    normalization_method:   Literal['minmax', 'zscore', 'none'] = 'zscore'
    
    # ============= SPLITTING =============
    split_trainval:         str = '2018-06-01'
    split_valtest:          str = '2019-06-01'
        
    # ============= COLUMN NAMES =============
    temporal_column:        str = 'timestamp'
    target_column:          str = 'incidence'
    id_column:              str = 'node'
    pred_column:            str = 'pred'
    
    # ============= OTHER =============
    incidence_scalar:       int = 10_000    
    verbose:                Literal[0,1,2] = 0

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert dates to timestamps
        self.temporal_frequency = self.temporal_frequency.lower()
        self._min_date          = pd.to_datetime(self.date_range[0])
        self._max_date          = pd.to_datetime(self.date_range[1])
        self._split_trainval    = pd.to_datetime(self.split_trainval)
        self._split_valtest     = pd.to_datetime(self.split_valtest)

        if self._min_date >= self._max_date:
            raise EpiConfigError(f"min_date ({self._min_date.date()}) must be before max_date ({self._max_date.date()})")
        
        if self._split_trainval >= self._split_valtest:
            raise EpiConfigError(f"split_trainval must be before split_valtest")
        
        if not (self._min_date <= self._split_trainval < self._split_valtest <= self._max_date):
            raise EpiConfigError(f"Splits must be within date range")
        
        # Validate horizon config
        if self.horizon_size < 1:
            raise EpiConfigError(f"horizon_size must be >= 1, got {self.horizon_size}")
        if self.horizon_leadtime < 1:
            raise EpiConfigError(f"horizon_leadtime must be >= 1, got {self.horizon_leadtime}")
        if self.sequence_length < 1:
            raise EpiConfigError(f"sequence_length must be >= 1, got {self.sequence_length}")
        if self.lag_num < 1:
            raise EpiConfigError(f"number of lags must be >= 1, got {self.lag_num}")            
        
        # validate task
        if self.target_column == 'incidence' and self.prediction_mode == 'classification':
            raise EpiConfigError(f'Invalid combination of target as incidence and prediction mode as classification. Please adjust')
        
        # time index
        if self.daily_time_index and self.disease != 'covid_daily':
            raise EpiConfigError(f'daily_time_index is only relevant to disease covid_daily. Please adjust')
        # Validate input
        self._validate_datapaths()
        self._validate_current_limitations()

    # ============= VALIDATION FUNCTIONS ==============
    def _validate_datapaths(self):
        """Validates data paths and Wissdatenmounting"""

        if not Path(self.data_env_dir).exists():
            raise WissdatenMountingError(self.data_env_dir)

        path_checks = [
            ("Disease data",            self.get_disease_path),
            ("Population data",         self.get_population_path),
            ("Shape data",              self.get_shapefile_path),
            ("Nuts names",              self.get_nuts_names_path),
            ("Berlin population data",  self.get_population_berlin_districts_path),
        ]
        for name, path_func in path_checks:
            path = path_func()
            if not path.exists():
                raise FileNotFoundError(f"{name} not found: {path}")
            
    def _validate_current_limitations(self):
        # temporal frequency
        if self.temporal_frequency not in ['m','w','d']:
            print(f'{warning_emoji} Currently temporal frequency limited to ["m","w","d"]: {self.temporal_frequency} is invalid and will be reset to "w"')
            self.temporal_frequency = "w"

        # gisd and nuts
        if self.nuts_level == 'nuts1' and self.include_gisd:
            raise EpiConfigError('currently no gisd data for nuts1 exists')
        if self.nuts_level == 'nuts3' and self.split_berlin:
            raise EpiConfigError('no gisd data for berlin districts exists. please remove gisd data or merge berlin')        

    # ============= COMPUTED PROPERTIES =============
    @property
    def lookback_weeks(self) -> int:
        """Total weeks of data needed for sequence + horizon + leadtime."""
        return self.sequence_length + self.horizon_size + self.horizon_leadtime - 1
    
    @property
    def min_date(self) -> pd.Timestamp:
        """Original minimum date (user-specified)."""
        return self._min_date
    
    @property
    def max_date(self) -> pd.Timestamp:
        """Maximum date."""
        return self._max_date
    
    @property
    def min_date_extended(self) -> pd.Timestamp:
        """Extended minimum date accounting for lookback period."""
        return self._min_date - pd.Timedelta(weeks=self.lookback_weeks)
    
    @property
    def split_trainval_ts(self) -> pd.Timestamp:
        """Train/val split as timestamp."""
        return self._split_trainval
    
    @property
    def split_valtest_ts(self) -> pd.Timestamp:
        """Val/test split as timestamp."""
        return self._split_valtest
    
    @property
    def split_columns(self) -> list[str]:
        """Names of split indicator columns."""
        return ['train', 'val', 'test']
    
    @property
    def data_path(self) -> Path:
        """Path object for data directory."""
        return Path(self.data_env_dir)
    
    # ============== Paths returning ===================

    def get_disease_path(self) -> Path:
        """Path to disease CSV file."""
        return self.data_path / 'processed/germany/epidemiology/casedata/survstat' / f'{self.disease}.csv'
    
    def get_population_path(self) -> Path:
        """Path to population CSV file."""
        return self.data_path / 'processed/germany/sociodemography/population_size_03.csv'
    
    def get_population_berlin_districts_path(self) -> Path:
        """Path to berlin - districts population (2024) CSV file."""
        return self.data_path / 'processed/germany/sociodemography/population_size_berlin_districts_03.csv'    
    
    def get_shapefile_path(self) -> Path:
        """Path to shapefile."""
        return self.data_path / f'processed/germany/geospatial/shapefiles/shape_{self.nuts_level}.shp'
    
    def get_nuts_names_path(self) -> Path:
        """Path to NUTS names file."""
        return self.data_path / 'processed/germany/geospatial/harmonization/nuts.tsv'
    
    def get_gisd_path(self) -> Path:
        """Path to gisd data names file."""
        return self.data_path / f'processed/germany/sociodemography/gisd_{self.nuts_level}.csv'        
    
    # ============= SUMMARY METHODS =============
    
    def summary(self) -> str:
        """Return formatted summary of configuration."""
        return f"""
EpiConfig Summary:
==================
Disease:            {self.disease}
NUTS Level:         {self.nuts_level}
Date Range:         {self.min_date.date()} to {self.max_date.date()}
Temporal frequency: {self.temporal_frequency}
Extended Start:     {self.min_date_extended.date()} (lookback: {self.lookback_weeks} weeks)

Task Configuration:
-------------------
Horizon Size:       {self.horizon_size}
Horizon Leadtime:   {self.horizon_leadtime}
Sequence Length:    {self.sequence_length}

Features:
--------
lag_num:            {self.lag_num}
lag_column:         {self.lag_column}
weekly time index:  {self.weekly_time_index}
daily time index:   {self.daily_time_index}
population size:    {self.include_population}

Splits:
-------
Train/Val:          {self._split_trainval.date()}
Val/Test:           {self._split_valtest.date()}

Processing:
-----------
Normalization:      {self.normalization_method}
Log Transform:      {self.log_transform}
Split Berlin:       {self.split_berlin}
"""
    
    def __str__(self) -> str:
        return self.summary()
    
    def __repr__(self) -> str:
        return (f"EpiConfig(disease={self.disease!r}, "
                f"nuts_level={self.nuts_level!r}, "
                f"date_range={self.date_range!r}, "
                f"horizon_size={self.horizon_size}, "
                f"sequence_length={self.sequence_length})")
