from dataclasses import dataclass
from typing import Literal, Optional, Dict, List
from pathlib import Path
import pandas as pd

from ...utils.helpers import get_data_env
from ...utils.textformatting import align, return_header_line
from ...issues.errors import WissdatenMountingError
from ...issues import IssueReport
from .issues import EpiConfigWarning, EpiConfigValidationError, EpiConfigLimitationError


@dataclass
class EpiConfig:
    """ 
    Note
    ----
    the dates listed in this class are solely strings.
    for proper dealing with timestamps, please refer to TemporalSummary   
    """

    # ============= MAIN =============
    disease:                str
    
    # ============= TEMPORAL =============
    temporal_frequency:     Literal['m','w','d']= 'w'
    min_date:               str = '2011-01-01'
    max_date:               str = '2020-06-01'
    split_trainval:         str = '2018-06-01'
    split_valtest:          str = '2019-06-01'
    
    # ============= GEOGRAPHY =============
    nuts_level:             Literal['nuts1', 'nuts2', 'nuts3'] = 'nuts3'
    split_berlin:           bool = True
    
    # ============= TASK CONFIG =============
    horizon_size:           int = 1
    horizon_leadtime:       int = 1
    quantiles:              Optional[List[float]] = None
    prediction_mode:        Literal['regression','classification'] = 'regression'
    predict_difference:     bool = False
    
    # ============= FEATURES =============
    time_index_d:           bool = False
    time_index_w:           bool = True
    time_index_m:           bool = False
    lag_column:             str  = 'incidence'
    lag_num:                int  = 1
    sequence_length:        int  = 1
    incidence_scalar:       int  = 10_000    

    feature_popsize:        bool = False      
    feature_popdens:        bool = False
    feature_gisd:           bool = False
    feature_popage:         bool = False
    
    # ============= NORMALIZATION =============
    normalization_method:   Optional[Literal['minmax', 'zscore']] = 'zscore'
    log_transform:          Optional[List[str]] = None
    log_shift:              float = 1.0        
            
    # ============= COLUMN NAMES =============
    temporal_column:        str = 'timestamp'
    target_column:          str = 'incidence'
    id_column:              str = 'nuts_node'
    pred_column:            str = 'pred'
    
    # ============= OTHER =============
    verbose:                Literal[0,1,2] = 0

    def __post_init__(self):

        # Validate input
        self._validate_input()
        self._validate_datapaths()
        self._validate_current_limitations()
        self._validate_warnings()

        self._set_hidden_attributes()

        self._classify_attributes()

    # ============= VALIDATION FUNCTIONS ==============
    def _validate_datapaths(self) -> None:

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
            
    def _validate_input(self) -> None:
        """
        Validates discrepancies in the initialization of an EpiConfig instance. These represent
        actual issues or errors, so an EpiConfigError is thrown suggesting to adjust the input.
        """
        validation_errors = []
        # temporal-related 
        if self.horizon_size < 1:
            validation_errors.append(EpiConfigValidationError(f"horizon_size must be >= 1, got {self.horizon_size}"))
        
        if self.horizon_leadtime < 1:
            validation_errors.append(EpiConfigValidationError(f"horizon_leadtime must be >= 1, got {self.horizon_leadtime}"))
        
        if self.sequence_length < 1:
            validation_errors.append(EpiConfigValidationError(f"sequence_length must be >= 1, got {self.sequence_length}"))
        
        if self.lag_num < 1:
            validation_errors.append(EpiConfigValidationError(f"number of lags must be >= 1, got {self.lag_num}"))
        
        if self.time_index_d and self.disease != 'covid_daily':
            validation_errors.append(EpiConfigValidationError(f'time_index_d is only relevant to disease covid_daily'))

        # task-related
        if self.target_column == 'incidence' and self.prediction_mode == 'classification':
            validation_errors.append(EpiConfigValidationError(f'Invalid combination of target == "incidence" prediction_mode as "classification"'))
        
        if self.quantiles:
            if not isinstance(self.quantiles, List):
               validation_errors.append(EpiConfigValidationError(f'Invalid input for quantiles ({self.quantiles}). Must be a List[float]'))
            
            for quantile in self.quantiles:
                if quantile >= 1 or quantile <= 0:
                   validation_errors.append(EpiConfigValidationError(f'Invalid input for quantiles ({self.quantiles}). Must be a List of values 0 < quantile < 1'))
        
        if len(validation_errors):
            raise IssueReport(validation_errors, context = "EpiConfig could not be created")
        
    def _validate_current_limitations(self) -> None:
        """
        Validates any issues in the initialization of an EpiConfig instance. 
        These represent CURRENT limitations, which are also things for me to develop further.
        An CurrentEpiConfigError is thrown suggesting to adjust the input.
        """

        limitation_errors = []

        # temporal frequency
        if self.temporal_frequency not in ['m','w','d']:
            limitation_errors.append(EpiConfigLimitationError(f'invalid valid for temporal_frequency (currently). Value must be in ["m","w","d"]'))

        # predicting deltas
        if self.predict_difference and self.horizon_leadtime > 1:
            limitation_errors.append(EpiConfigLimitationError(
                f"predict_difference=True is only supported with horizon_leadtime=1 (currently) "
                f"Got horizon_leadtime={self.horizon_leadtime}. "
                f"For multi-step forecasting with deltas, set horizon_leadtime=1 and use horizon_size > 1 instead."
            ))            
        
        # features
        # population density
        if self.feature_popdens:
            if self.nuts_level != 'nuts3':
                limitation_errors.append(EpiConfigLimitationError('Currently population-density-feature data only exists for nuts3. Please remove this feature, or switch to nuts3.'))
            
            if self.split_berlin:
                limitation_errors.append(EpiConfigLimitationError('Currently population-density-feature data only exists for Berlin as entirety, not split. Please adjust.'))
        # gisd
        if self.feature_gisd:
            if self.nuts_level not in ['nuts2','nuts3']:
                limitation_errors.append(EpiConfigLimitationError('GISD data only available for nuts2 or nuts3. Please adjust'))
            if self.split_berlin:
                limitation_errors.append(EpiConfigLimitationError('Crrently GISD data only exists for Berlin as entirety, not split. Please adjust.'))
            if pd.to_datetime(self.max_date) > pd.to_datetime('2021-12-31'):
                limitation_errors.append(EpiConfigLimitationError('Currently GISD data only available until 2021 while simulation max date exceeds that. Either remove the gisd data as feature, or decrease the timespawn.'))
        # population age
        if self.feature_popage:
            if self.nuts_level != 'nuts3':
                limitation_errors.append(EpiConfigLimitationError('population age only available when nuts is nuts3'))
            
        if self.feature_popsize:
            if not self.feature_popage:
                limitation_errors.append(EpiConfigLimitationError('Currently only popsize supported if popage is included as well'))
        
        if len(limitation_errors):
            raise IssueReport(limitation_errors, 'EpiConfig couldnt be created')

    def _validate_warnings(self) -> None:
        """
        Validates some combinations of inputs that are likely not meant as such, and shouldn't disrupt the pipeline any further. 
        A EpiConfigWarning is thrown
        """
        if self.prediction_mode != 'regression' and self.incidence_scalar != 10_000:
            w = EpiConfigWarning('incidence_scalar will not be taken into account when using prediction_mode != "regression"')
            print(w)

        if self.quantiles is not None and self.prediction_mode != 'regression':
            w = EpiConfigWarning('quantiles will not be taken into account when using prediction_mode != "regression"')                
            print(w)
            
    # ============= ATTRIBUTE ORGANIZATION ==============
    def _set_hidden_attributes(self) -> None:
        """post validation, sets hidden attributes that should be accessed (and are not available in representation)"""
        self._num_quantiles = len(self.quantiles) if self.quantiles else 0

    def _classify_attributes(self) -> None:
        """creates dictionaries of attributes and classifies those. Used for back-end and for interaction with repr/str dunders"""

        self.attributes_dict = vars(self)

        self.attributes_classified_dict = {
            'main'          :   ['disease'],
            'temporal'      :   ['temporal_frequency','min_date','max_date','split_trainval','split_valtest'],
            'geography'     :   ['nuts_level','split_berlin'],
            'task config'   :   ['horizon_size','horizon_leadtime','quantiles','prediction_mode','predict_difference'],
            'features'      :   ['time_index_d','time_index_w','time_index_m','lag_column','lag_num','sequence_length','incidence_scalar', 'feature_popsize','feature_popdens','feature_gisd','feature_popage'],
            'normalization' :   ['normalization_method','log_transform','log_shift'],    
            'column names'  :   ['temporal_column','target_column','id_column','pred_column'],
            'others'        :   ['verbose'],
            'none'          :   ['attributes_dict', 'attributes_classified_dict']
        }

        for attribute in self.attributes_dict:
            # hidden attributes are not of interest
            if attribute.startswith("_"):
                continue

            classified = any(attribute in value_list for value_list in self.attributes_classified_dict.values())
            if not classified:
                raise EpiConfigValidationError(f'Attribute {attribute} not classified.\nLikely stems from an update in EpiConfig class, without incorporating it into the classification dict in _classify_attributes()')

    # ============= PROPERTIES =============
    @property
    def split_columns(self) -> list[str]:
        """Names of split indicator columns."""
        return ['train', 'val', 'test']
    
    @property
    def data_path(self) -> Path:
        """Path object for data directory. This is the path to the data folder in my personal Wissdaten"""
        return Path(get_data_env())
    
    # ============== PATH-RETURNING ===================
    def get_disease_path(self) -> Path:
        """Path to disease CSV file."""
        return self.data_path / 'processed/germany/epidemiology/casedata/survstat' / f'{self.disease}.csv'
    
    def get_population_path(self) -> Path:
        """Path to population size of nuts3 CSV file."""
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
        """Path to shapefile of specified nuts-level. Is to be tokenized."""
        return self.data_path / f'processed/germany/geospatial/shapefiles/shape_{self.nuts_level}.shp'
    
    def get_nuts_harmonization_path(self) -> Path:
        """Path to NUTS names file."""
        return self.data_path / 'processed/germany/geospatial/harmonization/nuts.tsv'
    
    def get_shapefile_paths(self) -> Dict[str,Path]:
        """Path to all shapefiles of Germany -> to not be tokenized."""

        return_dict = {
            'nuts0':    self.data_path / f'processed/germany/geospatial/shapefiles/shape_nuts0.shp'  ,
            'nuts1':    self.data_path / f'processed/germany/geospatial/shapefiles/shape_nuts1.shp'  ,
            'nuts2':    self.data_path / f'processed/germany/geospatial/shapefiles/shape_nuts2.shp'  ,
            'nuts3':    self.data_path / f'processed/germany/geospatial/shapefiles/shape_nuts3.shp'  ,
        }

        return return_dict  
    
    # ============= SUMMARIES =============
    
    def minimal_summary(self) -> str: 
        """small - scale summary: selection of attributes displayed"""
        summary =(
            f"<{self.__class__.__name__}(disease={self.disease}, "
                f"nuts_level={self.nuts_level}, "
                f"min_date={self.min_date}, "
                f"max_date={self.max_date}, "                
                f"horizon_size={self.horizon_size}, "
                f"sequence_length={self.sequence_length})"  
        )      
        return summary

    def summary(self) -> str: 
        """extensive - summary: all attributes are displayed"""
        all_keys        = list(self.attributes_dict.keys())
        width           = max(len(k) for k in all_keys)
        indent          = 4
        
        lines = [f"<{self.__class__.__name__}("]     

        for attr_class, attr_list in self.attributes_classified_dict.items():
            if attr_class != 'none':
                lines.append(return_header_line(attr_class, n_indent_chars=12, indent = indent))
                for attr_name, attr_value in self.attributes_dict.items():
                    if attr_name in attr_list:
                        lines.append(align(attr_name, attr_value, width, indent = indent))
                lines.append("")

        lines.append(")>")
        summary = '\n'.join(lines)
        return summary
    
    def __repr__(self) -> str:
        return self.minimal_summary()
    
    def __str__(self) -> str:
        return self.summary()