from dataclasses import dataclass
from typing import Literal, Optional, Dict, List
from pathlib import Path
import pandas as pd
import yaml
import dataclasses

from ...utils.helpers import get_data_env
from ...utils.textformatting import align, return_header_line
from ...issues.errors import WissdatenMountingError
from ...issues import IssueReport
from .issues import EpiConfigWarning, EpiConfigValidationError, EpiConfigLimitationError, InvalidCovariatePath

@dataclass
class EpiConfig:
    """ 
    Dataclass that tells the EpiDataOrchestrator how to orchestrate that data,
    and prepare it uniformly for a set of possible dataloadermanagers.

    Upon initiation, internal validation is executed to ensure the data
    orchestrator will be able to work with this config.

    After initiation, and thus, validation, an instance can be saved using
    `save_config()`. EpiConfigs can also be loaded, using the classmethod
    `load_config()`.
    
    Note
    ----
    the dates listed in this class are solely strings.
    for proper dealing with timestamps, please refer to TemporalSummary   

    Examples
    --------
    ### loading config
    >>> config = EpiConfig.load_config('default_influenza_quantiles')

    ### default_influenza_point_predictions
    >>> config = EpiConfig(
        disease             = 'influenza',

        split_berlin        = False,

        horizon_leadtime    = 3,

        sequence_length     = 4,

        feature_popsize     = True,      
        feature_popdens     = True,
        feature_gisd        = True,
        feature_popage      = True,
        
        log_transform       = ['incidence']     
        )    

    ### default influenza quantile predictions
    >>> config = EpiConfig(
        disease             = 'influenza',

        split_berlin        = False,

        horizon_leadtime    = 3,
        quantiles           = [0.1,0.25,0.5,0.75,0.9],

        sequence_length     = 4

        feature_popsize     = True,      
        feature_popdens     = True,
        feature_gisd        = True,
        feature_popage      = True,
        
        log_transform       = ['incidence']     
        )     

    ### default covid point predictions
    >>> config = EpiConfig(
        disease             = 'covid_daily',

        temporal_frequency  = 'w',
        min_date            = '2020-03-02',
        max_date            = '2023-01-01',
        split_trainval      = '2022-06-01',
        split_valtest       = '2022-09-01', 

        split_berlin        = False,

        horizon_leadtime    = 7,

        time_index_d        = True,
        time_index_w        = False,
        sequence_length     = 7,

        feature_popsize     = True,      
        feature_popdens     = True,
        feature_popage      = True,
        
        log_transform       = ['incidence'],   
        )    

    ### saving config
    >>> config.save_config('default_covid_point_predictions')       
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
    split_berlin:           bool = False
    
    # ============= TASK =============
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
    feature_climateology:   bool = False   
    feature_kreise_classes: bool = False 
    feature_borders:        bool = False
    feature_vax:            bool = False
    
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

    # ============= DUNDER ============ #
    def __post_init__(self):

        # Validate input
        self._validate_input()
        self._validate_datapaths()
        self._validate_current_limitations()
        self._validate_warnings()

        self._set_hidden_attributes()

        self._classify_attributes()

    # ============ Methods =========== #
    def equals(self, other: 'EpiConfig', level: Literal[1,2]) -> bool:
        """
        returns whether or not two EpiConfig instances are equal

        Parameters
        ----------
        other: 'EpiConfig'
            the other epiconfig to be compares
        level: Literal[1,2]
            - 1: compares all attributes
            - 2: compares task attributes only (MAIN + Temporal + geographical + task)
        """
        if level == 1:
            return self == other
        else:
            attributes_groups   = ["main", "temporal", 'geography','task']
            attributes_to_check = [item for key in attributes_groups for item in self.attributes_classified_dict.get(key, [])]
            for attr in attributes_to_check:
                if getattr(self, attr) != getattr(other, attr):
                    return False 
            return True

    # ============ CONFIG LOADING/SAVING ==============
    def save_config(self, config_name: str):
        """ 
        saves EpiConfig to a .yaml of name `config_name` inside
        the directory returned by `get_config_path()`.
        """
        config_dict = dataclasses.asdict(self)
        path        = self.config_path / f'{config_name}.yaml'
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)    

        print(f'EpiConfig saved to {config_name}.yaml')

    @classmethod
    def load_config(cls, config_name: str) -> 'EpiConfig':
        """ 
        Loads a .yaml of name `config_name` into an EpiConfig
        the directory returned by `get_config_path()`.
        """        
        path = Path("config/epiconfigs") / f'{config_name}.yaml'
        with open(path) as f:
            d = yaml.safe_load(f)
        print(f'EpiConfig {config_name} loaded')            
        return cls(**d)      

    # ============= VALIDATION FUNCTIONS ==============
    def _validate_datapaths(self) -> None:

        datapath_errors = []

        if not Path(get_data_env()).exists():
            raise WissdatenMountingError(get_data_env())

        path_attributes = [attr for attr in dir(self) if attr.endswith("_path")]

        for path_attr in path_attributes:
            path = getattr(self, path_attr)
            if not path.exists():
                datapath_errors.append(InvalidCovariatePath(f"{path_attr} not found: {path}"))
            
        if len(datapath_errors):
            raise IssueReport(datapath_errors, 'EpiConfig couldnt be created')

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
            if not isinstance(self.quantiles, list):
               validation_errors.append(EpiConfigValidationError(f'Invalid input for quantiles ({self.quantiles}). Must be a List[float]'))
            
            middle_idx = int(len(self.quantiles) / 2)

            for quantile in self.quantiles:
                if quantile >= 1 or quantile <= 0:
                   validation_errors.append(EpiConfigValidationError(f'Invalid input for quantiles ({self.quantiles}). Must be a List of values 0 < quantile < 1'))

            if not len(self.quantiles) % 2:
                validation_errors.append(EpiConfigValidationError(f'Invalid input for quantiles ({self.quantiles}). Must be of odd length'))
            
            if self.quantiles[int(len(self.quantiles) / 2)] != 0.5:
                validation_errors.append(EpiConfigValidationError(f'Invalid input for quantiles ({self.quantiles}). Must be symmetric around quantile 0.5'))       

            if len(self.quantiles) > 1:

                for idx_l in range(0,middle_idx):
                    idx_r = len(self.quantiles) - idx_l - 1

                    if self.quantiles[idx_l] + self.quantiles[idx_r] != 1.0:                 
                        validation_errors.append(EpiConfigValidationError(f'Invalid input for quantiles ({self.quantiles}). Must be symmetric around quantile 0.5'))                                   

        if self.split_berlin:
            validation_errors.append(EpiConfigValidationError(f'Depcrecated: split_berlin must be False'))            

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
        # gisd
        if self.feature_gisd:
            if pd.to_datetime(self.max_date) > pd.to_datetime('2021-12-31'):
                limitation_errors.append(EpiConfigLimitationError('Currently GISD data only available until 2021 while simulation max date exceeds that. Either remove the gisd data as feature, or decrease the timespawn.'))
            if self.nuts_level == 'nuts1':
                limitation_errors.append(EpiConfigLimitationError('Currently GISD data only available for nuts levels 2 and 3. Please Adjust'))                
        
        if self.feature_climateology:
            limitation_errors.append(EpiConfigLimitationError('Currently no climateology features supported'))

        if self.feature_vax:
            limitation_errors.append(EpiConfigLimitationError('Currently no vax feature supported'))            

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
            'task'          :   ['horizon_size','horizon_leadtime','quantiles','prediction_mode','predict_difference'],
            'features'      :   ['time_index_d','time_index_w','time_index_m','lag_column','lag_num','sequence_length','incidence_scalar', 'feature_popsize','feature_popdens','feature_gisd','feature_popage','feature_climateology','feature_kreise_classes','feature_borders','feature_vax'],
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
    @property
    def config_path(self) -> Path:

        return Path("config/epiconfigs")

    @property
    def disease_path(self) -> Path:
        """Path to disease CSV file."""
        return self.data_path / 'processed/germany/epidemiology/casedata/survstat' / f'{self.disease}.csv'
    
    # features
    @property    
    def population_size_path(self) -> Path:
        """Path to population size of nuts3 CSV file."""
        return self.data_path / 'processed/germany/gnenv_covariates/population_size.csv'
    
    @property    
    def population_density_path(self) -> Path:
        """Path to population density CSV file."""
        return self.data_path / f'processed/germany/gnenv_covariates/population_density.csv'        

    @property
    def gisd_path(self) -> Path:
        """Path to gisd CSV file."""
        return self.data_path / f'processed/germany/gnenv_covariates/gisd.csv'    

    @property
    def population_age_path(self) -> Path:
        """Path to population age CSV file."""
        return self.data_path / f'processed/germany/gnenv_covariates/population_age.csv'        
    
    @property    
    def shapefile_path(self) -> Path:
        """Path to shapefile of specified nuts-level. Is to be tokenized."""
        return self.data_path / f'processed/germany/geospatial/shapefiles/nuts_shapes.shp'    
    
    @property    
    def nuts_harmonization_path(self) -> Path:
        """Path to NUTS names file."""
        return self.data_path / 'processed/germany/geospatial/harmonization/nuts_harmonization.tsv'
        
    @property        
    def kreise_classes_path(self) -> Path:
        """Path to kreise classification CSV file."""
        return self.data_path / f'processed/germany/gnenv_covariates/kreis_classes.csv' 
    
    @property    
    def borders_path(self) -> Path:
        """Path to borders CSV file."""
        return self.data_path / f'processed/germany/gnenv_covariates/borders.csv' 
    
    @property    
    def vacmap_path(self) -> Path:
        """Path to vacmap CSV file."""
        return self.data_path / f'processed/germany/gnenv_covariates/vacmap.csv'     

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