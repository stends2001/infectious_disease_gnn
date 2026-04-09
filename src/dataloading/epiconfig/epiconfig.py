from dataclasses import dataclass
from typing import Literal, Optional, List, assert_never
from pathlib import Path
import yaml
import dataclasses

from .pathmanager import EpiPathsManagerGermany, EpiPathsManagerNetherlands
from .validator import EpiConfigValidator

from .issues import EpiConfigValidationError

from ...utils.textformatting import align, return_header_line

@dataclass
class EpiConfig:
    # ============= MAIN =============
    disease:                str   
    
    # ============= TEMPORAL =============
    temporal_frequency:     Literal['m','w','d']= 'w'
    min_date:               str = '2011-01-01'
    max_date:               str = '2020-06-01'
    split_trainval:         str = '2018-06-01'
    split_valtest:          str = '2019-06-01'
    
    # ============= GEOGRAPHY =============
    country:                Literal['germany','netherlands']            = 'germany'
    level:                  Literal['nuts1','nuts2','nuts3','ggd','lau']= 'nuts3'
    
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
    
    # ============= NORMALIZATION =============
    normalization_method:   Optional[Literal['minmax', 'zscore']] = 'zscore'
    log_transform:          Optional[List[str]] = None
    log_shift:              float = 1.0        
            
    # ============= COLUMN NAMES =============
    temporal_column:        str = 'timestamp'
    target_column:          str = 'incidence'
    id_column:              str = 'node'
    pred_column:            str = 'pred'
    
    # ============= OTHER =============
    verbose:                Literal[0,1,2] = 0

    # ============= DUNDER ============ #
    def __post_init__(self):
        
        # set pathmanager
        match self.country:
        
            case 'germany':
                self.path_manager = EpiPathsManagerGermany(self.disease, self.level)
        
            case 'netherlands':
                self.path_manager = EpiPathsManagerNetherlands(self.disease, self.level)
            
            case _:
                assert_never(self.country)

        self.validator = EpiConfigValidator(self)
        self.validator.validate()

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
        path        = self.path_manager.get('config') / f'{config_name}.yaml'
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
            'geography'     :   ['country','level'],
            'task'          :   ['horizon_size','horizon_leadtime','quantiles','prediction_mode','predict_difference'],
            'features'      :   ['time_index_d','time_index_w','time_index_m','lag_column','lag_num','sequence_length','incidence_scalar', 'feature_popsize','feature_popdens','feature_gisd','feature_popage','feature_climateology','feature_kreise_classes','feature_borders'],
            'normalization' :   ['normalization_method','log_transform','log_shift'],    
            'column names'  :   ['temporal_column','target_column','id_column','pred_column'],
            'others'        :   ['verbose'],
            'none'          :   ['attributes_dict', 'attributes_classified_dict'],
            'helper_classes':   ['path_manager','validator']
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
    
    # ============= SUMMARIES =============
    
    def minimal_summary(self) -> str: 
        """small - scale summary: selection of attributes displayed"""
        summary =(
            f"<{self.__class__.__name__}(disease={self.disease}, "
                f"country={self.country}, "                
                f"level={self.level}, "
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