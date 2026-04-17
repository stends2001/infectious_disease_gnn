from dataclasses import dataclass
from typing import Literal, Optional, List, assert_never, Dict, Union
from pathlib import Path
import yaml
import dataclasses

from .pathmanager import EpiPathsManagerGermany, EpiPathsManagerNetherlands, EpiPathsManagerHungary
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
    country:                Literal['germany','netherlands','hungary']  = 'germany'
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

            case 'hungary':
                self.path_manager = EpiPathsManagerHungary(self.disease, self.level)                
            
            case _:
                assert_never(self.country)

        self.validator = EpiConfigValidator(self)
        self.validator.validate()

        self._set_hidden_attributes()
        self._classify_attributes()

    # ============ Methods =========== #
    def assert_equals(self, other: Union['EpiConfig', Dict[str,str]], level: Literal[0,1,2,3,4] = 1) -> None:
        """for DeepModel - loading use level = 1. For Evaluator use level = 2!"""        
        self_summary  = self.get_summary(level)
        if isinstance(other, dict):
            other_summary = other
        else:
            other_summary = other.get_summary(level)
        diff = {k: (self_summary[k], other_summary.get(k))
                for k in self_summary if self_summary[k] != other_summary.get(k)}
        if diff:
            raise ValueError(f"EpiConfig mismatch at level {level}:\n" + 
                            "\n".join(f"  {k}: {v[0]} vs {v[1]}" for k, v in diff.items()))

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
            'column_names'  :   ['temporal_column','target_column','id_column','pred_column'],
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
    
    # ============= DICT - SUMMARIES =========== #
    def get_summary(self, level: Literal[0, 1, 2, 3, 4]) -> Dict[str, str]:
        """
        Level 0: core identity only
        Level 1: + temporal, excl. max_date  (use for model loading — test period may differ)
        Level 2: + temporal incl. max_date   (use for evaluation — test period must match)
        Level 3: + features, normalization
        Level 4: all attributes        
        """
        CLASSES_BY_LEVEL = {
            0: {'main', 'geography', 'task', 'column_names'},
            1: {'main', 'geography', 'task', 'column_names', 'temporal'},
            2: {'main', 'geography', 'task', 'column_names', 'temporal'},            
            3: {'main', 'geography', 'task', 'column_names', 'temporal', 'features', 'normalization'},
            4: None,  # None = all classes
        }
        EXCLUDE_BY_CLASS = {
            'temporal': {'max_date'},  # testing period may differ
        }

        allowed_classes = CLASSES_BY_LEVEL[level]
        summary: Dict[str, str] = {}

        for attr_class, attr_list in self.attributes_classified_dict.items():
            if allowed_classes is not None and attr_class not in allowed_classes:
                continue

            # only exclude at level 1; at level 2+ max_date is included
            exclude = EXCLUDE_BY_CLASS.get(attr_class, set())
            
            if level > 1:
                exclude = set()

            for attr_name, attr_value in self.attributes_dict.items():
                if attr_name in attr_list and attr_name not in exclude:
                    summary[attr_name] = str(attr_value)

        return summary

    def __repr__(self) -> str:
        repr =(
            f"<{self.__class__.__name__}(disease={self.disease}, "
                f"country={self.country}, "                
                f"level={self.level}, "
                f"min_date={self.min_date}, "
                f"max_date={self.max_date}, "                
                f"horizon_size={self.horizon_size}, "
                f"sequence_length={self.sequence_length})"  
        )          
        return repr
    
    def __str__(self) -> str:
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
        repr = '\n'.join(lines)        
        return repr            