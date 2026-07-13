from src.dataloading.dataloaders import (
    BaseLineDataLoaderManager, DeepDataLoaderManager, GraphDataLoaderManager
)
from typing import Dict, Union, List
from dataclasses import dataclass, asdict, fields
from typing import Optional
from pathlib import Path
import yaml

from ..utils.helpers import PathNotFound
from ..utils.textformatting import align

import logging
logger = logging.getLogger(__name__)

class InvalidExtension(Exception):
    def __init__(self, extension_expected: str, extension_got: str):
        msg = f"Expected' {extension_expected}', got '{extension_got}'."
        super().__init__(msg)

class IncompatibleAttributeFound(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)

class AttributeNotFound(Exception):
    def __init__(self, missing_attr: str, class_name: str):
        msg = f"no attribute {missing_attr} in {class_name}"
        super().__init__(msg)


@dataclass 
class ExperimentConfig:
    """
    Config dataclass that contains information 
    to run and load an experiment

    Parameters
    ----------
    experiment_name: str
        name of the experiment. This will be the folder in 'project_utilities' in which
        this config is saved.
    filename_seperator: str    
        character used in model-names between elements of the name.
    variable: str
        variable that is being adjusted in the experiment. Should
        be a variable of EpiConfig.
    variable_alias: str
        alias for the variable, saved in model-names. For example,
        'hl' for 'horizon_leadtime'.
    variable_values: List[Any]
        list of the values that the variable takes.
    graphs: Optional[List[str]]
        list of graph-structures to be included.
    models: List[str]
        list of model-names to be included, directly related to the 
        modelclass-name without the 'model' in the name. For example,
        'lstm' for an instance of LSTMModel.
    seeds: List[int]    
        list of seeds to be run.

    Methods
    -------
    - `load()`
    - `save_config()`
    - `equals()`
    - `merge()`
    """
    experiment_name:    str
    filename_seperator: str    
    variable:           str
    variable_alias:     str
    variable_values:    List[Union[int, str, float]]
    graphs:             Optional[List[str]]
    models:             List[str]
    seeds:              List[int]
        
    FUNDAMENTAL_ATTRIBUTES = ['experiment_name', 'variable', 'variable_alias', 'filename_seperator']

    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """load an ExperimentConfig"""
        if not path.exists:
            raise PathNotFound(path)

        with open(path) as f:
            d = yaml.safe_load(f)      

        logger.info("ExperimentConfig loaded from %s", path)
        return cls(**d)   

    def save_config(self, path: Path) -> None: 
        """save ExperimentConfig to dedicated path (must include .yaml)"""

        if not path.exists():
            path.mkdir()
            logger.info("path %s made", path)            

        if path.suffix != '.yaml':
            raise InvalidExtension('.yaml',path.suffix)

        config_dict = asdict(self)

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)    
        
        logger.info("ExperimentConfig saved to %s", path)        

    def equals(self, other_cfg: 'ExperimentConfig', exclusion_attr: Union[str, List[str]] = []) -> bool:
        """
        Compare two ExperimentConfigs; returns whether or not these are identical

        Parameters
        ----------
        other_cfg: 'ExperimentConfig
            the ExperimentConfig instance which to compare self to
        exclusion_attr: Union[str, List[str]] = []
            the attributes not to take into account. May be any of:
            'variable_values', 'graphs', 'models', 'seeds'.
            Additionally, `exclusion_attr` could be 'ALL'. That is to say,
            except for the fundamental attributes, none are taken into account.
            NOTE these may not include any of self.FUNDAMENTAL_ATTRIBUTES.
                if any of those are not identical, the instances of 
                ExperimentConfig are fundametnally different.
        """
        if exclusion_attr != 'ALL':
            # check that exclusion attribute is actually an attribute
            for attr in exclusion_attr:
                if attr not in dir(self):
                    raise AttributeNotFound(attr, self.__class__.__name__)

        for attr in self.FUNDAMENTAL_ATTRIBUTES:
            # check that exclusion attribute are not fundamental attributes         
            if attr in exclusion_attr:
                raise IncompatibleAttributeFound(f'Invalid experimentconfig comparison. {attr} must be equal, cannot be put in exclusion_attr')
                   
        for attr in fields(self):
            # comparing attributes; attributes may be fundamental or not. 
            # Fundamental attributes: must be identical within the same experiment
            if attr.name in self.FUNDAMENTAL_ATTRIBUTES:
                if getattr(self, attr.name) != getattr(other_cfg, attr.name):
                    raise IncompatibleAttributeFound(f'Attribute {attr.name} is unequal, while it must be!')
                
            # non-fundamental attributes: identical-ness may be enforced
            # if there's a difference; return False.
            if attr.name not in exclusion_attr and exclusion_attr != 'ALL':
                if getattr(self, attr.name) != getattr(other_cfg, attr.name):
                    return False 
                
        return True

    def merge(self, other_cfg: 'ExperimentConfig') -> 'ExperimentConfig':
        """ 
        Merge two ExperimentConfigs. The `self.equal(other_cfg)` is asserted to be True, with no exclusion-attributes.
        """
        if not self.equals(other_cfg, 'ALL'):
            raise IncompatibleAttributeFound('These two configs cannot be merged based on these appending attributes!')
            
        logger.debug("ExperimentConfigs are allowed to be merged")

        # if identical, keep those from self, else append the list
        variable_values = self.variable_values  if self.variable_values == other_cfg.variable_values    else list(set(self.variable_values + other_cfg.variable_values))
        models          = self.models           if self.models == other_cfg.models                      else list(set(self.models +  other_cfg.models))
        seeds           = self.seeds            if self.seeds == other_cfg.seeds                        else list(set(self.seeds +  other_cfg.seeds))

        # with graphs it's slightly more complicated; may also be None
        if self.graphs == other_cfg.graphs:
            graphs              = self.graphs
        else:
            if self.graphs is None:
                graphs = other_cfg.graphs
            elif other_cfg.graphs is None:
                graphs = self.graphs 
            else:
                graphs = list(set(self.graphs + other_cfg.graphs))

        merged_expcfg = ExperimentConfig(
            # fundamental ones must be identical anyway
            experiment_name     = self.experiment_name,
            filename_seperator  = self.filename_seperator,
            variable            = self.variable,
            variable_alias      = self.variable_alias,

            # the added combinations of parameters
            variable_values     = variable_values,
            graphs              = graphs,
            models              = models,
            seeds               = seeds
            )
        logger.info("ExperimentConfigs merged")
        return merged_expcfg

    def __str__(self) -> str:
        """nicely alined representation"""
        all_keys    = ['experiment_name', 'variable', 'variable_alias', 'variable_values', 'models', 'graphs', 'seeds']
        width       = max(len(k) for k in all_keys) if all_keys else 20
        
        lines = [f'<{self.__class__.__name__}(']

        for key in all_keys:
            lines.append(align(key, getattr(self, key), width))
        lines.append(')>')
        
        return '\n'.join(lines)

    def __repr__(self) -> str:
        """one-liner representation"""
        all_keys    = ['experiment_name', 'variable', 'variable_alias', 'variable_values', 'models', 'graphs', 'seeds']

        line_0 = f'<{self.__class__.__name__}('
        line_1 = ')>'

        lines = []
        for key in all_keys:
            lines.append(f'{key} = {getattr(self, key)}')
        lines = ', '.join(lines)

        lines = line_0 + lines + line_1
        
        return lines

@dataclass(frozen= True)
class ExperimentDLMs:
    """ 
    For a single run (i.e. single value of a variable), this dataclass
    stores the dataloadermanagers
    """
    baseline:   BaseLineDataLoaderManager
    deep:       DeepDataLoaderManager
    graphs:     Dict[str, GraphDataLoaderManager]

@dataclass(frozen= True)
class ModelSpecs:
    """ 
    Dataclass containing information for a given model.
    """
    name:           str
    variable_alias: str 
    variable_value: Union[int, float, str]
    model:          str 
    graph:          Optional[str]
    seed:           int 

    def __str__(self) -> str:
        all_keys    = ['name', 'variable_alias', 'variable_value', 'model', 'graph', 'seed']
        width       = max(len(k) for k in all_keys) if all_keys else 20
        
        lines = [f'<{self.__class__.__name__}(']

        for key in all_keys:
            lines.append(align(key, getattr(self, key), width))

        lines.append(')>')
        
        return '\n'.join(lines)

    def __repr__(self) -> str:
        all_keys    = ['name', 'variable_alias', 'variable_value', 'model', 'graph', 'seed']

        line_0 = f'<{self.__class__.__name__}('
        line_1 = ')>'

        lines = []
        for key in all_keys:
            lines.append(f'{key} = {getattr(self, key)}')

        lines = ', '.join(lines)

        lines = line_0 + lines + line_1
        
        return lines