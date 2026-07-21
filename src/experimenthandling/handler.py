from typing import Optional, assert_never, Dict, Union 
import os

from .exceptions import DataLoaderManagerError, ExperimentDirectoryInvalidError
from .containers import ExperimentDLMs, ExperimentConfig
from ..dataloading.epiconfig import EpiConfig
from ..utils import PathManager, PathNotFound
from ..models.deep.deepmodel import DeepModel

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

class ExperimentHandler:
    """ 
    Parent class of those that deal with experiments, namely:
    - ExperimentRunner
    - ExperimentLoader
        - ExperimentAnalyzer(second degree; this is a sub-class to ExperimentLoader)
    
    This parent class only deals with the paths based on the experiment_name.
    This class should not be called in itself, but rather, its subclasses should.
    Also defines a `_get_dlm()` method which is shared among the subclasses.

    Parameters
    ----------
    experiment_name: str 
        name of the experiment. This string should be identical to the directory in which the models
        and configs are saved.     
    
    Methods
    -------
    - `_get_dlm()`
        returns a specific dataloadermanager from `.dataloadermanagers`.
    
    Attributes
    ----------
    dataloadermanagers: Optional[Dict[Union[int, str, float], ExperimentDLMs]] = None
        Dictionary of ExperimentDLMs per value of the variable in the experiment. `_get_dlm()` returns
        a specific dataloadermanager from this attribute.
    epicfg: EpiConfig
        To be set by sub-class, done differently for ExperimentLoader and for ExperimentRunner!
    expcfg: ExperimentConfig

    See Also
    --------
    The following dataclasses are also intimitately used:
    - ExperimentConfig
    - ExperimentDLMs
    - ModelSpecs
    """    
    dataloadermanagers:         Optional[Dict[Union[int, str, float], ExperimentDLMs]] = None               # TODO
    epicfg:                     EpiConfig           
    expcfg:                     ExperimentConfig

    expcfg_filename:            str = '_experiment_config.yaml'
    epicfg_filename:            str = '_epiconfig.yaml'

    def __init__(self, 
                 experiment_name: str):
        
        self.experiment          = experiment_name
        
        # paths
        self.pm                  = PathManager()
        self.path_exp_root       = self.pm.exp_out
        self.path_exp            = self.pm.exp_out / experiment_name
        self.exp_exists          = self._exp_directory_exists()
  
    def _exp_directory_exists(self) -> bool:
        """
        Takes care of testing the integrity of experiment-directory.
        Returns a boolean whether or not directory already exists.
        if it does, also checks that the experiment_config and the epiconfig are present.
        """
        # validate that the root of experiments - paths exists
        if not os.path.exists(self.path_exp_root):
            raise PathNotFound(self.path_exp_root)

        # test whether the path of the experiment already exists
        if not os.path.exists(self.path_exp):
            logger.debug("Experiment path %s doesn't exist.", self.path_exp)
            return False

        # if the experiment-path already exists, an expcfg and an epicfg NEED TO BE present
        files_to_check = [self.expcfg_filename, self.epicfg_filename]
        for ff in files_to_check:
            if ff not in os.listdir(self.path_exp):
                raise ExperimentDirectoryInvalidError(f'directory {self.experiment} already exists, but {ff} was not found.')        
            
        logger.debug("Valid experiment path %s found.", self.path_exp)        
        return True  

    def _get_dlm(self, modelclass: str, varvalue: Union[int, float, str], graphtype: Optional[str] = None):
        """Shared DLM lookup — identical in Runner and Loader."""

        if self.dataloadermanagers is None:
            raise DataLoaderManagerError('dataloadermanagers attribute has not been set')

        childclass  = DeepModel._childclasses[f"{modelclass}model"]
        expected_dlm= childclass._expected_dataloadermanager

        match expected_dlm:
        
            case "GraphDataBuilder":
                if graphtype is None:
                    raise DataLoaderManagerError(f"Graph required for {modelclass} but got none")
                
                return self.dataloadermanagers[varvalue].graphs[graphtype]
        
            case "BaseLineDataLoaderManager":
                raise DataLoaderManagerError("Baseline models should not be loaded via experiment runner")
        
        assert_never(expected_dlm)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.experiment})>"        