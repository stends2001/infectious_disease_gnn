from pathlib import Path 
from typing import Optional, assert_never, Dict, Union 
import os

from .issues import DataLoaderManagerError, ExperimentDirectoryInvalidError
from .containers import ExperimentDLMs, ExperimentConfig
from ..dataloading.epiconfig import EpiConfig
from ..utils.helpers import get_project_utilities_env
from ..models.deep.deepmodel import DeepModel

class ExperimentHandler:
    """ 
    Parent class of those that deal with experiments, namely:
    - ExperimentRunner
    - ExperimentLoader
    - ExperimentAnalyzer(second degree; this is a sub-class to ExperimentLoader)
    
    This parent class only deals with the paths based on the experiment_name.
    Also defines a `_get_dlm()` method which is shared among the subclasses.

    Parameters
    ----------
    experiment_name: str 
        name of the experiment. This string should be identical to the directory in which the models
        and configs are saved.
    verbose: int
        the level to which output/updates are to be returned.        
    
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
    
    # class attributes: these are never different!
    dataloadermanagers:         Optional[Dict[Union[int, str, float], ExperimentDLMs]] = None
    epicfg:                     EpiConfig
    expcfg:                     ExperimentConfig

    experiment_cfg_filename:    str = '_experiment_config.yaml'
    epicfg_filename:            str = '_epiconfig.yaml'

    def __init__(self, 
                 experiment_name: str,
                 verbose: int = 1):
        
        self.experiment         = experiment_name
        self.verbose            = verbose
        self.experiment_dir     = self._get_exp_dir()
        self.path_exist         = self._validate_exp_dir()          # set whether experiment already exists

    def _get_exp_dir(self) -> Path:
        return Path(get_project_utilities_env()) / "models" / self.experiment

    def _validate_exp_dir(self) -> bool:
        """
        Takes care of testing the integrity of experiment-directory.
        Returns a boolean whether or not directory already exists.
        if it does, also checks that the experiment_config and the epiconfig are present.
        """
        if not os.path.exists(self.experiment_dir):
            return False

        files_to_check = [self.experiment_cfg_filename, self.epicfg_filename]
        for ff in files_to_check:
            if ff not in os.listdir(self.experiment_dir):
                raise ExperimentDirectoryInvalidError(f'directory {self.experiment} already exists, but {ff} was not found.')        
        return True
        
    def _get_dlm(self, modelclass: str, varvalue: Union[int, float, str], graphtype: Optional[str] = None):
        """Shared DLM lookup — identical in Runner and Loader."""

        if self.dataloadermanagers is None:
            raise DataLoaderManagerError('dataloadermanagers attribute has not been set')

        childclass  = DeepModel._childclasses[f"{modelclass}model"]
        expected_dlm= childclass._expected_dataloadermanager

        match expected_dlm:

            case "DeepDataLoaderManager":
                return self.dataloadermanagers[varvalue].deep
        
            case "GraphDataLoaderManager":
                if graphtype is None:
                    raise DataLoaderManagerError(f"Graph required for {modelclass} but got none")
                
                return self.dataloadermanagers[varvalue].graphs[graphtype]
        
            case "BaseLineDataLoaderManager":
                raise ValueError("Baseline models should not be loaded via experiment runner")
        
        assert_never(expected_dlm)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.experiment})>"