from pathlib import Path 
from typing import Optional, assert_never, Dict, Union 
import os

from .containers import ExperimentConfig, ExperimentDLMs
from ..dataloading.epiconfig import EpiConfig
from ..dataloading.epidataorchestration import EpiDataOrchestrator
from ..dataloading.dataloaders import BaseLineDataLoaderManager, DeepDataLoaderManager, GraphDataLoaderManager
from ..utils.helpers import get_project_utilities_env
from ..models.deep.deepmodel import DeepModel

class ExperimentHandler:
    """ 
    Parent class to those that deal with experiments, namely:
    - ExperimentRunner
    - ExperimentLoader
    
    TODO These have slightly different downstream-behaviour that I need to clear up
    a bit still.
    
    Parameters
    ----------
    epiconfig: 'EpiConfig'
        the basis-epiconfig of the experiment
    experiment_cfg: 'ExperimentConfig'
        the basis-experimentconfig of this experiment   
    """
    
    # class attributes: these are never different!
    experiment_cfg_filename = '_experiment_config.yaml'
    epicfg_filename         = '_epiconfig.yaml'

    def __init__(self, epiconfig: 'EpiConfig', experiment_cfg: 'ExperimentConfig'):
        self.experiment         = experiment_cfg.experiment_name    # set experiment name
        self.epiconfig          = epiconfig
        self.experiment_cfg     = experiment_cfg

        self.path_exist         = self._set_exp_dir()               # set whether experiment already exists
        self._set_dlms()                                            # set dataloadermanagers per variable-value

    def _set_exp_dir(self) -> bool:
        """
        Takes care of testing the integrity of experiment-directory.
        Returns a boolean whether or not directory already exists.
        if it does, also checks that the experiment_config and the epiconfig are present.
        Further sets the attribute `experiment_dir`.
        """
        self.experiment_dir = Path(get_project_utilities_env()) / "models" / self.experiment

        if not os.path.exists(self.experiment_dir):
            return False

        files_to_check = [self.experiment_cfg_filename, self.epicfg_filename]
        for ff in files_to_check:
            if ff not in os.listdir(self.experiment_dir):
                raise ValueError(f'directory {self.experiment} already exists, but {ff} was not found.')        
        return True
        
    def _set_dlms(self) -> None:
        """Shared dataloader construction — called by Runner and Loader. Sets dlms into `dlms`."""
        dataloader_managers_dict: Dict[Union[int, str, float], ExperimentDLMs] = {}

        variable  = self.experiment_cfg.variable
        varvalues = self.experiment_cfg.variable_values

        for varvalue in varvalues:
            cfg = self.epiconfig.copy(**{variable: varvalue})
            epo = EpiDataOrchestrator(cfg).build()
            
            # graphs needs to be iterable: graph_list
            graphs_list = [] if self.experiment_cfg.graphs is None else self.experiment_cfg.graphs
    
            hl_dlms = ExperimentDLMs(
                baseline = BaseLineDataLoaderManager(epo),
                deep     = DeepDataLoaderManager(epo).build(),
                graphs   = {
                    graph: GraphDataLoaderManager(epo)
                               .retrieve_static_graph(graph)
                               .build()
                    for graph in graphs_list
                }
            )

            dataloader_managers_dict[varvalue] = hl_dlms

        self.dlms = dataloader_managers_dict   

    def _get_dlm(self, modelclass: str, varvalue: Union[int, float, str], graphtype: Optional[str] = None):
        """Shared DLM lookup — identical in Runner and Loader."""
        childclass  = DeepModel._childclasses[f"{modelclass}model"]
        expected_dlm= childclass._expected_dataloadermanager

        match expected_dlm:

            case "DeepDataLoaderManager":
                return self.dlms[varvalue].deep
        
            case "GraphDataLoaderManager":
                if graphtype is None:
                    raise ValueError(f"Graph required for {modelclass}")
                return self.dlms[varvalue].graphs[graphtype]
        
            case "BaseLineDataLoaderManager":
                raise ValueError("Baseline models should not be loaded via experiment runner")
        
        assert_never(expected_dlm)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.experiment})>"