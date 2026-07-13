from pathlib import Path 
from typing import Optional, Union, Dict, Type
import os
from itertools import product
from tqdm import tqdm

from .handler import ExperimentHandler
from .containers import ExperimentConfig
from ..dataloading.epiconfig import EpiConfig
from ..dataloading.epidataorchestration import EpiDataOrchestrator
from ..models.deep.deepmodel import DeepModel
from .containers import ExperimentConfig, ExperimentDLMs
from ..dataloading.dataloaders import BaseLineDataLoaderManager, DeepDataLoaderManager, GraphDataLoaderManager

import logging
logger = logging.getLogger(__name__)

class ExperimentRunner(ExperimentHandler):
    """ 
    Runs experiments
    First degree subclass of ExperimentHandler

    Parameters
    -------
    epiconfig: EpiConfig
        The base-epiconfig for the experiment. With the exception of the variable that is experimented with,
        by a range of values, the epiconfig must describe the entire experiment!
    experimentconfig: ExperimentConfig
        The config that covers the experiment.     
    
    Examples
    --------
    >>> runner = ExperimentRunner(base_cfg, experiment_cfg)
    >>> runner.run(global_hparams) 

    Methods
    -------
    - `run()`
        The main/only function to be ran by users. The hidden functions are all to run internally.
            
    See Also
    --------
    For more information on basic experiment-handling-behaviour see parent class ExperimentHandler.
    """
    def __init__(self,
                 epiconfig:        EpiConfig,
                 experimentconfig: ExperimentConfig,):

        super().__init__(experimentconfig.experiment_name)

        self.epicfg         = epiconfig
        self.expcfg         = experimentconfig       

        # store the new values before merging; preventing new experiment addition from running everything.
        # NOTE: the exp cfg saved contains that of previous sub-experiments, with the new variable added (`expcfg_to_save`).
        # What's executed in this instance, however, is the specific expcfg inputed, saved as `expcfg`.
        self.new_variable_values    = experimentconfig.variable_values
        self.expcfg_to_save         = self._update_experiment_cfg(epiconfig, experimentconfig)    

        self._set_dlms()
           
    # ======= METHODS ======= #
    def run(self, global_hparams: dict, show_progress: bool = False):
        """
        Main function; runs the experiment
        Feed in global_hparams for deepmodel instances
        """
        self._save_cfgs()
        
        graphlist = [None] if self.expcfg.graphs is None else self.expcfg.graphs

        logger.info("Models training ...")

        iterator = product(self.new_variable_values, self.expcfg.models)

        if show_progress:
            iterator =  tqdm(
                iterator,
                total=len(self.new_variable_values) * len(self.expcfg.models),
                desc="Training models",
            )

        for value, ml in iterator:
                
                modeltype  = f"{ml}model"
                childclass = DeepModel._childclasses[modeltype]

                for sd in self.expcfg.seeds:

                    match childclass._expected_dataloadermanager:

                        case 'GraphDataLoaderManager':
                            for graph in graphlist:
                                modelname = self._get_model_name(ml, value, sd, graph)
                                self._train_and_save_model(modelname, childclass, value, ml, global_hparams, graph)

                        case 'DeepDataLoaderManager':
                            modelname = self._get_model_name(ml, value, sd, None)
                            self._train_and_save_model(modelname, childclass, value, ml, global_hparams, None)

        logger.info("Models done training")
     
    # ======= HIDDEN METHODS ======== #
    def _train_and_save_model(self, 
                              modelname: str, 
                              childclass: Type['DeepModel'], 
                              value: Union[int, str, float], 
                              ml: str, 
                              global_hparams: dict, 
                              graph: Optional[str]):
        """
        loads, trains and saves single model
        """
        if not self._model_to_run(modelname):
            return

        model = self._load_model(modelname, childclass, value, ml, graph)

        model.set_model_hparams()
        model.set_global_hparams(**global_hparams)
        model.train()
        model.save_model(dir=Path(self.experiment))

        logger.info(
            "Model %s trained and saved in %s",
            modelname,
            Path(self.experiment),
        )

    def _update_experiment_cfg(self, new_epiconfig: EpiConfig , new_experimentconfig: ExperimentConfig) -> ExperimentConfig:
        """
        updates ExperimentConfig instance if necessary. Validates compatibility, 
        also for epiconfig, with pre-existing experiment-directory.
        """
        # if path does exist, save epiconfig and experimentconfig into pre-existing values (prex) and validate compatibility
        if self.exp_exists:
            current_expcfg      = ExperimentConfig.load(self.path_exp / self.expcfg_filename)
            current_epicfg      = EpiConfig.load_config(self.path_exp / self.epicfg_filename)
            current_epicfg.assert_equals(new_epiconfig)
            return current_expcfg.merge(new_experimentconfig)
        
        # else, experimentconfig doesn't change
        else:
            return new_experimentconfig
        
    def _set_dlms(self) -> None:
        """Only build dataloaders for the new variable values being run."""

        logger.info('Setting dataloadermanagers ...')

        dataloader_managers_dict: Dict[Union[int, str, float], ExperimentDLMs] = {}
        variable = self.expcfg.variable

        for varvalue in self.new_variable_values:
            epicfg                  = self.epicfg.copy(**{variable: varvalue})
            epidata_orchestrator    = EpiDataOrchestrator(epicfg).build()

            graphs_list = [] if self.expcfg.graphs is None else self.expcfg.graphs

            hl_dlms = ExperimentDLMs(
                baseline = BaseLineDataLoaderManager(epidata_orchestrator),
                deep     = DeepDataLoaderManager(epidata_orchestrator).build(),
                graphs   = {
                    graph: GraphDataLoaderManager(epidata_orchestrator)
                            .retrieve_static_graph(graph)
                            .build()
                    for graph in graphs_list
                }
            )
            dataloader_managers_dict[varvalue] = hl_dlms

        self.dataloadermanagers = dataloader_managers_dict
        
        logger.info('Dataloadermanagers set')
    
    def _load_model(self, modelname: str, childclass: Type['DeepModel'], value: Union[int, str, float], ml: str, graph: Optional[str]) -> 'DeepModel':
        """load instance of model"""
        dlm = self._get_dlm(ml, value, graph)                       # graph=None for DeepDataLoaderManager
        return childclass(name=modelname, dataloadermanager=dlm)    # type: ignore

    def _get_model_name(self, modeltype: str, varvalue: Union[str, float, int], seed: int, graph: Optional[str] = None) -> str:
        """return model name: to be saved as"""
        sep = self.expcfg.filename_seperator
        var = self.expcfg.variable_alias

        if graph is None:
            name = f"{modeltype}{sep}{var}{varvalue}{sep}s{seed}"

        else:
            name = f"{modeltype}{sep}{graph}{sep}{var}{varvalue}{sep}s{seed}"            
        
        return name
        
    def _save_cfgs(self) -> None:
        """save experiment config and epi config. If exp path doesnt yet exist, will be created here."""
        if not self.exp_exists:
            os.mkdir(self.path_exp)    
            logger.info("Experiment path made at %s", self.path_exp)
            self.exp_exists          = True


        self.epicfg.save_config(self.path_exp / self.epicfg_filename) 
        logger.info("EpiConfig saved into %s", self.path_exp / self.epicfg_filename)

        self.expcfg_to_save.save_config(self.path_exp / self.expcfg_filename)        
        logger.info("ExperimentConfig saved into %s", self.path_exp / self.epicfg_filename)
        
    def _model_to_run(self, modelname: str) -> bool:
        """boolean on whether the modelname already exists or not."""
        saved_models = {
            os.path.splitext(f)[0] 
            for f in os.listdir(self.path_exp) 
            if f.endswith('.pt')
        }
        if modelname in saved_models:
            print(f'Weights for model {modelname} already exist')
            return False
        return True