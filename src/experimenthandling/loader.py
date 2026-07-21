from typing import List, Union, Dict, Optional
import os
from tqdm import tqdm

from .exceptions import ExperimentDirectoryNotFoundError, InvalidModelNameError
from .handler import ExperimentHandler
from .containers import ExperimentConfig, ExperimentDLMs
from ..dataloading.epiconfig import EpiConfig
from ..models.deep.deepmodel import DeepModel
from ..models.base.basemodel import BaseModel
from ..dataloading.epidataorchestration import EpiDataOrchestrator
from ..dataloading.databuilders import BaseLineDataBuilder, GraphDataBuilder
from .containers import ExperimentConfig, ModelSpecs

import logging
logger = logging.getLogger(__name__)

class ExperimentLoader(ExperimentHandler):
    """ 
    Loads experiments
    First degree subclass of ExperimentHandler
    Parent class of ExperimentAnalyzer

    Parameters
    ----------
    experiment_name: str
        name of the experiment. This string should be identical to the directory in which the models and configs are saved.
        From here, EpiConfig and ExperimentConfig are loaded.

    Methods
    -------
    - `load_models()`
        The main/only function to be ran by users. The hidden functions are all to run internally.
            
    See Also
    --------
    #### Parentclass
    For more information on basic experiment-handling-behaviour see parent class ExperimentHandler.    
    """
    def __init__(self, 
                 experiment_name: str):
        
        super().__init__(experiment_name)     

        if not self.exp_exists:
            raise ExperimentDirectoryNotFoundError(f'directory for {experiment_name} not found under\n{self.path_exp}')

        self.epicfg     = self._load_epiconfig()
        self.expcfg     = self._load_experiment_config()
        self.models:    Optional[Dict[Union[int, str, float], List[BaseModel]]] = None
        self._set_dlms()

    # ======= METHODS =========== #
    def load_models(self, show_progress: bool = False) -> None:
        """ 
        returns a dictionary with values of the variable in key, and the list
        of models in value.

        See Also
        --------
        ### Helper methods:
        - `_parse_filename()`
        - `_get_dlm()`
        - `_load_model()`
        """
        varalias = self.expcfg.variable_alias       
        varvalues= self.expcfg.variable_values
        sep_char = self.expcfg.filename_seperator

        results: Dict[Union[int, str, float], List[BaseModel]] = {}

        logger.info('Loading models ...')

        iterator = varvalues if not show_progress else tqdm(varvalues, desc="Loading models per varvalue")

        for varvalue in iterator:

            files = [
                f for f in os.listdir(self.path_exp)
                if f.endswith(".pt") and f"{sep_char}{varalias}{varvalue}{sep_char}" in f
            ]

            models: List[BaseModel] = []

            for fname in files:
                try:
                    spec    = self._parse_filename(fname)
                    dlm     = self._get_dlm(spec.model, spec.variable_value, spec.graph)
                    model   = self._load_single_model(spec, dlm)
                    model.forecast()
                    models.append(model)

                except Exception as e:
                    logger.warning('Model %s could not be found', fname)

            results[varvalue] = models

        logger.info('Models loaded')

        self.models = results

    # ========= HIDDEN METHODS ======== #
    def _set_dlms(self) -> None:
        """Shared dataloader construction — called by Runner and Loader. Sets dlms into `dlms`."""
        dataloadermanagers: Dict[Union[int, str, float], ExperimentDLMs] = {}
        variable  = self.expcfg.variable
        varvalues = self.expcfg.variable_values

        logger.info('Setting dataloadermanagers ...')

        for varvalue in varvalues:
            cfg = self.epicfg.copy(**{variable: varvalue})
            epo = EpiDataOrchestrator(cfg).build()
            
            # graphs needs to be iterable: graph_list
            graphs_list = [] if self.expcfg.graphs is None else self.expcfg.graphs
    
            hl_dlms = ExperimentDLMs(
                baseline = BaseLineDataBuilder(epo),
                graphs   = {
                    graph: GraphDataBuilder(epo)
                               .retrieve_static_graph(graph)
                               .build()
                    for graph in graphs_list
                }
            )

            dataloadermanagers[varvalue] = hl_dlms

        self.dataloadermanagers = dataloadermanagers   

        logger.info('Dataloadermanagers set')

    def _load_experiment_config(self) -> 'ExperimentConfig':
        """returns ExperimentConfig"""
        return ExperimentConfig.load(self.path_exp / self.expcfg_filename)
    
    def _load_epiconfig(self) -> 'EpiConfig':
        """returns EpiConfig"""
        return EpiConfig.load_config(self.path_exp / self.epicfg_filename)        
        
    def _load_single_model(self, 
                    specs: ModelSpecs, 
                    dlm: GraphDataBuilder) -> DeepModel:
        """ 
        based on ModelSpecs, and using the dlm, instantiate a model.
        """
        modeltype = f"{specs.model}model"
        childclass = DeepModel._childclasses[modeltype]
        loaded_model = childclass.load_model(
            model_name          = specs.name,
            dir                 = str(self.path_exp),
            dataloadermanager   = dlm,
        )

        logger.info("model %s loaded", specs.name)
        return loaded_model

    def _parse_filename(self, filename: str) -> ModelSpecs:
        """ 
        splits filename into an instance of ModelSpecs with
        - variable (ex. hl)
        - varvalue (ex. 1)
        - modeltype (ex. lstm)
        - graph (default is None)
        - seed (ex 1)
        - name
        """
        splits      = filename.replace(".pt","").split(self.expcfg.filename_seperator)

        if len(splits) == 3:
            model, variable_value, sd = splits 
            graph = None
        
        elif len(splits) == 4:
            model, graph, variable_value, sd = splits         

        else:
            raise InvalidModelNameError(f'unexpected filename found: {filename}. Expected either 3 or 4 seperators (character: {self.expcfg.filename_seperator}) but got {len(splits)}.')

        return ModelSpecs(name      = filename, 
                   variable_alias   = self.expcfg.variable_alias,
                   variable_value   = int(variable_value.replace(self.expcfg.variable_alias, "")), # should be more dynamically resolved: currently ints
                   model            = model,
                   graph            = graph, 
                   seed             = int(sd.replace('s',""))
                   )