from typing import List, Union, Dict
import os
from tqdm import tqdm

from .issues import ExperimentDirectoryNotFoundError, InvalidModelNameError
from .handler import ExperimentHandler
from .containers import ExperimentConfig, ExperimentDLMs
from ..dataloading.epiconfig import EpiConfig
from ..utils.helpers import get_project_utilities_env
from ..models.deep.deepmodel import DeepModel
from ..models.base.basemodel import BaseModel
from ..dataloading.epidataorchestration import EpiDataOrchestrator
from ..dataloading.dataloaders import BaseLineDataLoaderManager, DeepDataLoaderManager, GraphDataLoaderManager
from .containers import ExperimentConfig, ModelSpecs
from ..dataloading.dataloaders import DeepDataLoaderManager, GraphDataLoaderManager

class ExperimentLoader(ExperimentHandler):
    """ 
    Loads experiments
    Subclass of ExperimentHandler

    See Also
    --------
    For more information, see ExperimentHandler

    Methods
    -------
    - `load_models()`
    """
    def __init__(self, 
                 experiment_name: str,
                 verbose:         int = 1):
        
        super().__init__(experiment_name, verbose)     

        if not self.path_exist:
            local_path = '/wissdaten/'+str(self.experiment_dir).split('/wissdaten/')[1]                             # remove the local_job piece
            raise ExperimentDirectoryNotFoundError(f'directory for {experiment_name} not found under\n{local_path}')

        self.epicfg     = self._load_epiconfig()
        self.expcfg     = self._load_experiment_config()
        self._set_dlms()

    # ======= METHODS =========== #
    def load_models(self) -> Dict[Union[int, str, float], List[BaseModel]]:
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

        for varvalue in varvalues:

            files = [
                f for f in os.listdir(self.experiment_dir)
                if f.endswith(".pt") and f"{sep_char}{varalias}{varvalue}{sep_char}" in f
            ]

            models: List[BaseModel] = []

            for fname in files:
                try:
                    spec    = self._parse_filename(fname)
                    dlm     = self._get_dlm(spec.model, spec.variable_value, spec.graph)
                    model   = self._load_model(spec, dlm)
                    models.append(model)

                except Exception as e:
                    print(f"✗ {fname}: {e}")

            results[varvalue] = models

        return results    

    # ========= HIDDEN METHODS ======== #
    def _set_dlms(self) -> None:
        """Shared dataloader construction — called by Runner and Loader. Sets dlms into `dlms`."""
        dataloadermanagers: Dict[int | str | float, ExperimentDLMs] = {}
        variable  = self.expcfg.variable
        varvalues = self.expcfg.variable_values

        if self.verbose > 0:
            iterator = tqdm(varvalues, desc = 'setting dataloadermanagers for varvalues')
        else:
            iterator = varvalues

        for varvalue in iterator:
            cfg = self.epicfg.copy(**{variable: varvalue})
            epo = EpiDataOrchestrator(cfg).build()
            
            # graphs needs to be iterable: graph_list
            graphs_list = [] if self.expcfg.graphs is None else self.expcfg.graphs
    
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

            dataloadermanagers[varvalue] = hl_dlms

        self.dataloadermanagers = dataloadermanagers   

    def _load_experiment_config(self) -> 'ExperimentConfig':
        """returns ExperimentConfig"""
        return ExperimentConfig.load(self.experiment_dir / self.experiment_cfg_filename, self.verbose > 1)
    
    def _load_epiconfig(self) -> 'EpiConfig':
        """returns EpiConfig"""
        return EpiConfig.load_config(self.experiment_dir / self.epicfg_filename, self.verbose > 1)        
        
    def _load_model(self, 
                    specs: ModelSpecs, 
                    dlm: Union[DeepDataLoaderManager, GraphDataLoaderManager]) -> DeepModel:
        """ 
        based on ModelSpecs, and using the dlm, instantiate a model.
        """
        modeltype = f"{specs.model}model"
        childclass = DeepModel._childclasses[modeltype]

        return childclass.load_model(
            model_name          = specs.name,
            subdir              = str(self.experiment_dir.name),
            dataloadermanager   = dlm,
        )

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
        
        # TODO
        return ModelSpecs(name      = filename, 
                   variable_alias   = self.expcfg.variable_alias,
                   variable_value   = int(variable_value.replace(self.expcfg.variable_alias, "")), # should be more dynamically resolved: currently ints
                   model            = model,
                   graph            = graph, 
                   seed             = int(sd.replace('s',""))
                   )