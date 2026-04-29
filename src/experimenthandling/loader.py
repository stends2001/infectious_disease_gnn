from pathlib import Path 
from typing import List, Union, Dict
import os

from .handler import ExperimentHandler
from .containers import ExperimentConfig
from ..dataloading.epiconfig import EpiConfig
from ..utils.helpers import get_project_utilities_env
from ..models.deep.deepmodel import DeepModel
from ..models.base.basemodel import BaseModel

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
    def __init__(self, experiment: str):
        self.experiment     = experiment
        self.experiment_dir = Path(get_project_utilities_env()) / "models" / experiment

        epiconfig      = self._load_epiconfig()
        experiment_cfg = self._load_experiment_config()

        super().__init__(epiconfig, experiment_cfg)     
        
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
        varalias = self.experiment_cfg.variable_alias       
        varvalues= self.experiment_cfg.variable_values
        sep_char = self.experiment_cfg.filename_seperator

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
    def _load_experiment_config(self) -> 'ExperimentConfig':
        """returns ExperimentConfig"""
        return ExperimentConfig.load(self.experiment_dir / self.experiment_cfg_filename)
    
    def _load_epiconfig(self) -> 'EpiConfig':
        """returns EpiConfig"""
        return EpiConfig.load_config(self.experiment_dir / self.epicfg_filename)        
        
    def _load_model(self, 
                    specs: ModelSpecs, 
                    dlm: Union[DeepDataLoaderManager, GraphDataLoaderManager]) -> DeepModel:
        """ 
        based on ModelSpecs, and using the dlm, instantiate a model.
        """
        modeltype = f"{specs.model}model"
        childclass = DeepModel._childclasses[modeltype]

        return childclass.load_model(
            model_name=specs.name,
            subdir=str(self.experiment_dir.name),
            dataloadermanager=dlm,
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
        splits      = filename.replace(".pt","").split(self.experiment_cfg.filename_seperator)

        if len(splits) == 3:
            model, variable_value, sd = splits 
            graph = None
        
        elif len(splits) == 4:
            model, graph, variable_value, sd = splits         

        else:
            raise ValueError(f'unexpected filename found: {filename}')
        
        return ModelSpecs(name = filename, 
                   variable_alias = self.experiment_cfg.variable_alias,
                   variable_value = int(variable_value.replace(self.experiment_cfg.variable_alias, "")), # should be more dynamically resolved: currently ints
                   model = model,
                   graph = graph, 
                   seed     = int(sd.replace('s',""))
                   )