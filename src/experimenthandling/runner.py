from pathlib import Path 
from typing import Optional, Union, Dict, Type
import os

from .handler import ExperimentHandler
from .containers import ExperimentConfig
from ..dataloading.epiconfig import EpiConfig
from ..dataloading.epidataorchestration import EpiDataOrchestrator
from ..utils.helpers import get_project_utilities_env
from ..models.deep.deepmodel import DeepModel
from .containers import ExperimentConfig, ExperimentDLMs
from ..dataloading.dataloaders import BaseLineDataLoaderManager, DeepDataLoaderManager, GraphDataLoaderManager

class ExperimentRunner(ExperimentHandler):
    """ 
    Runs experiments
    Subclass of ExperimentHandler

    See Also
    --------
    For more information, see ExperimentHandler

    Methods
    -------
    - `run()`
    """
    def __init__(self,
                epiconfig:        EpiConfig,
                experimentconfig: ExperimentConfig):

        self.experiment     = experimentconfig.experiment_name
        self.experiment_dir = Path(get_project_utilities_env()) / "models" / self.experiment
        self.path_exist     = os.path.exists(self.experiment_dir)

        # store the new values before merging; preventing new experiment addition from running everything
        self.new_variable_values = experimentconfig.variable_values

        experiment_cfg = self._update_experiment_cfg(epiconfig, experimentconfig)
        super().__init__(epiconfig, experiment_cfg)
    
    # ======= METHODS ======= #
    def run(self, global_hparams: dict):
        """
        Main function; runs the experiment

        Feed in global_hparams for deepmodel instances
        """
        self._save_cfgs()
        
        graphlist = [None] if self.experiment_cfg.graphs is None else self.experiment_cfg.graphs

        for value in self.new_variable_values:
            for ml in self.experiment_cfg.models:
                
                modeltype  = f"{ml}model"
                childclass = DeepModel._childclasses[modeltype]

                for sd in self.experiment_cfg.seeds:

                    match childclass._expected_dataloadermanager:

                        case 'GraphDataLoaderManager':
                            for graph in graphlist:
                                modelname = self._get_model_name(ml, value, sd, graph)
                                if self._model_to_run(modelname):
                                    model = self._load_model(modelname, childclass, value, ml, graph)
                                    model.set_model_hparams()
                                    model.set_global_hparams(**global_hparams)
                                    model.train()
                                    model.save_model(dir=Path(self.experiment))

                        case 'DeepDataLoaderManager':
                            modelname = self._get_model_name(ml, value, sd, None)
                            if self._model_to_run(modelname):
                                model = self._load_model(modelname, childclass, value, ml)
                                model.set_model_hparams()
                                model.set_global_hparams(**global_hparams)
                                model.train()
                                model.save_model(dir=Path(self.experiment))
    
    # ======= HIDDEN METHODS ======== #
    def _update_experiment_cfg(self, new_epiconfig: EpiConfig , new_experimentconfig: ExperimentConfig) -> ExperimentConfig:
        """
        updates ExperimentConfig instance if necessary. Validates compatibility, 
        also for epiconfig, with pre-existing experiment-directory.
        """
        # if path does exist, save epiconfig and experimentconfig into pre-existing values (prex) and validate compatibility
        if self.path_exist:
            prex_cfg      = ExperimentConfig.load(self.experiment_dir / self.experiment_cfg_filename)
            prex_epiconfig= EpiConfig.load_config(self.experiment_dir / self.epicfg_filename)
            prex_epiconfig.assert_equals(new_epiconfig)
            return prex_cfg.merge(new_experimentconfig)
        
        # else, experimentconfig doesn't change
        else:
            return new_experimentconfig
        
    def _set_dlms(self) -> None:
        """Only build dataloaders for the new variable values being run."""
        dataloader_managers_dict: Dict[Union[int, str, float], ExperimentDLMs] = {}
        variable = self.experiment_cfg.variable

        for varvalue in self.new_variable_values:
            cfg = self.epiconfig.copy(**{variable: varvalue})
            epo = EpiDataOrchestrator(cfg).build()

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
    
    def _load_model(self, modelname: str, childclass: Type['DeepModel'], value: Union[int, str, float], ml: str, graph: Optional[str]=None) -> 'DeepModel':
        """load instance of model"""
        dlm = self._get_dlm(ml, value, graph)                       # graph=None for DeepDataLoaderManager
        return childclass(name=modelname, dataloadermanager=dlm)    # type: ignore

    def _get_model_name(self, modeltype: str, varvalue: Union[str, float, int], seed: int, graph: Optional[str] = None) -> str:
        """return model name: to be saved as"""
        sep = self.experiment_cfg.filename_seperator
        var = self.experiment_cfg.variable_alias

        if graph is None:
            name = f"{modeltype}{sep}{var}{varvalue}{sep}s{seed}"

        else:
            name = f"{modeltype}{sep}{graph}{sep}{var}{varvalue}{sep}s{seed}"            
        
        return name
        
    def _save_cfgs(self) -> None:
        """
        saves ExperimentConfig and EpiConfig. 
        Two options: experiment folder already exists, or it does not.
        """
        # if it does not exist, make it and save epiconfig.
        if not self.path_exist:
            os.mkdir(self.experiment_dir)
            self.epiconfig.save_config(self.experiment_dir / '_epiconfig.yaml') 
        # save experimentconfig
        self.experiment_cfg.save_config(self.experiment_dir / '_experiment_config.yaml')        
       
    def _model_to_run(self, modelname: str) -> bool:
        """boolean on whether the modelname already exists or not."""
        saved_models = {
            os.path.splitext(f)[0] 
            for f in os.listdir(self.experiment_dir) 
            if f.endswith('.pt')
        }
        return modelname not in saved_models