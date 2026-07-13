from typing import List, Union, Dict, Optional
import pandas as pd
from tqdm import tqdm

from ..exceptions import DataLoaderManagerError

from ..loader import ExperimentLoader
from .metricsdfmixin import MetricsDFStateMixin
from .resultsmixin import ResultsMixin
from .moransmixin import MoransAnalysisMixin

from ...models.base.basemodel import BaseModel
from ...evaluation import Evaluator
from ...models import PersistenceModel, ClimateologyModel
from ...dataloading.epidataorchestration import EpiDataOrchestrator

import logging
logger = logging.getLogger(__name__)

class ExperimentAnalyzer(ExperimentLoader, 
                         MetricsDFStateMixin,
                         ResultsMixin,
                         MoransAnalysisMixin):
    """ 
    Loads experiments and analyzes them too
    First degree subclass of ExperimentAnalyzer
    Second degree subclass of ExperimentHandler

    Parameters
    ----------
    experiment_name: str
        name of the experiment. This string should be identical to the directory in which the models and configs are saved.
        From here, EpiConfig and ExperimentConfig are loaded.

    Examples
    --------
    #### Loading and saving an experiment's metrics
    >>> exp1a = ExperimentAnalyzer('experiment_1a')
    >>> exp1a.compile_metrics()
    >>> exp1a.save_metrics()

    #### Loading earlier-used 
    >>> exp1a = ExperimentAnalyzer('experiment_1a')
    >>> exp1a.load_metrics()

    Methods
    -------
    - `compile_metrics()`
            
    See Also
    --------
    #### Mixinclasses
    - MetricsDFStateMixin: deals with saving/loading of metrics_df
    - ResultsMixin: deals with plotting
    - MoransAnalysisMixin: deals with Moran's analysis

    #### Parentclass
    ###### 1st degree 
    For more information on basic experiment-loading-behaviour, see parent class ExperimentAnalyzer    
    ###### 2nd degree
    For more information on basic experiment-handling-behaviour, see parent class ExperimentHandler.      
    """
    evaluators: Dict[Union[int, str, float], Evaluator]

    metrics_df_filename = 'metrics.csv'

    def __init__(self, 
                 experiment_name:   str):
        
        super().__init__(experiment_name)            
        
        self.epidataorchestrators               = self._get_dataorchs() 
        self.metrics_df: Optional[pd.DataFrame] = None
        self.metrics_to_calculate: Optional[List[str]] = None
        self.variable_alias  = self.expcfg.variable_alias    # the alias of this variable

    def compile_metrics(self, show_progress: bool = False):
        """
        Evaluate all models across horizons and return flat metrics dataframe.
        Adds columns: horizon, seed, model_type, graph, model_color
        """
        if self.models is None:
            self.load_models()
        
        assert self.models is not None            
        self.evaluators = {}
        
        all_metrics     = []
        varvalues_list  = self.expcfg.variable_values   # a list of all values for the variable over which the experiment iteratres


        # Get metrics iteratively per varvalue, model-wise
        iterator = self.models.items()

        if show_progress:
            iterator = tqdm(
                iterator,
                desc=f"Compiling evaluators per value of {self.variable_alias}",
                total=len(self.models),
            )

        for varvalue, models in iterator:

            if varvalue not in varvalues_list:
                raise ValueError(f'value {varvalue} should not exist!')
            
            varvalue_models = self._get_varvalue_models(models, varvalue)
            varvalue_metrics= self._get_varvalue_metrics(varvalue_models, varvalue)

            all_metrics.append(varvalue_metrics)

        metrics_df          = self._merge_and_process_metrics(all_metrics)
        self.model_names    = list(metrics_df['model'].unique())
        self._set_modelcolors()
        
        metrics_df['model_color'] = metrics_df['model'].replace(self.model_colors) 

        self.metrics_df = metrics_df
        logger.info('metrics compiled')

    # ========= HIDDEN METHODS ========= #
    def _get_dataorchs(self) -> Dict[Union[int, str, float], EpiDataOrchestrator]:
        """extracts the epidataorchestrators from the `dlms`"""
        if self.dataloadermanagers is None:
            raise DataLoaderManagerError('no dataloadermanagers found')
        
        return {
            varvalue: self.dataloadermanagers[varvalue].baseline.dataorchestrator 
                  for varvalue in self.dataloadermanagers}

    def _merge_and_process_metrics(self, all_metrics: List[pd.DataFrame]) -> pd.DataFrame:
        """merge all varvalue - individual metric dfs and get the right order of columns"""
        mgd_metrics_df          = pd.concat(all_metrics, ignore_index=True)        
        model_name_remapping    = self._map_modelnames(list(mgd_metrics_df['model'].unique()))
        mgd_metrics_df['model'] = mgd_metrics_df['model'].replace(model_name_remapping)
        
        assert self.metrics_to_calculate is not None
        column_order = ['model','node', self.variable_alias] + self.metrics_to_calculate
        
        return mgd_metrics_df[column_order]

    def _get_varvalue_models(self, models: List[BaseModel], varvalue: int | str | float) -> List[BaseModel]:
        """returns a list of models: not just the experimented one (seediwse) but also baseline ones."""

        # `_get_dataorchs()` already asserts that this is the case, which is internally called upon init.        
        assert self.dataloadermanagers is not None                          
        baseline_dlm = self.dataloadermanagers[varvalue].baseline

        persistence  = PersistenceModel(baseline_dlm,  f'persistence-{self.variable_alias}{varvalue}')
        climatology  = ClimateologyModel(baseline_dlm, f'climateology-{self.variable_alias}{varvalue}')

        persistence.forecast('test')
        climatology.forecast('test')

        # all_models: List[BaseModel] = [persistence, climatology] + models
        all_models: List[BaseModel] = sorted(
            [persistence, climatology] + models,
            key=lambda m: m.name
        )      
        return all_models    

    def _get_varvalue_metrics(self, models: List[BaseModel], varvalue: int | str | float) -> pd.DataFrame:
        """
        constructs an evaluator and extracts calculated metrics
        
        See Also
        --------
        Evaluator
            much of the work is done by this class
        """
        evaluator  = Evaluator(models)
        evaluator.add_evaluation(horizon=0, dataset='test')
        self.evaluators[varvalue] = evaluator     

        metrics_varvalue = evaluator.data_compilation.get_data(0, 'test')['metrics'].copy()
        metrics_varvalue[self.variable_alias] = varvalue           

        if self.metrics_to_calculate is None:
            self.metrics_to_calculate = evaluator.metric_calculator.supported_metrics

        return metrics_varvalue
           
    def _map_modelnames(self, current_names: List[str])-> Dict[str,str]:
        # at this point, metrics are aggregated over seeds so instead of
        # gcn2-graph1-hl1-s42 and gcn2-graph1-hl1-s123 we now have a value per node for gcn2-graph1-hl1
        # we thus split the modelname to gcn2-graph1 to put the variable tested in a different column
        splitter = self.expcfg.filename_seperator+self.expcfg.variable_alias        
        mapping  = {}

        for mlname in current_names:
            mapping[mlname] = mlname.split(splitter)[0].replace("-","")
        return mapping

