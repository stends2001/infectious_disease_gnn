from typing import List, Union, Dict, Optional
import pandas as pd
from tqdm import tqdm

from ..loader import ExperimentLoader
from .metricxdfmixin import MetricsDFStateMixin
from .resultsmixin import ResultsMixin
from .moransmixin import MoransAnalysisMixin

from ...models.base.basemodel import BaseModel
from ...evaluation import Evaluator
from ...models import PersistenceModel, ClimateologyModel
from ...dataloading.epidataorchestration import EpiDataOrchestrator

class ExperimentAnalyzer(ExperimentLoader, 
                           MetricsDFStateMixin,
                           ResultsMixin,
                           MoransAnalysisMixin):
    """ 
    """
    evaluators: Dict[Union[int, str, float], Evaluator]


    def __init__(self, 
                 experiment_name:   str,
                 verbose:           int = 1):
        
        super().__init__(experiment_name, verbose)            
        
        self.epidataorchestrators               = self._get_dataorchs() 
        self.metrics_df: Optional[pd.DataFrame] = None
        self.metrics_to_calculate: Optional[List[str]] = None
        self.variable_alias  = self.expcfg.variable_alias    # the alias of this variable

    def compile_metrics(self):
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
        iterator = tqdm(self.models.items(), desc = f'compiling evaluators per value of {self.variable_alias}')

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

    # ========= HIDDEN METHODS ========= #
    def _get_dataorchs(self) -> Dict[Union[int, str, float], EpiDataOrchestrator]:
        """extracts the epidataorchestrators from the `dlms`"""
        if self.dataloadermanagers is None:
            raise ValueError('no dataloadermanagers found')
        
        return {
            varvalue: self.dataloadermanagers[varvalue].baseline.dataorchestrator 
                  for varvalue in self.dataloadermanagers}

    def _merge_and_process_metrics(self, all_metrics: List[pd.DataFrame]) -> pd.DataFrame:
        mgd_metrics_df          = pd.concat(all_metrics, ignore_index=True)        
        model_name_remapping    = self._map_modelnames(list(mgd_metrics_df['model'].unique()))
        mgd_metrics_df['model'] = mgd_metrics_df['model'].replace(model_name_remapping)
        assert self.metrics_to_calculate is not None
        return mgd_metrics_df[['model','node', self.variable_alias] + self.metrics_to_calculate]

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
        evaluator  = Evaluator(models, verbose = self.verbose)
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

