from typing import List, Union, Dict
import pandas as pd
from tqdm import tqdm

from ..loader import ExperimentLoader
from ...models.base.basemodel import BaseModel
from ...evaluation import Evaluator
from ...models import PersistenceModel, ClimateologyModel
from ...dataloading.epidataorchestration import EpiDataOrchestrator

from .plottermixin import ExperimentAnalyzerPlotterMixin

class ExperimentAnalyzer(ExperimentLoader, 
                         ExperimentAnalyzerPlotterMixin):
    """ 
    Analyzes experiments
    Subclass 1st degree of ExperimentLoader
    Sublcass 2nd degree of ExperimentHandler
    
    Parameters
    -------
    experiment_name: str
        name of the experiment. This string should be identical to the directory in which the models and configs are saved.
        From here, EpiConfig and ExperimentConfig are loaded.
    verbose: int
        the level to which output/updates are to be returned.

    Methods
    -------
    #### Main
    - `compile_metrics()`

    #### Mixins
    ##### ExperimentAnalyzerPlotterMixin
    - `plot_metric_over_reference()`
    - `plot_graph_advantage()`

    Mixins
    ------
    ExperimentAnalyzerPlotterMixin
        Allows plotting of experiment

    See Also
    --------
    #### Parentclass
    For more information on basic experiment-handling-behaviour see (1st degree) parent class ExperimentHandler.    
    For more information on experiment-loading-behaviour see (2nd degree) parent class ExperimentLoader.    

    #### Helper classes
    for evaluation-specific (i.e. how are metrics defined and computed) information, see Evaluator.
    """
    def __init__(self, 
                 experiment_name:   str,
                 verbose:           int = 1):
        
        super().__init__(experiment_name, verbose)            
        
    # ========= HIDDEN METHODS ========= #
    def _load_dataorchs(self) -> Dict[Union[int, str, float], EpiDataOrchestrator]:
        if self.dataloadermanagers is None:
            raise ValueError('no dataloadermanagers found')
        
        return {
            hl: self.dataloadermanagers[hl].baseline.dataorchestrator 
                  for hl in self.dataloadermanagers}

    def compile_metrics(self
    ) -> pd.DataFrame:
        """
        Evaluate all models across horizons and return flat metrics dataframe.
        Adds columns: horizon, seed, model_type, graph, model_color
        """
        self.epidataorchestrators                               = self._load_dataorchs()
        self.models                                             = self.load_models()
        self.evaluators: Dict[Union[int, str, float], Evaluator]= {}
        
        color_map       = {}
        all_metrics     = []
        varvalues_list  = self.expcfg.variable_values
        variable_alias  = self.expcfg.variable_alias

        if self.dataloadermanagers is None:
            raise ValueError('no dataloadermanagers found')

        if self.verbose <= 0 and self.verbose > 1:
            iterator = self.models.items()

        else:
            iterator = tqdm(self.models.items(), desc = f'compiling evaluators per value of {variable_alias}')

        for varvalue, models in iterator:

            # TODO
            if varvalue not in varvalues_list:
                raise ValueError(f'value {varvalue} should not exist!')

            baseline_dlm = self.dataloadermanagers[varvalue].baseline

            persistence  = PersistenceModel(baseline_dlm,  f'persistence-{variable_alias}{varvalue}')
            climatology  = ClimateologyModel(baseline_dlm, f'climateology-{variable_alias}{varvalue}')

            persistence.forecast('test')
            climatology.forecast('test')

            # all_models: List[BaseModel] = [persistence, climatology] + models
            all_models: List[BaseModel] = sorted(
                [persistence, climatology] + models,
                key=lambda m: m.name
            )            

            for ml in all_models:
                ml.forecast()

            evaluator  = Evaluator(all_models, verbose = self.verbose)
            evaluator.add_evaluation(horizon=0, dataset='test')

            # TODO remove this junk: want everything to flow from evaluators.
            metrics_hl = evaluator.data_compilation.get_data(0, 'test')['metrics'].copy()
            metrics_hl[variable_alias] = varvalue

            # parse model name into components
            def parse_name(name: str) -> dict:
                parts = name.replace('.pt', '').split('-')
                # examples: gcn2-graph1-hl1-s42, lstm-hl1-s42, persistence-hl1, climateology-hl1
                if parts[0] in ('persistence', 'climateology'):
                    return {'model_type': parts[0], 'graph': None}
                if len(parts) == 3:  # gcn2-graph1-hl1
                    return {'model_type': parts[0], 'graph': parts[1]}
                if len(parts) == 2:  # lstm-hl1
                    return {'model_type': parts[0], 'graph': None}
                return {'model_type': parts[0], 'graph': None}

            parsed      = metrics_hl['model'].apply(lambda n: pd.Series(parse_name(n)))
            metrics_hl  = pd.concat([metrics_hl, parsed], axis=1)
            metrics_hl['model_color'] = metrics_hl['model'].replace(evaluator.model_colors)
            
            all_metrics.append(metrics_hl)
            self.evaluators[varvalue] = evaluator
    
            if self.verbose > 1:
                verbose_line = f"{variable_alias} = {varvalue} done"

                print(verbose_line)

        mgd_metrics_df  = pd.concat(all_metrics, ignore_index=True)
        self.metrics_df = mgd_metrics_df[['node', variable_alias, 'model_type', 'graph', 'model_color'] + evaluator.metric_calculator.supported_metrics]