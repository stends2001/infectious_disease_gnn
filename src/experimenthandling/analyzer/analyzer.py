from typing import List, Union, Dict
import pandas as pd
from tqdm import tqdm

from ..loader import ExperimentLoader
from ...models.base.basemodel import BaseModel
from ...evaluation import Evaluator
from ...models import PersistenceModel, ClimateologyModel
from ...dataloading.epidataorchestration import EpiDataOrchestrator

from .plottermixin import ExperimentAnalyzerPlotterMixin
from .tablemixin import ExperimentAnalyzerTableMixin
from .autocorrelationmixin import ExperimentAnalyzerSpatialAutocorrMixin

class ExperimentAnalyzer(ExperimentLoader, 
                         ExperimentAnalyzerPlotterMixin, 
                         ExperimentAnalyzerTableMixin, 
                         ExperimentAnalyzerSpatialAutocorrMixin):
    """ 
    Analyzes experiments
    Subclass of ExperimentLoader, which is in turn a subclass ofExperimentHandler.

    See Also
    --------
    For more information, see parent classes:
    - ExperimentLoader
    - ExperimentHandler

    Methods
    -------
    - `compile_metrics()`
    
    For more methods, see ExperimentAnalyzerPlotterMixin
    """
    def __init__(self, 
                 experiment_name:   str,
                 verbose:           int = 1):
        
        super().__init__(experiment_name, verbose)            
        
    # ========= HIDDEN METHODS ========= #
    def _load_dataorchs(self) -> Dict[Union[int, str, float], EpiDataOrchestrator]:
        return {
            hl: self.dataloadermanagers[hl].baseline.dataorchestrator 
                  for hl in self.dataloadermanagers}

    def compile_metrics(self
    ) -> pd.DataFrame:
        """
        Evaluate all models across horizons and return flat metrics dataframe.
        Adds columns: horizon, seed, model_type, graph, model_color
        """
        self.epidataorchestrators   = self._load_dataorchs()
        self.models                 = self.load_models()
        
        all_metrics     = []
        varvalues_list  = self.expcfg.variable_values
        variable_alias  = self.expcfg.variable_alias

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

            all_models: List[BaseModel] = [persistence, climatology] + models

            for ml in all_models:
                ml.forecast()

            evaluator  = Evaluator(all_models, verbose = self.verbose)
            evaluator.add_evaluation(horizon=0, dataset='test')

            metrics_hl = evaluator.data_compilation.get_data(0, 'test')['metrics'].copy()
            metrics_hl[variable_alias] = varvalue

            # parse model name into components
            def parse_name(name: str) -> dict:
                parts = name.replace('.pt', '').split('-')
                # examples: gcn2-graph1-hl1-s42, lstm-hl1-s42, persistence-hl1, climateology-hl1
                if parts[0] in ('persistence', 'climateology'):
                    return {'model_type': parts[0], 'graph': None, 'seed': None}
                if len(parts) == 4:  # gcn2-graph1-hl1-s42
                    return {'model_type': parts[0], 'graph': parts[1], 'seed': int(parts[3].replace('s', ''))}
                if len(parts) == 3:  # lstm-hl1-s42
                    return {'model_type': parts[0], 'graph': None, 'seed': int(parts[2].replace('s', ''))}
                return {'model_type': parts[0], 'graph': None, 'seed': None}

            parsed = metrics_hl['model'].apply(lambda n: pd.Series(parse_name(n)))
            metrics_hl = pd.concat([metrics_hl, parsed], axis=1)

            # assign colors — differentiate by graph type for GCN
            color_map = {
                ('persistence',  None)     : '#9E9E9E',
                ('climateology', None)     : '#9C27B0',
                ('lstm',         None)     : '#4CAF50',
                ('gcn2',         'graph1') : '#BBDEFB',   # identity — light blue
                ('gcn2',         'graph2') : '#2196F3',   # geo — blue
                ('gcn2',         'graph3') : '#FF9800',   # random — orange
                ('gcn2',         'graph4') : '#F44336',   # commuter — red
            }
            metrics_hl['model_color'] = metrics_hl.apply(
                lambda r: color_map.get((r['model_type'], r['graph']), '#BDBDBD'), axis=1
            )

            # clean model label for plotting
            metrics_hl['model_label'] = metrics_hl.apply(
                lambda r: r['model_type'] if r['graph'] is None else f"{r['model_type']}\n{r['graph']}", axis=1
            )

            all_metrics.append(metrics_hl)
            del evaluator

            if self.verbose > 1:
                verbose_line = f"{variable_alias} = {varvalue} done"

                print(verbose_line)

        self.metrics_df = pd.concat(all_metrics, ignore_index=True)
