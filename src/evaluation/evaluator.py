from typing import Union, List, Literal, Union, Optional, Dict
import pandas as pd
from tqdm import tqdm 

from .metrics import Metrics
from ..utils.textformatting import align, warning_emoji
from ..models.base.basemodel import BaseModel
from .evaluationplotter import EvaluationPlotter
from .containers import PredictionCompilation, MetricCompilation

class Evaluator:

    """
    Evaluates models on spatiotemporal epidemioligal predictions

    Examples:
    --------
    >>> evaluator = Evaluator([ml1, ml2])
    >>> evaluator.add_evaluation()
    >>> evaluator.plotter.plot_metric("ccc", plot_type = "map")

    See Also:
    --------
    EvaluationPlotter
    Metrics
    """

    def __init__(self, models: Union[BaseModel, List[BaseModel]]):
        models_list             = models if isinstance(models, list) else [models]
        self.evaluated_models   = {ml.clean_name: ml for ml in models_list}
        
        # Column names
        self.target_col     = 'target'
        self.pred_col       = 'pred'
        self.id_col         = 'node'
        self.temporal_col   = 'timestamp'
        
        # Storage
        self.evaluation_entries: Dict[str, pd.DataFrame]         = {}
        self.metrics = ['mse','rmse','spearman_corr', 'pearson_corr' ,'ccc','node_smape']


        self.plotter = EvaluationPlotter(self)

        self.prediction_compilations = PredictionCompilation()
        self.metric_compilations     = MetricCompilation()
            
    def add_evaluation(self, 
                       horizon:     int  = 0,
                       dataset:     Literal['train', 'val', 'test'] = 'test') -> 'Evaluator':
        """
        Add evaluation entry for specified horizon.
        
        Parameters:
        -----------
        horizon : int
            Prediction horizon
        dataset : str
            Which dataset to evaluate
        """
        if dataset in self.prediction_compilations.compilations:
            if f'horizon_{horizon}' in self.prediction_compilations.compilations[dataset]:
                print(f'{warning_emoji} horizon_{horizon} already exists for {dataset}')
                return self 
        else:
            compilation_preds   = self._compile_predictions(horizon, dataset)
            self.prediction_compilations.add_horizon(compilation_preds, f'horizon_{horizon}', dataset)

            # compute metrics
            compilation_metrics = self._compile_metrics(horizon, dataset)
            self.metric_compilations.add_horizon(compilation_metrics, f'horizon_{horizon}', dataset)
        return self

    def _get_metrics_calculator(self, model: BaseModel) -> Metrics:
        """Create metrics calculator with model's graph structure."""
        edge_index  = getattr(model.dataloadermanager, 'edge_index', None)
        edge_weight = getattr(model.dataloadermanager, 'edge_weight', None)
        
        return Metrics(
            target_col  = self.target_col,
            pred_col    = self.pred_col,
            id_col      = self.id_col,
            temporal_col= self.temporal_col,
            edge_index  = edge_index,
            edge_weight = edge_weight
        )

    def _compile_metrics(self, horizon, dataset) -> dict:
        metrics_dict = {}
        for metric_name in tqdm(self.metrics, desc = 'computing metrics'):
            metric_df = self._compute_all_models_metric(
                metric_name, horizon, dataset
            )
            metrics_dict[metric_name] = metric_df
        return metrics_dict

    def _compile_predictions(self, horizon, dataset) -> pd.DataFrame:
        predictions_compilation = None

        for name, model in self.evaluated_models.items():
            model_predictions = model.predictions.get_preds(dataset).get_original(horizon).rename(columns = {'pred' : f'pred_{name}'})
            if predictions_compilation is None:
                predictions_compilation= model_predictions
            else:
                predictions_compilation = pd.merge(predictions_compilation, model_predictions[['timestamp','node',f'pred_{name}']], on = ['timestamp','node'])

        if predictions_compilation is None:
            raise IndexError(f'predictions_compilation is invalid')        
        
        return predictions_compilation

    def _compute_standard_metric(self, df: pd.DataFrame, 
                                 metric_name: str, 
                                 model: BaseModel) -> pd.DataFrame:
        """Compute standard metric per node using groupby."""
        calculator  = self._get_metrics_calculator(model)
        metric_func = getattr(calculator, metric_name)
        
        # Drop id_col before groupby to avoid the warning
        df_for_grouping = df.drop(columns=[self.id_col])
        
        return (
            df_for_grouping
            .groupby(df[self.id_col])  # Group by the original column
            .apply(metric_func)
            .rename(metric_name)
            .reset_index()
        )
   
    def _compute_all_models_metric(self, 
                                   metric_name: str,
                                   horizon,
                                   dataset) -> pd.DataFrame:
        """Compute specified metric for all models."""
        metric_df = None
        
        for name, model in self.evaluated_models.items():
            model_predictions = model.predictions.get_preds(dataset).get_original(horizon)

            # Route to appropriate computation method
            result = self._compute_standard_metric(model_predictions, metric_name, model)

            result.columns.values[-1] = name
            
            if metric_df is None:
                metric_df = result
            else:
                metric_df = pd.merge(metric_df, result, on=[self.id_col])

        if metric_df is None:
            raise ValueError('No metric dataframe found. Something is wrong in the evaluation datasets')
                
        return metric_df