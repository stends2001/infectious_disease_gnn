from typing import Union, List, Literal, Union, Optional, Dict
import pandas as pd
from tqdm import tqdm 

from ..utils import check_dataset
from .metrics import ClassificationMetrics, RegressionMetrics
from ..utils.textformatting import align, warning_emoji
from ..models.base.basemodel import BaseModel
from .evaluationplotter import EvaluationPlotter
from .containers import PredictionCompilation, MetricCompilation

class Evaluator:

    """
    """

    def __init__(self, models: Union[BaseModel, List[BaseModel]]):
        models_list             = models if isinstance(models, list) else [models]
        self.evaluated_models   = {ml.name: ml for ml in models_list}
        self._validate_prediction_modes()
        
        # Column names
        self.target_col     = 'target'
        self.pred_col       = 'pred'
        self.id_col         = models[0].epiconfig.id_column
        self.temporal_col   = 'timestamp'
        
        # Storage
        self.evaluation_entries: Dict[str, pd.DataFrame]         = {}

        self.plotter = EvaluationPlotter(self)

        self.prediction_compilations = PredictionCompilation()
        self.metric_compilations     = MetricCompilation()
        self._setup_metric_calculator()
  
    def _setup_metric_calculator(self):
        """iniates self.metric_calculator attribute"""
        if self.prediction_mode == 'classification':
            self.metric_calculator = ClassificationMetrics(self.target_col, self.pred_col, None, self.id_col, self.temporal_col)
        elif self.prediction_mode == 'regression':
            self.metric_calculator = RegressionMetrics(self.target_col, self.pred_col, self.id_col, self.temporal_col)

    @check_dataset()
    def add_evaluation(self, 
                       horizon:     int  = 0,
                       dataset:     Literal['train', 'val', 'test'] = 'test') -> 'Evaluator':
        """
        Add evaluation entry for specified horizon.
        
        Parameters
        -----------
        horizon : int
            Prediction horizon
        dataset : str
            Which dataset to evaluate
        """
        # in case prediction compilation already established
        if dataset in self.prediction_compilations.compilations:
            if f'horizon_{horizon}' in self.prediction_compilations.compilations[dataset]:
                print(f'{warning_emoji} horizon_{horizon} already exists for {dataset}')
                return self 
            
        # if prediction compilation doesn't exist yet
        else:
            compilation_preds   = self._compile_predictions(horizon, dataset)
            self.prediction_compilations.add_horizon(compilation_preds, f'horizon_{horizon}', dataset)

            # compute metrics
            compilation_metrics = self._compile_metrics(horizon, dataset)
            self.metric_compilations.add_horizon(compilation_metrics, f'horizon_{horizon}', dataset)
        return self

    @check_dataset()
    def _compile_metrics(self, horizon, dataset) -> dict:
        metrics_dict = {}
        for metric_name in tqdm(self.metric_calculator.supported_metrics, desc = 'computing metrics'):
            
            metric_df = self._compute_all_models_metric(
                metric_name, horizon, dataset
            )
            metrics_dict[metric_name] = metric_df
        return metrics_dict

    @check_dataset()
    def _compile_predictions(self, horizon, dataset) -> pd.DataFrame:
        predictions_compilation = None

        for name, model in self.evaluated_models.items():
            model_predictions = model.predictions.get_preds(dataset).get_original(horizon).rename(columns = {'pred' : f'pred_{name}'})
            if predictions_compilation is None:
                predictions_compilation= model_predictions
            else:
                predictions_compilation = pd.merge(predictions_compilation, model_predictions[[self.temporal_col,self.id_col ,f'pred_{name}']], on = [self.temporal_col,self.id_col ])

        if predictions_compilation is None:
            raise IndexError(f'predictions_compilation is invalid')        
        
        return predictions_compilation

    def _compute_standard_metric(self, df: pd.DataFrame, 
                                 metric_name: str, 
                                 model: BaseModel) -> pd.DataFrame:
        """Compute standard metric per node using groupby."""
        metric_func = getattr(self.metric_calculator, metric_name)
        
        # Drop id_col before groupby to avoid the warning
        df_for_grouping = df.drop(columns=[self.id_col])
        
        return (
            df_for_grouping
            .groupby(df[self.id_col])  # Group by the original column
            .apply(metric_func)
            .rename(metric_name)
            .reset_index()
        )
   
    @check_dataset()   
    def _compute_all_models_metric(self, 
                                   metric_name: str,
                                   horizon,
                                   dataset) -> pd.DataFrame:
        """Compute specified metric for all models."""
        metric_df = None
        
        for name, model in self.evaluated_models.items():
            model_predictions = model.predictions.get_preds(dataset).get_original(horizon)
            result = self._compute_standard_metric(model_predictions, metric_name, model)

            result.columns.values[-1] = name
            
            if metric_df is None:
                metric_df = result
            else:
                metric_df = pd.merge(metric_df, result, on=[self.id_col])

        if metric_df is None:
            raise ValueError('No metric dataframe found. Something is wrong in the evaluation datasets')
                
        return metric_df
    
    def _validate_prediction_modes(self):
        """cross checks all prediction modes"""
        self.prediction_mode = None
        for mlname, ml in self.evaluated_models.items():
            if self.prediction_mode is None:
                self.prediction_mode = ml.prediction_mode
            else:
                if self.prediction_mode != ml.prediction_mode:
                    raise ValueError('incompatible prediction modes accross models found!')

            