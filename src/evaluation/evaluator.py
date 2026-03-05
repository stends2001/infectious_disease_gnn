from typing import Union, List, Literal, Union, Optional, Dict
import pandas as pd
from tqdm import tqdm 

from .containers import EvaluationPredictionsCompilation
from .metrics import QuantileRegressionMetricsCalculator, PointRegressionMetricsCalculator

from ..utils import check_dataset, warning_emoji
from ..models.base.basemodel import BaseModel
from ..dataloading import EpiConfig

from .evaluationplotter import EvaluationPlotter

class Evaluator:

    """
    Evaluates models

    Parameters
    ----------
    models: List of models
    """

    def __init__(self, models: List[BaseModel]):
        models_list             = models if isinstance(models, list) else [models]
        self.evaluated_models   = {ml.name: ml for ml in models_list}
        self.model_names        = list(self.evaluated_models.keys())
        self.epiconfig          = self._validate_epiconfigs()
        
        # Column names
        self.target_col     = 'target'
        self.pred_cols      = models[0].pred_cols
        self.id_col         = self.epiconfig.id_column
        self.temporal_col   = self.epiconfig.temporal_column
        
        # Storage
        self.evaluation_entries: Dict[str, pd.DataFrame]         = {}

        self.plotter                = EvaluationPlotter(self)
        self.data_compilation       = EvaluationPredictionsCompilation(self.model_names)
        self.metric_calculator      = self._setup_metric_calculator()

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
        horizon_str = f'horizon_{horizon}'

        # in case prediction compilation already established
        if dataset in self.data_compilation.datasets:
            if horizon_str in self.data_compilation.horizons[dataset]:
                print(f'{warning_emoji} horizon_{horizon} already exists for {dataset}. Nothing will be added')
                return self 
            
        # if prediction compilation doesn't exist yet
        else:
            preds   = self._compile_predictions(horizon, dataset)
            metrics = self._compile_metrics(preds)

            self.data_compilation.add_data(preds, metrics, horizon, dataset)

        return self
  
    def _setup_metric_calculator(self) -> PointRegressionMetricsCalculator | QuantileRegressionMetricsCalculator:
        """
        Iniates self.metric_calculator class
        which is either of:
        - ClassificationMetrics
        - RegressionPointPredictionMetrics
        - RegressionQuantilePredictionMetrics
        
        """
        if self.epiconfig.prediction_mode == 'classification':
            raise ValueError('no classification metric calculator found')

        if self.epiconfig.quantiles is None:
            calculator = PointRegressionMetricsCalculator(self.target_col, self.pred_cols, self.id_col, self.temporal_col)

        else:
            calculator = QuantileRegressionMetricsCalculator(self.target_col, self.pred_cols, self.id_col, self.temporal_col, self.epiconfig.quantiles)  

        return calculator              

    def _compile_predictions(self, horizon: int, dataset: Literal['train','val','test']) -> pd.DataFrame:
        """
        compile predictions of set horizon and dataset in the following form:
        _____________________________________________________
        | timestamp | node | target | model | pred-cols ... |

        """
        predictions_compilation = None

        for name, model in self.evaluated_models.items():
            model_predictions           = model.predictions.get_preds(dataset).get(horizon, is_original = False, spatially_aggregated= False)
            model_predictions['model'] = name

            if predictions_compilation is None:
                predictions_compilation= model_predictions

            else:
                predictions_compilation = pd.concat([predictions_compilation, model_predictions])

        if predictions_compilation is None:
            raise IndexError(f'predictions_compilation is invalid') 

        predictions_compilation[self.id_col] = predictions_compilation[self.id_col].astype("category")
        predictions_compilation['model']  = predictions_compilation['model'].astype("category")
       
        return predictions_compilation.reset_index(drop=True)

    def _compile_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        
        records = []
        groups = df.groupby([self.id_col, "model"], observed=True)

        for (node, model), group in tqdm(groups, total=groups.ngroups):
            y    = group[self.target_col].to_numpy()
            yhat = group[self.pred_cols].to_numpy() 

            if yhat.ndim == 2 and yhat.shape[1] == 1:
                yhat = yhat.squeeze(1)

            row  = {self.id_col: node, "model": model}             
            for metric_name in self.metric_calculator.supported_metrics:
                row[metric_name] = getattr(self.metric_calculator, metric_name)(y, yhat)
            records.append(row)

        metrics_df = pd.DataFrame(records)

        metric_cols = self.metric_calculator.supported_metrics
        metrics_df[metric_cols] = metrics_df[metric_cols].apply(
            pd.to_numeric, errors='coerce'
        )

        return metrics_df

    def _validate_epiconfigs(self) -> 'EpiConfig':
        """cross checks all prediction modes"""
        epiconfig = None
        for mlname, ml in self.evaluated_models.items():
            if epiconfig is None:
                epiconfig = ml.epiconfig
            else:
                if not epiconfig.equals(ml.epiconfig, level = 2):
                    raise ValueError('incompatible prediction modes accross models found!')
        if epiconfig is None: 
            raise ValueError('epiconfig is still None')            
                
        return epiconfig

    def __repr__(self) -> str:

        representation = [f"<{self.__class__.__name__}("]
        representation.append(f'models: {self.model_names},')
        representation.append(f'\n\tdata_compilation: {self.data_compilation}') 
        representation.append(")>")       
        return "".join(representation)