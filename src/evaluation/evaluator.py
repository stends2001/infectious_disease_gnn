from typing import Self, List, Literal, Dict, Any, Union, Optional
import pandas as pd
from tqdm import tqdm 

from .containers import EvaluationPredictionsCompilation
from .metrics import QuantileRegressionMetricsCalculator, PointRegressionMetricsCalculator
from .peakevaluator import PeakEvaluator

from ..utils import check_dataset, warning_emoji
from ..models.base.basemodel import BaseModel
from ..dataloading.databuilders import DataBuilder
from ..dataloading.epiconfig import EpiConfig

from .evaluationplotter import EvaluationPlotter 

from ..utils.types import DataSetSplit

import re
from matplotlib.colors import to_rgb, to_hex
import colorsys

class Evaluator:

    """
    Evaluates models

    Parameters
    ----------
    models: List[BaseModel[Any]]
        list of any models
    verbose: int
        the extent to which return output/updates. Levels are:
        - v <= 0    --> no output
        - 0 < v <=1 --> minimal output
        - v > 1     --> maximal output

    Attributes
    -------
    plotter             -> EvalutionPlotter class
    data_compilation    -> EvaluationPredictionsCompilation class
    """

    def __init__(self, models: List[BaseModel[DataBuilder]], verbose: int = 1, aggregate_seeds: bool = True):
        self.verbose            = verbose

        models_list             = models if isinstance(models, list) else [models]
        self.evaluated_models   = {ml.name: ml for ml in models_list}
        self.model_names        = list(self.evaluated_models.keys())
        self.epiconfig          = self._validate_epiconfigs()
        self.aggregate_seeds    = aggregate_seeds

        # Column names
        self.target_col     = 'target'
        self.pred_cols      = models[0].pred_cols
        self.id_col         = self.epiconfig.id_column
        self.temporal_col   = self.epiconfig.temporal_column          
                
        # Storage
        self.evaluation_entries: Dict[str, pd.DataFrame] = {}

        # Extensions
        self.plotter                = EvaluationPlotter(self)
        self.data_compilation       = EvaluationPredictionsCompilation(self.model_names)
        self.metric_calculator      = self._return_metric_calculator()      
        self.peak_evaluator         = PeakEvaluator(self)        

    @check_dataset()
    def add_evaluation(self, 
                       horizon: int  = 0,
                       dataset: DataSetSplit = 'test') -> Self:
        """
        Add evaluation entry for specified horizon.
        
        Parameters
        -----------
        horizon: int
            Prediction horizon
        dataset: Literal['train', 'val', 'test'] = 'test'
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

            if self.aggregate_seeds:
                metrics = self._aggregate_over_seeds(metrics)

            self.data_compilation.add_data(preds, metrics, horizon, dataset)

        return self
  
    def _return_metric_calculator(self) -> Union[PointRegressionMetricsCalculator, QuantileRegressionMetricsCalculator]:
        """
        Iniates self.metric_calculator class, based on epiconfig
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
        frames: List[pd.DataFrame] = []

        for name, model in self.evaluated_models.items():
            model_predictions           = model.predictions.get_preds(dataset).get(horizon, is_original = True, spatially_aggregated= False)
            model_predictions['model']  = name
            frames.append(model_predictions)
        
        if not frames:
            raise ValueError('No predictions found in evaluated_models')

        predictions_compilation = pd.concat(frames)

        predictions_compilation[self.id_col] = predictions_compilation[self.id_col].astype("category") # fine to keep node - tokens alphabetically

        # model values will be kept in the same order as user plugged them in
        predictions_compilation["model"] = pd.Categorical( 
            predictions_compilation["model"],
            categories=predictions_compilation["model"].unique(),
            ordered=True
        )
       
        return predictions_compilation.reset_index(drop=True)

    def _compile_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        compile metrics of set horizon and dataset in the following form:
        ___________________________________
        | node | model | metric-cols ... |

        """        
        records = []
        groups = df.groupby([self.id_col, "model"], observed=True) # observed => whether to drop category-combinations that don't appear in data

        if self.verbose > 1:
            iterator = tqdm(groups, desc='computing metrics nodewise', total = groups.ngroups)
        else:
            iterator = groups

        for (node, modelname), group in iterator:
            y    = group[self.target_col].to_numpy()
            yhat = group[self.pred_cols].to_numpy() 

            if yhat.ndim == 2 and yhat.shape[1] == 1:
                yhat = yhat.squeeze(1)

            row  = {self.id_col: node, "model": modelname}             
            for metric_name in self.metric_calculator.supported_metrics:
                metric_value: Optional[float]   = getattr(self.metric_calculator, metric_name)(y, yhat)
                row[metric_name]                =  pd.NA if metric_value is None else metric_value
            records.append(row)

        metrics_df  = pd.DataFrame(records)
        metric_cols = self.metric_calculator.supported_metrics
        metrics_df[metric_cols] = metrics_df[metric_cols].apply(
            pd.to_numeric, 
            errors='coerce'
        )

        return metrics_df

    def _validate_epiconfigs(self) -> 'EpiConfig':
        """cross checks all prediction modes"""
        epiconfig = None
        compared  = None

        for model_name, ml in self.evaluated_models.items():

            if epiconfig is None:
                epiconfig = ml.epiconfig

            else:
                # just compare task - level, not featurewise
                epiconfig.assert_equals(ml.epiconfig, level = 2)
                
        if epiconfig is None: 
            raise ValueError(f'No valid EpiConfig found among {list(self.evaluated_models.keys())}')            
                
        return epiconfig

    def _aggregate_over_seeds(self, metrics: pd.DataFrame) -> pd.DataFrame:
        metrics           = metrics.copy()
        unique_modelnames = metrics['model'].unique()

        mapping_dict = {
            name: re.sub(r'-s\d+$', '', name)
            for name in unique_modelnames
        }

        metrics['model'] = metrics['model'].replace(mapping_dict)
        
        metric_cols = self.metric_calculator.supported_metrics
        
        agg = metrics.groupby([self.id_col, 'model'])[metric_cols]
        
        mean_df = agg.mean().reset_index()
        
        return mean_df

    @property
    def base_model_names(self) -> list[str]:
        """Model names after seed aggregation — what actually appears in metrics."""
        
        if self.aggregate_seeds:
            seen = {}
            for name in self.model_names:
                base = re.sub(r'-s\d+$', '', name)
                seen[base] = None  # dict preserves insertion order, deduplicates
            return list(seen.keys())
        
        return list(self.evaluated_models.keys())

    @property
    def model_colors(self) -> dict[str, str]:
        """
        Returns {model_name: color} mapping, accounting for:
        - seed aggregation (base name used as key)
        - duplicate colors (alternating lighter/darker shades when models share a color)
        """

        def _adjust_lightness(hex_color: str, factor: float) -> str:
            r, g, b = to_rgb(hex_color)
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            l = max(0.0, min(1.0, l + factor))
            return to_hex(colorsys.hls_to_rgb(h, l, s))

        # alternating shade adjustments for duplicates: light, dark, lighter, darker ...
        shade_cycle = [0.12, -0.12, 0.22, -0.22]

        seen_colors = {}   # original color -> list of base_names that claimed it
        name_colors = {}   # base_name -> final color

        for ml in self.evaluated_models.values():
            base_name = re.sub(r'-s\d+$', '', ml.name) if self.aggregate_seeds else ml.name

            if base_name in name_colors:
                continue  # already resolved (multiple seeds of same base model)

            original_color = ml.model_color

            if original_color not in seen_colors:
                seen_colors[original_color] = []
                final_color = original_color  # first model keeps original
            else:
                n           = len(seen_colors[original_color])
                factor      = shade_cycle[(n - 1) % len(shade_cycle)]
                final_color = _adjust_lightness(original_color, factor)

            seen_colors[original_color].append(base_name)
            name_colors[base_name] = final_color

        return name_colors

    @property
    def seed_counts(self) -> dict[str, int]:
        """Returns {base_model_name: n_seeds} for all models."""
        counts = {}
        for name in self.model_names:
            base = re.sub(r'-s\d+$', '', name) if self.aggregate_seeds else name
            counts[base] = counts.get(base, 0) + 1
        return counts

    def __repr__(self) -> str:
        representation = [f"<{self.__class__.__name__}("]
        representation.append(f'models: {self.model_names},')
        representation.append(f'\n\tdata_compilation: {self.data_compilation}') 
        representation.append(f'\n\tmetric_calculator: {self.metric_calculator}')         
        representation.append(")>")       
        return "".join(representation)