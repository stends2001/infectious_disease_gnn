from typing import Union, List, Literal, Union
import pandas as pd
from tqdm import tqdm 

from .metrics import Metrics
from .plotter import EvaluationPlotter
from ..utils.textformatting import align
from ..models.base.basemodel import BaseModel

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
        self.evaluated_models = models if isinstance(models, list) else [models]
        self.horizon_leadtime = self._validate_leadtime()
        
        # Column names
        self.target_col = 'incidence'
        self.pred_col = 'pred'
        self.id_col = 'node'
        self.temporal_col = 'timestamp'
        
        # Storage
        self.evaluation_entries = {}
        self.evaluation_compilations = {}
        self.metrics = ['mse','rmse','spearman_corr','ccc','neighborhood_ccc','node_smape']
        # Define which metrics are spatial (need wide_df context)
        self.spatial_metrics = { 'neighborhood_ccc'}
        self.plotter = EvaluationPlotter(self)
    
    def add_evaluation(self, 
                       horizon: int = 0,
                       transformed: bool = False,
                       dataset: Literal['train', 'val', 'test'] = 'test') -> 'Evaluator':
        """
        Add evaluation entry for specified horizon.
        
        Parameters:
        -----------
        horizon : int
            Prediction horizon
        transformed : bool
            Use normalized data
        dataset : str
            Which dataset to evaluate
        """
        transformed_dataset = 'transformed' if transformed else 'nontransformed'
        horizon_dataset     = f'horizon_{horizon}'
        
        # Compile predictions from all models
        compiled_df = self._compile_predictions(dataset, transformed_dataset, horizon_dataset)
        self.evaluation_compilations[horizon_dataset] = compiled_df
        
        # Compute all metrics
        metrics_dict = {}
        for metric_name in tqdm(self.metrics, desc = 'computing metrics'):
            metric_df = self._compute_all_models_metric(
                metric_name, dataset, transformed_dataset, horizon_dataset
            )
            metrics_dict[metric_name] = metric_df
        
        self.evaluation_entries[horizon_dataset] = metrics_dict
        return self

    def _compile_predictions(self, dataset: str, 
                             transformed_dataset: str, 
                            horizon_dataset: str) -> pd.DataFrame:
        """Merge predictions from all models into single dataframe."""
        context_columns = ['timestamp', 'node']
        merged_df = None
        
        for model in self.evaluated_models:
            df = model.evaluation_datasets[dataset][transformed_dataset][horizon_dataset].copy()
            eval_df = df[context_columns + ['incidence', 'pred']]
            eval_df = eval_df.rename(columns={'pred': f'pred_{model.name}'})
            
            if merged_df is None:
                merged_df = eval_df
            else:
                merged_df = pd.merge(merged_df, eval_df, on=['timestamp', 'node', 'incidence'])

        if merged_df is None:
            raise ValueError('No merged dataframe found. Something is wrong in the evaluation datasets')
        
        return merged_df

    def _get_metrics_calculator(self, model: BaseModel) -> Metrics:
        """Create metrics calculator with model's graph structure."""
        edge_index = getattr(model.dataloader, 'edge_index', None)
        edge_weight = getattr(model.dataloader, 'edge_weight', None)
        
        return Metrics(
            target_col=self.target_col,
            pred_col=self.pred_col,
            id_col=self.id_col,
            temporal_col=self.temporal_col,
            edge_index=edge_index,
            edge_weight=edge_weight
        )
    
    def _compute_standard_metric(self, df: pd.DataFrame, metric_name: str, 
                                 model: BaseModel) -> pd.DataFrame:
        """Compute standard metric per node using groupby."""
        calculator = self._get_metrics_calculator(model)
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
    
    def _compute_spatial_metric(self, df: pd.DataFrame, metric_name: str, 
                               model: BaseModel) -> pd.DataFrame:
        """
        Compute spatial metric that needs neighbor context.
        Converts to wide format once, then processes per node.
        """
        calculator = self._get_metrics_calculator(model)
        metric_func = getattr(calculator, metric_name)
        
        wide_df = df.pivot_table(
            index=self.temporal_col,
            columns=self.id_col,
            values=self.pred_col
        ).reset_index()
        focus_col = None
        long_df = df[[self.temporal_col, self.id_col, self.pred_col, self.target_col]]
        
        # Compute metric for each node
        results = []
        unique_nodes = df[self.id_col].unique()
        
        for node in unique_nodes:
            node_df = long_df[long_df[self.id_col] == node].copy()
            
            # Call metric function with both node data and wide context
            metric_value = metric_func(node_df, wide_df, focus_col)
            results.append({self.id_col: node, metric_name: metric_value})
        
        return pd.DataFrame(results)
    
    def _compute_all_models_metric(self, metric_name: str, dataset: str, 
                                   transformed_dataset: str, horizon_dataset: str) -> pd.DataFrame:
        """Compute specified metric for all models."""
        metric_df = None
        
        for model in self.evaluated_models:
            df = model.evaluation_datasets[dataset][transformed_dataset][horizon_dataset].copy()
            
            # Route to appropriate computation method
            if metric_name in self.spatial_metrics:
                result = self._compute_spatial_metric(df, metric_name, model)
            else:
                result = self._compute_standard_metric(df, metric_name, model)
            
            result.columns.values[-1] = model.name
            
            if metric_df is None:
                metric_df = result
            else:
                metric_df = pd.merge(metric_df, result, on=[self.id_col])

        if metric_df is None:
            raise ValueError('No metric dataframe found. Something is wrong in the evaluation datasets')
                
        return metric_df
    
    def _validate_leadtime(self)-> int:
        horizon_leadtime = None
        
        for ml in self.evaluated_models:
            model_leadtime = ml.dataloader.horizon_leadtime

            if horizon_leadtime:
                if horizon_leadtime != model_leadtime:
                    raise ValueError(f'Different model horizon leadtimes found! Make sure to evaluate comparable models!')
                
            else:
                horizon_leadtime = model_leadtime

        if not horizon_leadtime:
            raise ValueError(f'No attribute horizon_leadtime found')

        return horizon_leadtime

    def __str__(self):
        all_keys = (
            ['models', 'metrics', 'evaluated datasets']
        )
        width = max(len(k) for k in all_keys) if all_keys else 20
        lines = ['<Evaluator(']
        lines.append(align('models', self.evaluated_models[0].name, width))
        for ml in self.evaluated_models[1:]:
            lines.append(align('', ml.name, width))

        lines.append('')
        lines.append(align('metrics', self.metrics, width))
        lines.append('')
        lines.append(align('evaluated datasets', list(self.evaluation_compilations.keys()), width))
        lines.append(')>')
        return '\n'.join(lines)

    def __repr__(self):
        repr = f'Evaluator of {[ml.name for ml in self.evaluated_models]} for {list(self.evaluation_compilations.keys())}'
        return repr
