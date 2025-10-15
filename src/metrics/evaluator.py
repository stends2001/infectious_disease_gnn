from typing import Union, List, Literal
from scipy.stats import spearmanr
from ..new_models_module.base.basemodel import BaseModel
import seaborn as sns
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd



class Evaluator:

    """" 
    Evaluates model predictions against ground truth

    Examples:
    --------
    >>> eval = Evaluator([ml2_bn, ml2_id])
    >>> eval.add_evaluation()
    >>> eval.plot_metric(metric = 'corr');
    
    """



    def __init__(self, models: List[BaseModel]):

        self.evaluated_models = models

        self.target_col       = 'incidence'
        self.pred_col         = 'pred'
        self.id_col           = 'node'

        self.evaluation_entries = {}

        self.metrics = {
        'corr': self._return_spearman_corr,
        'mse': self._return_mse,
        'rmse': self._return_rmse,
        'ccc': self._return_ccc}

        for ml in models:
            self.evaluation_entries[ml.name] = {}


    def add_evaluation(self, 
                       horizon: int = 0,
                       transformed: bool = False,
                       dataset: Literal['train','val','test'] = 'test') -> 'Evaluator':
        
        """ 
        Adds an evaluation entry, specific to the horizon specified. This entry
        is added to `self.evaluation_entries`.
        Evaluation entries are currently limited to:
        - corr => pearson correlation value per node
        - mse  => mse value per node
        - rmse => rmse value per node

        Parameters:
        ----------
        horizon: int = 0
            predictions are saved under keys `horzion_0` etc. where the number corresponds
            to the length with respect to the horizon size.
        transformed: bool = False
            whether to use the normalized (`transformed = True`) or the non-normalized
            (`transformed = False`) incidence rates.
        dataset: Literal['train','val','test'] = 'test' 
            which dataset to use
        """
        
        transformed_dataset = 'transformed' if transformed else 'nontransformed'
        horizon_dataset     = f'horizon_{horizon}'        
        
        for model in self.evaluated_models:

            predictions_df = model.evaluation_datasets[dataset][transformed_dataset][horizon_dataset].copy()

            self.evaluation_entries[model.name][horizon_dataset] = {}
            for metric_name, func in self.metrics.items():
                
                self.evaluation_entries[model.name][horizon_dataset][metric_name] = self._compute_metric(predictions_df, func, metric_name)
        
        return self

    def plot_metric(self, metric: Literal['corr','mse','rmse','ccc'], horizon: int = 0, plot_type: Literal['violin', 'box'] = 'violin') -> 'Evaluator':
        """ 
        Returns a violinplot of the metric chosen
        An evaluation entry should be present!
        """

        horizon_dataset = f'horizon_{horizon}'

        rows = []
        model_colors = {}

        for model in self.evaluated_models:
            model_name = model.name
            color = model.model_color  # <- assumes each model has a `.model_color` attribute (e.g., an RGB tuple)
            model_colors[model_name] = color

            metrics_dict = self.evaluation_entries[model_name]
            metric_df = metrics_dict[horizon_dataset][metric]

            for _, row in metric_df.iterrows():
                rows.append({
                    'model': model_name,
                    'node': row['node'],
                    'metric': metric,
                    'value': row[metric]
                })

        df = pd.DataFrame(rows)

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
       
        if plot_type == 'violin':
            sns.violinplot(
                data=df,
                x='model',
                y='value',
                hue='model',
                ax=ax,
                palette=model_colors,
                cut=0  # optional: limits violin tails to data range
            )
        elif plot_type == 'box':
            sns.boxplot(
                data=df,
                x='model',
                y='value',
                hue='model',
                ax=ax,
                palette=model_colors
            )
        else:
            raise ValueError("plot_type must be either 'violin' or 'box'")

        ax.set_title('Model Evaluation Metrics per Node')
        ax.set_ylabel(f'{metric}')

        return self

    def _compute_metric(self, predictions_df: pd.DataFrame, metric_func, metric_name: str) -> pd.DataFrame:
        return (
            predictions_df
            .drop(columns=[self.id_col])
            .groupby(predictions_df[self.id_col])
            .apply(metric_func)
            .rename(metric_name)
            .reset_index()
        )

    def _return_spearman_corr(self, df: pd.DataFrame):
        target = df[self.target_col]
        pred = df[self.pred_col]
        if target.nunique() < 2 or pred.nunique() < 2:
            return pd.NA
        corr, _ = spearmanr(target, pred, nan_policy='omit')
        return corr

    def _return_mse(self, df: pd.DataFrame):
        target = df[self.target_col]
        pred = df[self.pred_col]
        errors = target - pred
        mse = np.mean(errors ** 2)
        return mse

    def _return_rmse(self, df: pd.DataFrame):
        target = df[self.target_col]
        pred = df[self.pred_col]
        errors = target - pred
        rmse = np.sqrt(np.mean(errors ** 2))
        return rmse

    def _return_ccc(self, df: pd.DataFrame):
        target = df[self.target_col]
        pred = df[self.pred_col]

        mean_target = target.mean()
        mean_pred   = pred.mean()

        var_target = target.var(ddof=0)
        var_pred = pred.var(ddof=0)


        cov = ((target - mean_target) * (pred - mean_pred)).mean()

        numerator = 2 * cov
        denominator = var_target + var_pred + (mean_target - mean_pred) ** 2 # type: ignore

        if denominator == 0:
            return pd.NA  # Handle degenerate case

        ccc = numerator / denominator
        return ccc
