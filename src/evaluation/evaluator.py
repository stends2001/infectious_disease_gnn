from typing import Union, List, Literal
from scipy.stats import spearmanr
from ..models.base.basemodel import BaseModel
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

    def __init__(self, models: List[BaseModel], horizon_leadtime: int):

        self.evaluated_models = models

        self.target_col       = 'incidence'
        self.pred_col         = 'pred'
        self.id_col           = 'node'

        self.evaluation_entries = {}

        self.metrics = {
        'corr': self._return_spearman_corr,
        'mse': self._return_mse,
        'rmse': self._return_rmse,
        'ccc': self._return_ccc,
        'lag_corr': self._return_lag_correlation  # <- NEW
        }

        self.horizon_leadtime = horizon_leadtime

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

    def plot_metric(self, metric: Literal['corr','mse','rmse','ccc','lag_corr'], horizon: int = 0, plot_type: Literal['violin', 'box'] = 'violin') -> 'Evaluator':
        """ 
        Returns a violinplot of the metric chosen
        An evaluation entry should be present!
        """

        horizon_dataset = f'horizon_{horizon}'

        rows = []
        model_class_colors = {}

        for model in self.evaluated_models:
            model_name = model.name
            model_class = model.model_class
            color = model.model_color
            
            # Map model_class to color (will handle duplicates automatically)
            if model_class not in model_class_colors:
                model_class_colors[model_class] = color

            metrics_dict = self.evaluation_entries[model_name]
            metric_df = metrics_dict[horizon_dataset][metric]

            for _, row in metric_df.iterrows():
                rows.append({
                    'model': model_name,
                    'model_class': model_class,
                    'node': row['node'],
                    'metric': metric,
                    'value': row[metric]
                })

        df = pd.DataFrame(rows)

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
        if plot_type == 'violin':
            sns.violinplot(
                data=df,
                x='model',
                y='value',
                hue='model_class',
                ax=ax,
                palette=model_class_colors,
                cut=0,
                legend=False
            )
        elif plot_type == 'box':
            sns.boxplot(
                data=df,
                x='model',
                y='value',
                hue='model_class',
                ax=ax,
                palette=model_class_colors,
                legend=False
            )
        else:
            raise ValueError("plot_type must be either 'violin' or 'box'")

        ax.set_title('Model Evaluation Metrics per Node')
        ax.set_ylabel(f'{metric}')
        ax.set_xlabel('Model')
        ax.grid()
        
        # Rotate x-axis labels 90 degrees
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='center', fontsize=8)
        
        # Create custom legend with model_class colors
        handles = [plt.Rectangle((0,0),1,1, facecolor=color) for color in model_class_colors.values()] # type: ignore
        labels = list(model_class_colors.keys())
        ax.legend(handles, labels, title='Model Class', loc='best')
        
        plt.tight_layout()  # Prevents label cutoff

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
        """
        ccc => concordance correlation coefficient;
        
        
        """
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

    def _return_lag_correlation(self, df: pd.DataFrame):
        """
        Compute correlation between predictions and lagged ground truth.
        High correlation suggests the model is just memorizing recent values
        rather than truly forecasting.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with target and pred columns
        lag : int
            Which lag to check (typically 1 or matching your horizon_leadtime)
            
        Returns:
        --------
        float : Correlation coefficient
        """
        pred = df[self.pred_col]
        target = df[self.target_col]
        
        lag = self.horizon_leadtime

        # Shift target by lag
        lagged_target = target.shift(lag)
        
        # Remove NaN values from shifting
        valid_mask = ~lagged_target.isna() & ~pred.isna()
        
        if valid_mask.sum() < 2:
            return pd.NA
            
        valid_pred = pred[valid_mask]
        valid_lagged = lagged_target[valid_mask]
        
        if valid_pred.nunique() < 2 or valid_lagged.nunique() < 2:
            return pd.NA
        
        corr, _ = spearmanr(valid_pred, valid_lagged, nan_policy='omit')
        return corr
    
    def __repr__(self) -> str:
        return f"Evaluator of {[ml.name for ml in self.evaluated_models]}"