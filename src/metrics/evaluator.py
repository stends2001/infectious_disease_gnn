from typing import Union, List, Literal
from scipy.stats import spearmanr
from ..models._basemodel import BaseModel
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

            node_specific_corr = predictions_df.groupby(self.id_col).apply(self._return_spearman_corr).rename("corr").reset_index()
            node_specific_mse = predictions_df.groupby(self.id_col).apply(self._return_mse).rename("mse").reset_index()
            node_specific_rmse = predictions_df.groupby(self.id_col).apply(self._return_rmse).rename("rmse").reset_index()
            node_specific_ccc = predictions_df.groupby(self.id_col).apply(self._return_ccc).rename("ccc").reset_index()


            self.evaluation_entries[model.name][horizon_dataset] = {
                'corr'      : node_specific_corr,
                'mse'       : node_specific_mse,
                'rmse'      : node_specific_rmse,
                'ccc'       : node_specific_ccc
                }
     
        
        return self

    def plot_metric(self, metric: Literal['corr','mse','rmse','ccc'], horizon: int = 0) -> 'Evaluator':
        """ 
        Returns a violinplot of the metric chosen
        An evaluation entry should be present!

        Parameters:
        ----------
        metric: Literal['corr','mse','rmse']

        horizon: int = 0
            predictions are saved under keys `horzion_0` etc. where the number corresponds
            to the length with respect to the horizon size.
        """

        horizon_dataset     = f'horizon_{horizon}'

        rows = []
        for model_name, metrics_dict in self.evaluation_entries.items():
            # corr dataframe (node, spearman_corr)
            metric_df = metrics_dict[horizon_dataset][metric]
            for _, row in metric_df.iterrows():
                rows.append({'model': model_name, 'node': row['node'], 'metric': metric, 'value': row[metric]})
        
        df = pd.DataFrame(rows)

        fig, ax = plt.subplots(1, 1, figsize = (12,6))
        sns.violinplot(data=df, x='model', y='value', hue = 'model', palette='Blues', ax = ax)
        ax.set_title('Model Evaluation Metrics per Node')
        ax.set_ylabel(f'{metric}')    
        
        return self


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
