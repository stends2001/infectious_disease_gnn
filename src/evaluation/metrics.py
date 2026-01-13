import numpy as np
import pandas as pd 
from typing import Optional, cast, Tuple, List, Union
import torch
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from pandas._libs.missing import NAType

import warnings 
from sklearn.exceptions import UndefinedMetricWarning

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

class RegressionMetrics:
    """
    compute metrics for dataframe    
    """

    def __init__(self, 
                 target_col: str    = 'incidence', 
                 pred_col: str      = 'pred', 
                 id_col: str        = 'node', 
                 temporal_col: str  = 'timestamp'):
        """
        Initialize metrics calculator (configuration only, not data).
        
        Parameters
        -----------
        target_col, pred_col, id_col, temporal_col : str
            Column names
        edge_index : torch.Tensor, optional
            Graph structure [2, num_edges]
        edge_weight : torch.Tensor, optional
            Edge weights [num_edges]
        """
        self.target_col     = target_col
        self.pred_col       = pred_col
        self.id_col         = id_col
        self.temporal_col   = temporal_col
        self.supported_metrics = ['mse','rmse','pearson_corr','spearman_corr','ccc','node_smape']

    # =============== Node-level metrics =============== #
    def mse(self, df: pd.DataFrame) -> float:
        """return mean square error"""
        return float(np.mean((df[self.target_col] - df[self.pred_col]) ** 2)) # return of np.mean is floating[any] (supports complex numbers)
    
    def rmse(self, df: pd.DataFrame) -> float:
        """Root mean squared error."""
        return np.sqrt(self.mse(df))

    def pearson_corr(self, df: pd.DataFrame) -> Union[float, NAType]:
            """Pearson correlation, with check for zero variance"""
            
            # Check variance of both columns
            if df[self.target_col].var() == 0 or df[self.pred_col].var() == 0:
                return pd.NA
            
            # Calculate Pearson correlation
            corr, _ = pearsonr(df[self.target_col], df[self.pred_col])
            
            return float(corr)  # type: ignore

    def spearman_corr(self, df: pd.DataFrame) -> Union[float, NAType]:
        """Spearman correlation, with check for zero variance"""
        
        # Check variance of both columns
        if df[self.target_col].var() == 0 or df[self.pred_col].var() == 0:
            return pd.NA
        
        # Calculate Spearman correlation
        corr, _ = spearmanr(df[self.target_col], df[self.pred_col], nan_policy='omit')
        
        return float(corr)  # type: ignore
            
    def ccc(self, df: pd.DataFrame) -> Union[float, NAType]:
        """concordance correlation coefficient."""
        target, pred = df[self.target_col], df[self.pred_col]
                
        mean_target, mean_pred  = target.mean(), pred.mean()
        var_target,  var_pred   = target.var(ddof=1), pred.var(ddof=1)

        if var_target == 0 or var_pred == 0:
            return pd.NA

        # Sample covariance (divides by n-1)
        cov = ((target - mean_target) * (pred - mean_pred)).sum() / (len(target) - 1)
        
        numerator   = 2 * cov
        denominator = var_target + var_pred + (mean_target - mean_pred) ** 2 # type: ignore

        if denominator == 0:
            return pd.NA
        else:
            return cast(float, numerator / denominator)

    def node_smape(self, df, epsilon=1e-6):
        # Mask rows where both true and predicted values are zero
        true_col = self.target_col
        pred_col  = self.pred_col
        non_zero_mask = ~(df[true_col] == 0) & ~(df[pred_col] == 0)
        df_filtered = df[non_zero_mask]

        # Replace 0's in the true or predicted values with epsilon to avoid division by zero
        true_values = df_filtered[true_col].replace(0, epsilon)
        pred_values = df_filtered[pred_col].replace(0, epsilon)

        # Compute the numerator (absolute difference)
        numerator = abs(true_values - pred_values)
        
        # Compute the denominator (mean of absolute values)
        denominator = (abs(true_values) + abs(pred_values)) / 2
        
        # Ensure the denominator isn't zero or extremely small
        denominator = np.maximum(denominator, epsilon)  # Prevent division by very small numbers

        # Calculate SMAPE as a percentage
        smape = (numerator / denominator).mean() * 100
        return smape

class ClassificationMetrics:
    """

    """
    def __init__(self, 
                 target_col: str = 'target', 
                 pred_col: str = 'pred',  # probabilities
                 pred_binary_col: Optional[str] = None,  # optional binary predictions
                 id_col: str = 'node', 
                 temporal_col: str = 'timestamp'):
        """
        pred_col : str
            Column with probability predictions (0-1)
        pred_binary_col : str, optional
            Column with binary predictions (0 or 1). If None, will be computed from pred_col
        """
        self.target_col = target_col
        self.pred_col = pred_col
        self.pred_binary_col = pred_binary_col
        self.id_col = id_col
        self.temporal_col = temporal_col
        self.supported_metrics = ['roc_auc','accuracy','precision','recall','f1','fpr','fnr']

    def _get_binary_predictions(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """"""
        if self.pred_binary_col and self.pred_binary_col in df.columns:
            return df[self.pred_binary_col]
        else:
            return (df[self.pred_col] > threshold).astype(int)

    # =============== Threshold-dependent metrics ===============
    def accuracy(self, df: pd.DataFrame, threshold: float = 0.5) -> float:
        """"""
        y_pred = self._get_binary_predictions(df, threshold)
        return float(accuracy_score(df[self.target_col], y_pred))

    def precision(self, df: pd.DataFrame, threshold: float = 0.5) -> float:
        """"""
        y_pred = self._get_binary_predictions(df, threshold)
        return float(precision_score(df[self.target_col], y_pred, zero_division=0))

    def recall(self, df: pd.DataFrame, threshold: float = 0.5) -> float:
        """"""
        y_pred = self._get_binary_predictions(df, threshold)
        return float(recall_score(df[self.target_col], y_pred, zero_division=0))

    def f1(self, df: pd.DataFrame, threshold: float = 0.5) -> float:
        """"""
        y_pred = self._get_binary_predictions(df, threshold)
        return float(f1_score(df[self.target_col], y_pred, zero_division=0))

    def fpr(self, df: pd.DataFrame, threshold: float = 0.5) -> float:
        """"""
        y_pred = self._get_binary_predictions(df, threshold)
        cm = confusion_matrix(
            df[self.target_col], y_pred, labels=[0, 1]
        )  # always 2x2
        tn, fp, fn, tp = cm.ravel()
        return float(fp / (fp + tn)) if (fp + tn) > 0 else float('nan')

    def fnr(self, df: pd.DataFrame, threshold: float = 0.5) -> float:
        """"""
        y_pred = self._get_binary_predictions(df, threshold)
        cm = confusion_matrix(
            df[self.target_col], y_pred, labels=[0, 1]
        )
        tn, fp, fn, tp = cm.ravel()
        return float(fn / (fn + tp)) if (fn + tp) > 0 else float('nan')

    def conf_matrix(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """"""
        y_pred = self._get_binary_predictions(df, threshold)
        return confusion_matrix(df[self.target_col], y_pred, labels=[0, 1])

    # =============== Threshold-independent metrics ===============
    def roc_auc(self, df: pd.DataFrame) -> float:
        """
        note that for some groups there's only a single class!
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UndefinedMetricWarning
            )
            try:
                return float(
                    roc_auc_score(df[self.target_col], df[self.pred_col])
                )
            except ValueError:
                # Single-class ground truth
                return float('nan')
        
    def average_precision(self, df: pd.DataFrame) -> float:
        """"""
        try:
            return float(average_precision_score(df[self.target_col], df[self.pred_col]))
        except ValueError:
            return float('nan')