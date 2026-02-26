from abc import ABC, abstractmethod
import numpy as np
import pandas as pd 
from typing import Optional, List, TypeVar, cast
from scipy.stats import spearmanr, pearsonr

import inspect

class MetricsCalculatorBase(ABC):
    """
    compute metrics for dataframe, when predictions
    represent regression-quantile-predictions  

    The following (1) metrics are calculated nodewise
    - WIS

    Parameters
    ----------
    target_col: str
        ground truth column. Typically "incidence"
    pred_cols: str 
        prediction column. Typically "pred_q0" ... "pred_qQ"
    id_col: str
        geographical nodes column. Typically "node"
    temporal_col: str
        timestamps column. Typically "timestamp"
    """
    def __init__(self, 
                 target_col:    str,
                 pred_cols:     List[str],
                 id_col:        str,
                 temporal_col:  str,
                 quantiles:     Optional[List[float]] = None
                 ):
        
        self.target_col         = target_col
        self.pred_cols          = pred_cols
        self.id_col             = id_col
        self.temporal_col       = temporal_col
        self.quantiles          = quantiles

        self.supported_metrics  = self._return_supported_metrics()
        self._validate_input()

    def _validate_input(self):
        if self._requires_quantiles and self.quantiles is None:
            raise ValueError(f'_require_quantiles set to True while quantiles is None')
        
        if not self._requires_quantiles and self.quantiles is not None:
            raise ValueError(f'_require_quantiles set to False while quantiles supplied')        

    def _return_supported_metrics(self) -> List[str]:
        """return a list of metric-methods implemented"""
        methods = [
            name for name, member in inspect.getmembers(self.__class__, predicate=inspect.isfunction)
            if member.__qualname__.split(".")[0] == self.__class__.__name__
            and not name.startswith("_")
        ]   
        return methods 

    @property
    @abstractmethod
    def _requires_quantiles(self) -> bool:
        pass

    def __repr__(self) -> str:
        representation = f"<{self.__class__.__name__}(supported_metrics: {self.supported_metrics})>"
        return representation

class QuantileRegressionMetricsCalculator(MetricsCalculatorBase):  

    def wis(self, y: np.ndarray, yhats: np.ndarray) -> float:
        """
        Parameters
        ----------
        y     : (T,)      ground truth
        yhats : (T, Q)    quantile predictions, columns ordered q0 ... qQ
        """
        assert self.quantiles is not None  # narrows type for linter
        n_intervals = len(self.quantiles) // 2
        mid         = n_intervals

        components = []

        for i in range(n_intervals):
            lower  = yhats[:, i]
            upper  = yhats[:, n_intervals - 1 - i]
            alpha  = self.quantiles[i] * 2

            width          = upper - lower
            penalty_lower  = (2 / alpha) * np.maximum(lower - y, 0)
            penalty_upper  = (2 / alpha) * np.maximum(y - upper, 0)
            components.append(width + penalty_lower + penalty_upper)

        # median absolute error component
        components.append(np.abs(y - yhats[:, mid]))

        # WIS = mean over components, then mean over timestamps
        return float(np.mean(np.stack(components, axis=1)))

    @property
    def _requires_quantiles(self) -> bool:
        return True 

class PointRegressionMetricsCalculator(MetricsCalculatorBase):

    def mse(self, y, yhat):
        return float(np.mean((y - yhat) ** 2))

    def mae(self, y, yhat):
        return float(np.mean(np.abs(y - yhat)))

    def r2(self, y, yhat):
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return pd.NA
        return float(1 - ss_res / ss_tot)

    def rmse(self, y, yhat):
        return float(np.sqrt(self.mse(y, yhat)))

    def pearson(self, y, yhat):
        if np.all(y == y[0]) or np.all(yhat == yhat[0]):
            return pd.NA
        corr = cast(float, pearsonr(y, yhat).statistic)     # type: ignore
        return corr

    def spearman(self, y, yhat):
        if np.all(y == y[0]) or np.all(yhat == yhat[0]):
            return pd.NA
        corr = cast(float, spearmanr(y, yhat, nan_policy = 'omit').statistic)     # type: ignore
        return corr        

    def ccc(self, y, yhat):
        mean_y, mean_yhat = y.mean(), yhat.mean()
        var_y = y.var(ddof=1)
        var_yhat = yhat.var(ddof=1)
        if var_y == 0 or var_yhat == 0:
            return pd.NA
        cov = np.sum((y - mean_y) * (yhat - mean_yhat)) / (len(y) - 1)
        numerator = 2 * cov
        denominator = var_y + var_yhat + (mean_y - mean_yhat) ** 2
        if denominator == 0:
            return pd.NA
        return float(numerator / denominator)
    
    def smape(self, y, yhat, epsilon=1e-6):
        mask = ~((y == 0) & (yhat == 0))
        y, yhat = y[mask], yhat[mask]
        if len(y) == 0:
            return pd.NA
        y = np.where(y == 0, epsilon, y)
        yhat = np.where(yhat == 0, epsilon, yhat)
        denominator = np.maximum((np.abs(y) + np.abs(yhat)) / 2, epsilon)
        return float(np.mean(np.abs(y - yhat) / denominator) * 100)

    def mape(self, y, yhat, epsilon=1e-6):
        mask = y != 0
        y, yhat = y[mask], yhat[mask]
        if len(y) == 0:
            return pd.NA
        denominator = np.maximum(np.abs(y), epsilon)
        return float(np.mean(np.abs(y - yhat) / denominator) * 100)
    
    @property
    def pred_col(self) -> str:
        if len(self.pred_cols) > 1:
            raise ValueError('PointRegressionMetricsCalculator expects a List of only 1 prediction column')
        return self.pred_cols[0]

    @property
    def _requires_quantiles(self) -> bool:
        return False     