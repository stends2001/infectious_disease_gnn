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

    def coverage_score(self, y: np.ndarray, yhats: np.ndarray) -> float:
        """
        Returns a single scalar: fraction of timestamps where y falls within any predicted interval.
        """
        assert self.quantiles is not None  # narrows type for linter
        n_intervals = len(self.quantiles) // 2
        inside = np.zeros_like(y, dtype=bool)

        for i in range(n_intervals):
            lower = yhats[:, i]
            upper = yhats[:, yhats.shape[1]-1-i]
            inside |= (y >= lower) & (y <= upper)

        return float(np.mean(inside))
    
    def sharpness_score(self, y: np.ndarray, yhats: np.ndarray) -> float:
        """
        Returns a scalar where higher means sharper predictions.
        """
        assert self.quantiles is not None  # narrows type for linter
        n_intervals = len(self.quantiles) // 2
        widths = []

        for i in range(n_intervals):
            lower = yhats[:, i]
            upper = yhats[:, yhats.shape[1]-1-i]
            widths.append(upper - lower)

        mean_width: float = np.mean(np.stack(widths, axis=1))
        return 1 / (1 + mean_width)  # simple transform: smaller width → higher score    

    def node_forecast_score(self, y: np.ndarray, yhats: np.ndarray, alpha=0.5) -> float:
        """
        Combines coverage and sharpness into one scalar per node.
        alpha: weight for coverage vs sharpness (0..1)
        """
        cov = self.coverage_score(y, yhats)
        sharp = self.sharpness_score(y, yhats)
        return alpha * cov + (1 - alpha) * sharp

    def norm_wis(self, y: np.ndarray, yhats: np.ndarray) -> float:
        """
        Returns a single scalar per node summarizing the quality of quantile forecasts.
        Inspired by WIS, normalized to [0,1], higher is better.

        Parameters
        ----------
        y        : (T,)      ground truth
        yhats    : (T, Q)    quantile predictions, columns ordered q0 ... qQ
        quantiles: (Q,)      list of quantiles corresponding to yhats
        """

        assert self.quantiles is not None  # narrows type for linter
        n_intervals = len(self.quantiles) // 2
        mid = n_intervals

        # Components: interval widths and penalties
        widths = []
        penalties = []
        inside_flags = np.zeros_like(y, dtype=bool)

        for i in range(n_intervals):
            lower = yhats[:, i]
            upper = yhats[:, len(self.quantiles)-1-i]
            alpha = self.quantiles[i] * 2

            width = upper - lower
            penalty_lower = (2 / alpha) * np.maximum(lower - y, 0)
            penalty_upper = (2 / alpha) * np.maximum(y - upper, 0)

            widths.append(width)
            penalties.append(penalty_lower + penalty_upper)

            # For coverage check
            inside_flags |= (y >= lower) & (y <= upper)

        # Median absolute error
        median_error = np.abs(y - yhats[:, mid])

        # Combine components like WIS
        wis_like = np.mean(np.stack(widths + penalties + [median_error], axis=1), axis=1)  # per timestamp

        # Convert to a 0-1 score
        # Step 1: scale by max possible value (or 1+mean width for normalization)
        mean_wis = np.mean(wis_like)
        mean_width = np.mean(np.stack(widths, axis=1))
        score = 1 / (1 + mean_wis)  # simple monotone transform: smaller WIS → higher score

        # Step 2: adjust for coverage (optional additive factor)
        coverage = np.mean(inside_flags)  # fraction of y inside any interval
        final_score = score * coverage   # high only if WIS is low AND coverage is good

        return float(final_score)

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

    def mbe(self, y, yhat):
        """part of ccc: mean bias error"""
        mean_y, mean_yhat = y.mean(), yhat.mean()
        return mean_yhat - mean_y
    
    def vr(self, y, yhat):
        """part of ccc: variance ratio"""
        var_y, var_yhat = y.var(ddof = 1), yhat.var(ddof = 1)
        if var_y == 0 or var_yhat == 0:
            return pd.NA 
        sd_y, sd_yhat = np.sqrt(var_y), np.sqrt(var_yhat)
        return sd_yhat / sd_y

    def bcf(self, y, yhat):
        """part of ccc: bias correction factor"""    
        mbe             = self.mbe(y, yhat)

        var_y, var_yhat = y.var(ddof = 1), yhat.var(ddof = 1)
        if var_y == 0 or var_yhat == 0:
            return pd.NA 
        sd_y, sd_yhat = np.sqrt(var_y), np.sqrt(var_yhat)
        variance_ratio =  sd_yhat / sd_y

        delta = mbe / sd_y
        cb = (2 * variance_ratio) / (1 + variance_ratio**2 + delta**2)    
        return cb

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