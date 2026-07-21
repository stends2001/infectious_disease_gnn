from abc import ABC, abstractmethod
import numpy as np
import pandas as pd 
from typing import Optional, List, TypeVar, cast, Union
from scipy.stats import spearmanr, pearsonr

import inspect

class MetricsCalculatorBase(ABC):
    """
    Parent class for MetricsCalculators.

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
        """validate quantile - requirement: uncertainty vs point - predictions"""
        if self._requires_quantiles and self.quantiles is None:
            raise ValueError(f'_require_quantiles set to True while quantiles is None')
        
        if not self._requires_quantiles and self.quantiles is not None:
            raise ValueError(f'_require_quantiles set to False while quantiles supplied')        

    def _return_supported_metrics(self) -> List[str]:
        return [
            name for name, member in inspect.getmembers(self, predicate=inspect.ismethod)
            if not name.startswith("_")
            and name not in vars(MetricsCalculatorBase)  # exclude base class non-private methods
        ]

    @property
    @abstractmethod
    def _requires_quantiles(self) -> bool:
        """each childclass must implement this boolean flag; whether or not quantiles are required"""
        pass

    def __repr__(self) -> str:
        representation = f"<{self.__class__.__name__}(supported_metrics: {self.supported_metrics})>"
        return representation

class QuantileRegressionMetricsCalculator(MetricsCalculatorBase):  
    """
    Metrics-calculator for Uncertainty predictions
    
    Includes the following (5) metrics in methods, each
    of which takes in y and yhats, and returns a float

    Methods
    -------
    - wis
    - coverage_score
    - sharpness_score
    - node_forecast_score
    - norm_wis
    """
    def wis(self, y: np.ndarray, yhats: np.ndarray) -> float:
        """
        WIS: an all-in-one metric for uncertainty - predictions

        Parameters
        ----------
        y: np.ndarray
            target - shape N; number of nodes
        yhats: np.ndarray
            predictions - shape N,Q; number of nodes, number of quantiles. 
            Quantiles must be ordered
        """        
        assert self.quantiles is not None  # narrows type for linter

        n_intervals = len(self.quantiles) // 2
        mid         = n_intervals

        components = []

        for i in range(n_intervals):
            lower  = yhats[:, i]
            upper  = yhats[:, len(self.quantiles) - 1 - i]
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
        Coverage score: fraction of timestamps where y falls within any predicted interval.
        Sharpness, or thickness of quantile (interval) is not evaluated.

        Parameters
        ----------
        y: np.ndarray
            target - shape N; number of nodes
        yhats: np.ndarray
            predictions - shape N,Q; number of nodes, number of quantiles. 
            Quantiles must be ordered        
        """
        assert self.quantiles is not None  # narrows type for linter

        n_intervals = len(self.quantiles) // 2
        inside      = np.zeros_like(y, dtype=bool)

        for i in range(n_intervals):
            lower = yhats[:, i]
            upper = yhats[:, yhats.shape[1]-1-i]
            inside |= (y >= lower) & (y <= upper)

        return float(np.mean(inside))
    
    def sharpness_score(self, y: np.ndarray, yhats: np.ndarray) -> float:
        """
        Sharpness score: thin-interval is rewarded. where higher means sharper predictions.

        Parameters
        ----------
        y: np.ndarray
            target - shape N; number of nodes
        yhats: np.ndarray
            predictions - shape N,Q; number of nodes, number of quantiles. 
            Quantiles must be ordered        
        """
        assert self.quantiles is not None  # narrows type for linter    

        n_intervals = len(self.quantiles) // 2
        widths = []

        for i in range(n_intervals):
            lower = yhats[:, i]
            upper = yhats[:, yhats.shape[1]-1-i]
            widths.append(upper - lower)

        mean_width: float = np.mean(np.stack(widths, axis=1)) # type: ignore
        
        # simple transform: smaller width → higher score  
        sharpness_score = 1 / (1 + mean_width)       
        return sharpness_score

    def node_forecast_score(self, y: np.ndarray, yhats: np.ndarray, alpha: float = 0.5) -> float:
        """
        Node Forecast score: all-in-one scalar per node; evaluated coverage and sharpness.

        Parameters
        ----------
        y: np.ndarray
            target - shape N; number of nodes
        yhats: np.ndarray
            predictions - shape N,Q; number of nodes, number of quantiles. 
            Quantiles must be ordered        
        alpha: float
            weight for coverage vs sharpness (0-1)
            this is not really a variable that can be tweeked! done here.
        """        
        cov     = self.coverage_score(y, yhats)
        sharp   = self.sharpness_score(y, yhats)
        score   = alpha * cov + (1 - alpha) * sharp
        return score

    def norm_wis(self, y: np.ndarray, yhats: np.ndarray) -> float:
        """
        Normalized wis score. Inspired by WIS, this returns a value in [0,1] where higher is better

        Parameters
        ----------
        y: np.ndarray
            target - shape N; number of nodes
        yhats: np.ndarray
            predictions - shape N,Q; number of nodes, number of quantiles. 
            Quantiles must be ordered        
        """  

        assert self.quantiles is not None  # narrows type for linter

        n_intervals = len(self.quantiles) // 2
        mid         = n_intervals

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
        score = 1 / (1 + mean_wis)  # simple monotone transform: smaller WIS → higher score

        # Step 2: adjust for coverage (optional additive factor)
        coverage = np.mean(inside_flags)  # fraction of y inside any interval
        final_score = score * coverage   # high only if WIS is low AND coverage is good

        return float(final_score)

    def cis(self, y: np.ndarray, yhats: np.ndarray) -> float:
        """
        Concordance Interval Score (CIS) — probabilistic analogue of CCC.

        CIS = CCC x coverage_score
            = rho x C_b x coverage_score

        where coverage_score = 1 - mean |empirical_coverage - nominal_coverage|
        across all interval levels.

        Parameters
        ----------
        y     : np.ndarray — targets, shape [N]
        yhats : np.ndarray — predictions, shape [N, Q]
                            quantiles must be ordered (same as wis)
                            middle index is the point/median prediction
        """
        assert self.quantiles is not None

        n_intervals = len(self.quantiles) // 2
        mid         = n_intervals

        # ── point prediction: middle quantile ─────────────────────────────
        y_pred = yhats[:, mid]

        # ── CCC decomposition ─────────────────────────────────────────────
        mu_true  = y.mean()
        mu_pred  = y_pred.mean()
        sd_true  = y.std()
        sd_pred  = y_pred.std()

        if sd_true == 0 or sd_pred == 0:
            return float('nan')

        pearson  = float(np.corrcoef(y, y_pred)[0, 1])
        nu       = sd_pred / sd_true                                # variance ratio
        u        = (mu_pred - mu_true) / np.sqrt(sd_pred * sd_true)  # normalised mean bias
        cb       = 2.0 / (nu + 1.0 / nu + u ** 2)                  # bias correction factor
        ccc      = pearson * cb

        # ── coverage score ─────────────────────────────────────────────────
        coverage_errors = []

        for i in range(n_intervals):
            lower            = yhats[:, i]
            upper            = yhats[:, len(self.quantiles) - 1 - i]
            nominal_coverage = 1.0 - self.quantiles[i] * 2   # e.g. q=0.1 → 80% interval
            empirical_coverage = float(np.mean((y >= lower) & (y <= upper)))
            coverage_errors.append(abs(empirical_coverage - nominal_coverage))

        coverage_score = max(0.0, 1.0 - float(np.mean(coverage_errors)))

        # ── CIS ────────────────────────────────────────────────────────────
        return float(ccc * coverage_score)

    @property
    def _requires_quantiles(self) -> bool:
        return True 

class PointRegressionMetricsCalculator(MetricsCalculatorBase):
    """
    Metrics-calculator for point predictions
    
    Includes the following (12) metrics in methods, each
    of which takes in y and yhats, and returns a float

    Methods
    -------
    - rmse    
    - mse
    - mae
    - smape
    - mape   
    - ccc     
    - r2
    - pearson
    - spearman
    - mbe 
    - vr
    - bcf
    """
    def mse(self, y: np.ndarray, yhat: np.ndarray) -> float:
        """simple mse"""
        return float(np.mean((y - yhat) ** 2))

    def rmse(self, y: np.ndarray, yhat: np.ndarray) -> float:
        """simple mse"""        
        return float(np.sqrt(self.mse(y, yhat)))

    def mae(self, y: np.ndarray, yhat: np.ndarray) -> float:
        """simple mae"""        
        return float(np.mean(np.abs(y - yhat)))

    def smape(self, y: np.ndarray, yhat: np.ndarray, epsilon: float=1e-6) -> Optional[float]:
        """simple smape. Returns None when target includes only zeroes"""
        mask = ~((y == 0) & (yhat == 0))

        y, yhat = y[mask], yhat[mask]
        
        if len(y) == 0:
            return None
        
        y    = np.where(y == 0, epsilon, y)
        yhat = np.where(yhat == 0, epsilon, yhat)
        denominator = np.maximum((np.abs(y) + np.abs(yhat)) / 2, epsilon)
        return float(np.mean(np.abs(y - yhat) / denominator) * 100)

    def mape(self, y: np.ndarray, yhat: np.ndarray, epsilon: float=1e-6) -> Optional[float]:
        """simple mape. Returns None when target includes only zeroes"""        
        mask = y != 0
        y, yhat = y[mask], yhat[mask]
        
        if len(y) == 0:
            return None
        
        denominator = np.maximum(np.abs(y), epsilon)
        return float(np.mean(np.abs(y - yhat) / denominator) * 100)

    def r2(self, y: np.ndarray, yhat: np.ndarray) -> Optional[float]:
        """
        Coefficient of determination: the fraction of the data's variance explained by model
        When zero variance in target: returns None
        """
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return None
        
        return float(1 - ss_res / ss_tot)

    def pearson(self, y: np.ndarray, yhat: np.ndarray) -> Optional[float]:
        """
        Pearson correlation
        When zero variance in target or zero variance in predictions, 
        returns None
        """
        if np.all(y == y[0]) or np.all(yhat == yhat[0]):
            return None 
        
        corr = cast(float, pearsonr(y, yhat).statistic)     # type: ignore
        return corr

    def spearman(self, y: np.ndarray, yhat: np.ndarray) -> Optional[float]:
        """
        Spearman correlation
        When zero variance in target or zero variance in predictions, 
        returns None
        """        
        if np.all(y == y[0]) or np.all(yhat == yhat[0]):
            return None 
        
        corr = cast(float, spearmanr(y, yhat, nan_policy = 'omit').statistic)     # type: ignore
        return corr        

    def mbe(self, y: np.ndarray, yhat: np.ndarray) -> float:
        """part of ccc: mean bias error"""
        mean_y, mean_yhat = y.mean(), yhat.mean()
        return mean_yhat - mean_y
    
    def vr(self, y: np.ndarray, yhat: np.ndarray) -> Optional[float]:
        """part of ccc: variance ratio. when variation in target or in predictions is 0, None is returned"""
        var_y, var_yhat = y.var(ddof = 1), yhat.var(ddof = 1)
        
        if var_y == 0 or var_yhat == 0:
            return None
        
        sd_y, sd_yhat = np.sqrt(var_y), np.sqrt(var_yhat)
        return sd_yhat / sd_y

    def bcf(self, y: np.ndarray, yhat: np.ndarray) -> Optional[float]:
        """
        Bias correction factor (part of CCC).
        Returns None when variation in target or predictions is 0.
        """
        vr = self.vr(y, yhat)
        if vr is None:
            return None

        mbe   = self.mbe(y, yhat)
        sd_y  = float(np.sqrt(y.var(ddof=1)))
        delta = mbe / sd_y

        return float((2 * vr) / (1 + vr**2 + delta**2))

    def ccc(self, y: np.ndarray, yhat: np.ndarray) -> Optional[float]:
        """
        part of cross concordance coefficient: a mix of pearson and a matching coefficienct 
        when variation in target or in predictions is 0, None is returned
        """            
        mean_y, mean_yhat = y.mean(), yhat.mean()
        var_y = y.var(ddof=1)
        var_yhat = yhat.var(ddof=1)
        
        if var_y == 0 or var_yhat == 0:
            return None
        
        cov = np.sum((y - mean_y) * (yhat - mean_yhat)) / (len(y) - 1)
        numerator = 2 * cov
        denominator = var_y + var_yhat + (mean_y - mean_yhat) ** 2
        
        if denominator == 0:
            return None
        
        return float(numerator / denominator)
    
    def mda(self, y: np.ndarray, yhat: np.ndarray) -> Optional[float]:
        """
        Mean Directional Accuracy.
        Fraction of timesteps where predicted direction of change 
        matches observed direction of change.
        Returns None if fewer than 2 observations.
        """
        if len(y) < 2:
            return None

        actual_dir    = np.sign(np.diff(y))
        predicted_dir = np.sign(np.diff(yhat))

        # exclude timesteps where actual direction is flat (no change)
        mask = actual_dir != 0
        if mask.sum() == 0:
            return None

        return float(np.mean(actual_dir[mask] == predicted_dir[mask]))

    @property
    def pred_col(self) -> str:
        if len(self.pred_cols) > 1:
            raise ValueError('PointRegressionMetricsCalculator expects a List of only 1 prediction column')
        
        return self.pred_cols[0]

    @property
    def _requires_quantiles(self) -> bool:
        return False     