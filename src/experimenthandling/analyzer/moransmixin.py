from typing import Literal 
import numpy as np 
import pandas as pd

class MoransAnalysisMixin:
    """"
    Mixinclass to ExperimentAnalyzer that deals with Moran's analysis. 

    Methods
    -------
    - `global_morans_i_timeseries()`
    - `local_morans_i_timeseries()` 
    """

    def _morans_i(self, x: np.ndarray, W: np.ndarray) -> float:
        """
        caluclate a single Moran's I globally:

        I = \frac{N}{S_0} \frac{\sum_n\sum_jw_{nj}(x_n-\bar{x})(x_j-\bar{x})}{\sum_n(x_n - \bar{x})^2}
        """
        x = np.asarray(x, dtype=float)

        x_mean      = x.mean()
        x_centered  = x - x_mean

        N   = len(x)
        S0  = W.sum()

        numerator   = x_centered @ W @ x_centered
        denominator = x_centered @ x_centered

        # safe guards
        if S0 == 0 or denominator == 0:
            return np.nan

        return (N / S0) * (numerator / denominator)

    def _local_morans_i(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """ 
        calculate a single Moran's I locally
        """
        x = np.asarray(x, dtype=float)

        x_centered  = x - x.mean()
        N           = len(x)

        m2 = (x_centered @ x_centered) / N

        if m2 == 0:
            return np.full(N, np.nan)

        spatial_lag = W @ x_centered

        return (x_centered * spatial_lag) / m2

    def global_morans_i_timeseries(self, df: pd.DataFrame, W: np.ndarray, feature: Literal['incidence','residual'] = 'incidence') -> pd.DataFrame:
        """
        """
        results = []

        for ts, group in df.groupby("timestamp"):

            # ensure node order matches adjacency matrix
            x = (
                group
                .sort_values("node")
                [feature]
                .to_numpy()
            )

            I = self._morans_i(x, W)

            results.append({
                "timestamp": ts,
                "global_morans_i": I
            })

        return pd.DataFrame(results).sort_values("timestamp")

    def local_morans_i_timeseries(self, df: pd.DataFrame, W: np.ndarray, feature: Literal['incidence','residual'] = 'incidence'):

        results = []

        for ts, group in df.groupby("timestamp"):

            group = group.sort_values("node")

            x = group[feature].to_numpy()

            I_local = self._local_morans_i(x, W)

            group = group.copy()
            group["local_morans_i"] = I_local

            results.append(group)

        return pd.concat(results, ignore_index=True)