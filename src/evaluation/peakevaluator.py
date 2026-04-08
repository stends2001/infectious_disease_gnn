from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


class PeakEvaluator:
    """
    Evaluates model performance specifically around epidemic peak periods.

    Motivation
    ----------
    Aggregate metrics (MAE, CCC, etc.) computed over all weeks are dominated
    by the many low-incidence, high-autocorrelation weeks where a simple
    persistence baseline already does well. Graph structure is most likely
    to add value at *change points* — the weeks around a node's peak, where
    self-history alone is insufficient and neighbour signal (nodes that peaked
    earlier) can improve the forecast.

    This class slices the compiled predictions around per-node peaks and
    recomputes all metrics on those slices, letting you compare graph types
    specifically where it should matter most.

    Parameters
    ----------
    evaluator : Evaluator
        A fitted Evaluator instance with at least one horizon added via
        add_evaluation(). Predictions are read from evaluator.data_compilation.
    peak_method : {'raw', 'smoothed'}
        How to locate the peak week per node in the *target* series.
        - 'raw'      : argmax of the raw target values (fast, noisy for some nodes)
        - 'smoothed' : argmax after applying a rolling mean of `smooth_window`
                       weeks (recommended for weekly influenza data)
    window_half_width : int
        Number of weeks on each side of the peak to include in the slice.
        window_half_width=2 means weeks [peak-2, peak-1, peak, peak+1, peak+2]
        are included (5 weeks total). Default 2.
    smooth_window : int
        Width of the uniform smoothing kernel used when peak_method='smoothed'.
        Default 3 (3-week rolling mean). Ignored when peak_method='raw'.
    dataset : str
        Which dataset split to evaluate. Default 'test'.

    Usage
    -----
        pe = PeakEvaluator(evaluator, peak_method='smoothed', window_half_width=2)
        peak_metrics = pe.compute(horizon=0)   # per-node, per-model metrics around peak
        summary      = pe.summary(horizon=0)   # model-level aggregation
        rolling      = pe.rolling_window(horizon=0, step_weeks=1)  # Witzke-style rolling eval
    """

    def __init__(self,
                 evaluator,
                 peak_method:       Literal['raw', 'smoothed'] = 'smoothed',
                 window_half_width: int  = 2,
                 smooth_window:     int  = 3,
                 dataset:           str  = 'test'):

        self.evaluator          = evaluator
        self.peak_method        = peak_method
        self.window_half_width  = window_half_width
        self.smooth_window      = smooth_window
        self.dataset            = dataset

        self.id_col             = evaluator.id_col
        self.temporal_col       = evaluator.temporal_col
        self.target_col         = evaluator.target_col
        self.pred_cols          = evaluator.pred_cols
        self.metric_calculator  = evaluator.metric_calculator

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def compute(self, horizon: int = 0) -> pd.DataFrame:
        """
        Compute per-node, per-model metrics on the peak window slice.

        Returns
        -------
        pd.DataFrame with columns:
            [id_col, 'model', 'peak_date', 'n_weeks_in_window', *metric_cols]
        """
        preds   = self._get_predictions(horizon)
        peaks   = self._find_peaks(preds)           # {node -> peak_timestamp}
        sliced  = self._slice_around_peaks(preds, peaks)
        metrics = self._compute_metrics(sliced, peaks)
        return metrics

    def summary(self, horizon: int = 0) -> pd.DataFrame:
        """
        Model-level summary: mean of each metric across all nodes.

        Returns
        -------
        pd.DataFrame indexed by model, columns = metric names.
        """
        node_metrics = self.compute(horizon)
        metric_cols  = self.metric_calculator.supported_metrics
        available    = [c for c in metric_cols if c in node_metrics.columns]
        return (
            node_metrics
            .groupby('model', observed=True)[available]
            .mean()
            .round(4)
        )

    def rolling_window(self,
                       horizon:     int = 0,
                       step_weeks:  int = 1) -> pd.DataFrame:
        """
        Witzke et al.-style rolling evaluation: for each rolling window of
        calendar weeks, collect the nodes whose peak falls inside that window
        and compute metrics on those nodes using predictions from their peak week.

        This shows whether models with richer graph structure improve *earlier*
        (i.e. for nodes that peak early) compared to models with weaker graphs.

        Parameters
        ----------
        horizon : int
            Forecast horizon to evaluate.
        step_weeks : int
            Step size of the rolling window in weeks. Default 1.

        Returns
        -------
        pd.DataFrame with columns:
            ['window_start', 'window_end', 'model', 'n_nodes', *metric_cols]
        """
        preds   = self._get_predictions(horizon)
        peaks   = self._find_peaks(preds)
        sliced  = self._slice_around_peaks(preds, peaks)

        all_peak_dates  = sorted(peaks.values())
        timestamps      = sorted(pd.Series(all_peak_dates).unique())

        if len(timestamps) < 2:
            raise ValueError("Not enough distinct peak dates to build a rolling window.")

        window_size = max(2 * self.window_half_width + 1, 3)   # at least 3 weeks wide
        records     = []

        for i in range(0, len(timestamps) - window_size + 1, step_weeks):
            win_start   = timestamps[i]
            win_end     = timestamps[min(i + window_size - 1, len(timestamps) - 1)]

            # nodes whose peak falls in this window
            nodes_in_window = {
                node for node, pk in peaks.items()
                if win_start <= pk <= win_end
            }

            if not nodes_in_window:
                continue

            win_slice = sliced[sliced[self.id_col].isin(nodes_in_window)]

            # compute metrics per model for this window
            for model_name, model_group in win_slice.groupby('model', observed=True):
                row = {
                    'window_start': win_start,
                    'window_end':   win_end,
                    'model':        model_name,
                    'n_nodes':      len(nodes_in_window),
                }
                for metric_name in self.metric_calculator.supported_metrics:
                    y    = model_group[self.target_col].to_numpy()
                    yhat = model_group[self.pred_cols].to_numpy()
                    if yhat.ndim == 2 and yhat.shape[1] == 1:
                        yhat = yhat.squeeze(1)
                    val = getattr(self.metric_calculator, metric_name)(y, yhat)
                    row[metric_name] = val
                records.append(row)

        return pd.DataFrame(records)

    def peak_table(self, horizon: int = 0) -> pd.DataFrame:
        """
        Returns a flat table of every node's peak date and incidence value.
        Useful for sanity-checking that peaks look epidemiologically plausible
        before running the full evaluation.

        Returns
        -------
        pd.DataFrame with columns: [id_col, 'peak_date', 'peak_incidence']
        sorted by peak_date.
        """
        preds = self._get_predictions(horizon)
        peaks = self._find_peaks(preds)

        # get the target value at the peak week for each node
        records = []
        for node, peak_date in peaks.items():
            node_data   = preds[
                (preds[self.id_col] == node) &
                (preds[self.temporal_col] == peak_date) &
                (preds['model'] == preds['model'].cat.categories[0])   # any model; target is same
            ]
            peak_val = node_data[self.target_col].values[0] if len(node_data) else np.nan
            records.append({self.id_col: node, 'peak_date': peak_date, 'peak_incidence': peak_val})

        return (
            pd.DataFrame(records)
            .sort_values('peak_date')
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_predictions(self, horizon: int) -> pd.DataFrame:
        """Retrieve the compiled per-timestamp predictions from the evaluator."""
        return self.evaluator.data_compilation.get_data(horizon, self.dataset)['predictions']

    def _find_peaks(self, preds: pd.DataFrame) -> Dict[str, pd.Timestamp]:
        """
        Find the peak timestamp per node using the target column.
        Uses only one model's rows (target is identical across models).

        Returns
        -------
        Dict mapping node identifier -> peak pd.Timestamp
        """
        # use the first model to avoid duplicate target rows
        first_model = preds['model'].cat.categories[0]
        target_df   = (
            preds[preds['model'] == first_model]
            [[self.id_col, self.temporal_col, self.target_col]]
            .copy()
        )
        target_df[self.temporal_col] = pd.to_datetime(target_df[self.temporal_col])
        target_df = target_df.sort_values([self.id_col, self.temporal_col])

        peaks = {}
        for node, group in target_df.groupby(self.id_col, observed=True):
            values = group[self.target_col].to_numpy()

            if self.peak_method == 'smoothed' and len(values) >= self.smooth_window:
                # uniform_filter1d: fast causal-ish smoothing; mode='nearest' pads edges
                values = uniform_filter1d(values.astype(float), size=self.smooth_window, mode='nearest')

            peak_idx    = int(np.argmax(values))
            peak_date   = group[self.temporal_col].iloc[peak_idx]
            peaks[node] = peak_date

        return peaks

    def _slice_around_peaks(self,
                             preds: pd.DataFrame,
                             peaks: Dict) -> pd.DataFrame:
        """
        For each node, keep only rows within window_half_width weeks of its peak.

        Returns the filtered predictions DataFrame (all models).
        """
        preds = preds.copy()
        preds[self.temporal_col] = pd.to_datetime(preds[self.temporal_col])
        week_delta = pd.Timedelta(weeks=self.window_half_width)

        masks = []
        for node, peak_date in peaks.items():
            mask = (
                (preds[self.id_col] == node) &
                (preds[self.temporal_col] >= peak_date - week_delta) &
                (preds[self.temporal_col] <= peak_date + week_delta)
            )
            masks.append(mask)

        if not masks:
            raise ValueError("No nodes found when slicing around peaks.")

        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask = combined_mask | m

        return preds[combined_mask].reset_index(drop=True)

    def _compute_metrics(self,
                          sliced: pd.DataFrame,
                          peaks:  Dict) -> pd.DataFrame:
        """
        Compute all supported metrics per node per model on the sliced DataFrame.
        """
        records = []
        groups  = sliced.groupby([self.id_col, 'model'], observed=True)

        for (node, model_name), group in groups:
            y    = group[self.target_col].to_numpy()
            yhat = group[self.pred_cols].to_numpy()
            if yhat.ndim == 2 and yhat.shape[1] == 1:
                yhat = yhat.squeeze(1)

            row = {
                self.id_col:        node,
                'model':            model_name,
                'peak_date':        peaks.get(node),
                'n_weeks_in_window': len(group),
            }
            for metric_name in self.metric_calculator.supported_metrics:
                val         = getattr(self.metric_calculator, metric_name)(y, yhat)
                row[metric_name] = val
            records.append(row)

        return pd.DataFrame(records)