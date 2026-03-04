import torch
import pandas as pd
from scipy.stats import pearsonr

def lagged_spatial_correlation(df, edge_index, target_col, id_col, time_col, lag=1):
    """
    For each edge (i, j), correlate node i at time t with node j at time t+lag.
    Compare to temporal autocorrelation of node i with itself at t+lag.
    If spatial > temporal, diffusion is real.
    """
    results = []
    nodes_i = edge_index[0].tolist()
    nodes_j = edge_index[1].tolist()

    for src, dst in zip(nodes_i, nodes_j):
        ts_src = df[df[id_col] == src].sort_values(time_col)[target_col].values
        ts_dst = df[df[id_col] == dst].sort_values(time_col)[target_col].values

        min_len = min(len(ts_src), len(ts_dst)) - lag
        if min_len < 10:  # too short to be meaningful
            continue

        spatial_corr, _ = pearsonr(ts_src[:min_len], ts_dst[lag:lag+min_len])
        self_corr, _    = pearsonr(ts_src[:min_len], ts_src[lag:lag+min_len])

        results.append({
            'src': src, 'dst': dst,
            'spatial_corr': spatial_corr,
            'self_corr':    self_corr,
            'delta':        spatial_corr - self_corr  # positive = neighbour more predictive than self
        })

    return pd.DataFrame(results)