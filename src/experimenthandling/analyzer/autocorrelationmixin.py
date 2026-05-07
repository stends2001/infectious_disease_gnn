from typing import Optional, Dict, Union, TYPE_CHECKING
import pandas as pd
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from ...dataloading.epiconfig import EpiConfig
    from ..containers import ExperimentDLMs

from ..containers import ExperimentConfig

class ExperimentAnalyzerSpatialAutocorrMixin:
    """
    Mixin class to ExperimentAnalyzer.
    Provides spatial autocorrelation (Moran's I) computation and analysis.

    Methods
    -------
    - `compute_morans_i()`
    - `summarise_morans_i()`
    - `plot_morans_i()`

    See Also
    --------
    ExperimentAnalyzer
    """

    dataloadermanagers:         Optional[Dict[Union[int, str, float], 'ExperimentDLMs']] = None
    epicfg:                     'EpiConfig'
    expcfg:                     ExperimentConfig

    def compute_morans_i(
        self,
        graph_key:  str             = 'graph2',
        varvalue:   Optional[int]   = None,
        dataset:    str             = 'all',
    ) -> pd.DataFrame:
        """
        Compute Moran's I for each timestep in the dataset.

        Parameters
        ----------
        graph_key : str
            graph to use as spatial weight matrix. Default 'graph2' (geographic adjacency).
        varvalue : Optional[int]
            variable value to use for dataorchestrator. Defaults to first available.
            For Moran's I the horizon doesn't matter — uses hl=1 by default.
        dataset : str
            'all', 'train', 'val', or 'test'. Default 'all'.

        Returns
        -------
        pd.DataFrame with columns: timestamp, morans_i
        """
        from src.dataloading.dataloaders import GraphDataLoaderManager

        # use first varvalue if not specified — horizon doesn't affect spatial structure
        if varvalue is None:
            varvalue = self.expcfg.variable_values[0]

        data_orch = self.dataloadermanagers[varvalue].baseline.dataorchestrator

        # ── build weight matrix from graph ───────────────────────────────
        dlm = (GraphDataLoaderManager(data_orch)
               .retrieve_static_graph(graph_key)
               .build())

        snapshot    = dlm.dataloader_main[0]
        edge_index  = snapshot.edge_index.numpy()
        edge_weight = (snapshot.edge_weight.numpy()
                       if snapshot.edge_weight is not None else None)
        n_nodes     = snapshot.x.shape[0]

        W = np.zeros((n_nodes, n_nodes))
        for i, (src, dst) in enumerate(edge_index.T):
            w           = float(edge_weight[i]) if edge_weight is not None else 1.0
            W[src, dst] = w
            W[dst, src] = w

        # row-normalise
        row_sums            = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W                   = W / row_sums
        W_sum               = W.sum()

        # ── get incidence data ────────────────────────────────────────────
        df           = data_orch.data_final.data_denorm.copy()
        temporal_col = data_orch.config.temporal_column
        id_col       = data_orch.config.id_column
        target_col   = data_orch.column_registration.get_by_type('target')[1]

        if dataset == 'train':
            df = df[df['train']]
        elif dataset == 'val':
            df = df[df['val']]
        elif dataset == 'test':
            df = df[df['test']]

        # ── compute Moran's I per timestep ───────────────────────────────
        records = []
        for ts in sorted(df[temporal_col].unique()):
            ts_df  = df[df[temporal_col] == ts].sort_values(id_col)
            values = ts_df[target_col].values

            if len(values) != n_nodes:
                continue

            if np.std(values) == 0:
                # all values identical — Moran's I undefined, skip
                continue

            n    = len(values)
            mean = values.mean()
            z    = values - mean
            mi   = float(n * (z @ W @ z) / (W_sum * (z @ z)))

            records.append({'timestamp': ts, 'morans_i': mi})

        return pd.DataFrame(records)

    def summarise_morans_i(
        self,
        mi_df:      pd.DataFrame,
        label:      str = '',
        print_out:  bool = True,
    ) -> dict:
        """
        Compute summary statistics of Moran's I distribution.

        Parameters
        ----------
        mi_df    : output of compute_morans_i()
        label    : label for printing
        print_out: whether to print the summary

        Returns
        -------
        dict with summary statistics
        """
        mi      = mi_df['morans_i']
        summary = {
            'label'         : label,
            'n_timesteps'   : len(mi),
            'mean'          : mi.mean(),
            'median'        : mi.median(),
            'std'           : mi.std(),
            'min'           : mi.min(),
            'max'           : mi.max(),
            'pct_above_0'   : (mi > 0).mean() * 100,
            'pct_above_01'  : (mi > 0.1).mean() * 100,
            'pct_above_03'  : (mi > 0.3).mean() * 100,
            'pct_above_05'  : (mi > 0.5).mean() * 100,
        }

        if print_out:
            print(f"\n── Moran's I: {label} {'─' * max(0, 50 - len(label))}")
            print(f"  n timesteps      : {summary['n_timesteps']}")
            print(f"  Mean             : {summary['mean']:.4f}")
            print(f"  Median           : {summary['median']:.4f}")
            print(f"  Std              : {summary['std']:.4f}")
            print(f"  Min / Max        : {summary['min']:.4f} / {summary['max']:.4f}")
            print(f"  % weeks > 0.0    : {summary['pct_above_0']:.1f}%")
            print(f"  % weeks > 0.1    : {summary['pct_above_01']:.1f}%")
            print(f"  % weeks > 0.3    : {summary['pct_above_03']:.1f}%")
            print(f"  % weeks > 0.5    : {summary['pct_above_05']:.1f}%")

        return summary

    def plot_morans_i(
        self,
        mi_df:      pd.DataFrame,
        label:      str = '',
        ax:         Optional[plt.Axes] = None,
        color:      str = '#2196F3',
    ) -> plt.Figure:
        """
        Plot Moran's I distribution over time and as histogram.

        Parameters
        ----------
        mi_df : output of compute_morans_i()
        label : label for plot title
        ax    : optional axes to plot on. If None creates new figure.
        color : line/bar color

        Returns
        -------
        plt.Figure
        """
        standalone = ax is None
        if standalone:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            ax_time, ax_hist = axes
        else:
            fig = ax.get_figure()
            ax_time = ax
            ax_hist = None

        # time series
        ax_time.plot(mi_df['timestamp'], mi_df['morans_i'],
                     color=color, linewidth=1.0, alpha=0.8)
        ax_time.axhline(0,   color='black', linewidth=0.8, linestyle='--', alpha=0.4)
        ax_time.axhline(0.1, color='orange', linewidth=0.8, linestyle=':', alpha=0.6,
                        label='0.1 threshold')
        ax_time.axhline(0.3, color='red', linewidth=0.8, linestyle=':', alpha=0.6,
                        label='0.3 threshold')
        ax_time.set_xlabel('Time', fontsize=11)
        ax_time.set_ylabel("Moran's I", fontsize=11)
        ax_time.set_title(f"Moran's I over time — {label}", fontsize=11, fontweight='bold')
        ax_time.legend(fontsize=9)
        ax_time.grid(alpha=0.3)

        # histogram
        if ax_hist is not None:
            ax_hist.hist(mi_df['morans_i'], bins=40, color=color, alpha=0.8, edgecolor='white')
            ax_hist.axvline(mi_df['morans_i'].mean(), color='black', linewidth=2,
                            linestyle='--', label=f"mean = {mi_df['morans_i'].mean():.3f}")
            ax_hist.axvline(0.1, color='orange', linewidth=1.5, linestyle=':',
                            label='0.1 threshold')
            ax_hist.axvline(0.3, color='red', linewidth=1.5, linestyle=':',
                            label='0.3 threshold')
            ax_hist.set_xlabel("Moran's I", fontsize=11)
            ax_hist.set_ylabel('Frequency', fontsize=11)
            ax_hist.set_title(f"Moran's I distribution — {label}", fontsize=11, fontweight='bold')
            ax_hist.legend(fontsize=9)
            ax_hist.grid(alpha=0.3)

        if standalone:
            plt.tight_layout()
            plt.show()

        return fig

    def full_morans_i_analysis(
        self,
        graph_key:  str = 'graph2',
        dataset:    str = 'all',
    ) -> dict:
        """
        Run full Moran's I analysis and print summary for interpretation.
        Call this and paste output here.

        Parameters
        ----------
        graph_key : graph to use as spatial weight matrix
        dataset   : 'all', 'train', 'val', or 'test'

        Returns
        -------
        dict: {varvalue: {'mi_df': pd.DataFrame, 'summary': dict}}
        """
        experiment_name = self.expcfg.experiment_name
        variable_alias  = self.expcfg.variable_alias

        print(f"\n{'='*60}")
        print(f"  MORAN'S I ANALYSIS")
        print(f"  experiment : {experiment_name}")
        print(f"  graph_key  : {graph_key}")
        print(f"  dataset    : {dataset}")
        print(f"{'='*60}")

        # use first varvalue only — spatial structure doesn't change with horizon
        varvalue = self.expcfg.variable_values[0]

        mi_df   = self.compute_morans_i(
            graph_key = graph_key,
            varvalue  = varvalue,
            dataset   = dataset,
        )

        summary = self.summarise_morans_i(
            mi_df    = mi_df,
            label    = f"{experiment_name} ({graph_key})",
            print_out= True,
        )

        self.plot_morans_i(
            mi_df = mi_df,
            label = f"{experiment_name} ({graph_key})",
            color = '#2196F3',
        )

        print(f"\n{'='*60}\n")

        return {'mi_df': mi_df, 'summary': summary}
