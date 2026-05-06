from pathlib import Path 
from typing import List, Union, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from .loader import ExperimentLoader
from .handler import ExperimentHandler
from .containers import ExperimentConfig
from ..utils.helpers import get_project_utilities_env
from ..models.base.basemodel import BaseModel
from ..evaluation import Evaluator
from ..models import PersistenceModel, ClimateologyModel

class ExperimentAnalyzerPlotterMixin:
    """
    Mixin class to ExperimentAnalyzer;
    simply a library, if you will, of functions.

    Methods
    -------
    - `plot_metric_over_reference()`
    - `plot_graph_advantage()`

    See Also
    --------
    For more information, see ExperimentAnalyzer

    TODO
    need to restructure some functionality here, particularly using model colors.
    """
    metrics_df:         pd.DataFrame
    experiment_config:  ExperimentConfig

    def plot_metric_over_reference(
        self,
        metric:         str                 = 'ccc',
        reference:      str                 = 'persistence',
        higher_is_better: bool              = True,
        title:          str                 = None,
    ) -> plt.Figure:
        """
        ...
        """

        variable_alias  = self.experiment_config.variable_alias
        varvalues       = self.experiment_config.variable_values

        model_config = {
                'persistence'   : {'label': 'Persistence',     'color': '#9E9E9E', 'ls': '--', 'lw': 1.5},
                'climateology'  : {'label': 'Climatology',     'color': '#9C27B0', 'ls': '--', 'lw': 1.5},
                'lstm'          : {'label': 'LSTM',             'color': '#4CAF50', 'ls': '-',  'lw': 2.0},
                'gcn2_graph1'   : {'label': 'GCN2 (identity)', 'color': '#BBDEFB', 'ls': '-',  'lw': 2.0},
                'gcn2_graph2'   : {'label': 'GCN2 (geo)',      'color': '#2196F3', 'ls': '-',  'lw': 2.0},
                'gcn2_graph3'   : {'label': 'GCN2 (random)',   'color': '#FF9800', 'ls': '-',  'lw': 2.0},
                'gcn2_graph4'   : {'label': 'GCN2 (commuter)', 'color': '#F44336', 'ls': '-',  'lw': 2.0},
            }

        # ── build model_key if not present ───────────────────────────────────
        df = self.metrics_df.copy()
        if 'model_key' not in df.columns:
            df['model_key'] = df.apply(
                lambda r: f"{r['model_type']}_{r['graph']}"
                if pd.notna(r.get('graph')) and r.get('graph') is not None
                else r['model_type'],
                axis=1
            )

        # ── compute reference mean per horizon ────────────────────────────────
        ref_mean = (df[df['model_key'] == reference]
                    .groupby(variable_alias)[metric]
                    .mean()
                    .rename('ref_mean'))

        if ref_mean.empty:
            raise ValueError(f"Reference model '{reference}' not found in metrics_df. "
                            f"Available model_keys: {df['model_key'].unique().tolist()}")

        df = df.merge(ref_mean, on=variable_alias)

        if higher_is_better:
            df['delta'] = df[metric] - df['ref_mean']
        else:
            df['delta'] = df['ref_mean'] - df[metric]

        # ── aggregate: nodes → seeds → summary ───────────────────────────────
        # step 1: mean across nodes per (horizon, model_key, seed)
        per_seed = (df.groupby([variable_alias, 'model_key', 'seed'])
                    [[metric, 'delta']]
                    .mean()
                    .reset_index())

        # step 2: mean/std across seeds
        summary = (per_seed
                .groupby([variable_alias, 'model_key'])
                .agg(
                    metric_mean  = (metric,   'mean'),
                    metric_std   = (metric,   'std'),
                    delta_mean   = ('delta',  'mean'),
                    delta_std    = ('delta',  'std'),
                )
                .reset_index())

        # ── labels ───────────────────────────────────────────────────────────
        metric_labels = {
            'ccc'    : 'CCC',
            'r2'     : 'R²',
            'rmse'   : 'RMSE',
            'mae'    : 'MAE',
            'pearson': 'Pearson r',
            'mda'    : 'MDA',
            'mbe'    : 'MBE',
        }
        metric_label = metric_labels.get(metric, metric.upper())

        ref_label = model_config.get(reference, {}).get('label', reference)
        direction = 'improvement over' if higher_is_better else 'reduction vs'
        delta_label = f'Δ{metric_label} ({direction} {ref_label})'

        auto_title = (f'{metric_label} — absolute and relative to {ref_label}')

        # ── plot ──────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax, y_mean, y_std, ylabel, panel_title in zip(
            axes,
            ['metric_mean', 'delta_mean'],
            ['metric_std',  'delta_std'],
            [metric_label,  delta_label],
            [f'Absolute {metric_label}',
            f'{metric_label} relative to {ref_label}'],
        ):
            for model_key, cfg in model_config.items():
                sub = summary[summary['model_key'] == model_key].sort_values(variable_alias)
                if sub.empty:
                    continue

                ax.plot(sub[variable_alias], sub[y_mean],
                        label     = cfg['label'],
                        color     = cfg['color'],
                        linestyle = cfg['ls'],
                        linewidth = cfg['lw'],
                        marker    = 'o',
                        markersize= 5)

                ax.fill_between(
                    sub[variable_alias],
                    sub[y_mean] - sub[y_std],
                    sub[y_mean] + sub[y_std],
                    color = cfg['color'],
                    alpha = 0.10,
                )

            # reference line on delta panel
            if y_mean == 'delta_mean':
                ax.axhline(0, color='black', linewidth=0.8,
                        linestyle=':', alpha=0.6, label=f'{ref_label} (reference)')

            ax.set_xlabel('Forecast horizon (weeks ahead)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(panel_title, fontsize=12, fontweight='bold')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.legend(fontsize=9, framealpha=0.9)
            ax.grid(alpha=0.3)

        fig.suptitle(title or auto_title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return fig, summary    

    def plot_graph_advantage(self,
        graph_ref:      str = 'graph1',   # identity — reference
        graph_a:        str = 'graph2',   # geo
        graph_b:        str = 'graph3',   # random
        metric:         str = 'ccc',
    ):
        """
        ...
        """
        variable_alias  = self.experiment_config.variable_alias
        varvalues       = self.experiment_config.variable_values        

        records = []

        for varvalue in varvalues:
            hl_df = self.metrics_df[
                (self.metrics_df[variable_alias] == varvalue) &
                (self.metrics_df['model_type'] == 'gcn2')
            ].copy()

            for seed in hl_df['seed'].dropna().unique():
                seed_df = hl_df[hl_df['seed'] == seed]

                def node_mean(graph):
                    sub = seed_df[seed_df['graph'] == graph]
                    return sub.groupby('node')[metric].mean()

                ref = node_mean(graph_ref)
                a   = node_mean(graph_a)
                b   = node_mean(graph_b)

                common_a = ref.index.intersection(a.index)
                common_b = ref.index.intersection(b.index)

                if len(common_a) > 0:
                    gap_a = (a.loc[common_a] - ref.loc[common_a]).mean()
                    records.append({variable_alias: varvalue, 'seed': seed, 'comparison': f'{graph_a} vs {graph_ref}', 'gap': gap_a})

                if len(common_b) > 0:
                    gap_b = (b.loc[common_b] - ref.loc[common_b]).mean()
                    records.append({variable_alias: varvalue, 'seed': seed, 'comparison': f'{graph_b} vs {graph_ref}', 'gap': gap_b})

        gap_df = pd.DataFrame(records)

        gap_summary = (gap_df
                    .groupby([variable_alias, 'comparison'])['gap']
                    .agg(mean='mean', std='std')
                    .reset_index())

        palette = {
            f'{graph_a} vs {graph_ref}': '#2196F3',
            f'{graph_b} vs {graph_ref}': '#FF9800',
        }

        fig, ax = plt.subplots(figsize=(10, 5))

        for comp in gap_summary['comparison'].unique():
            sub   = gap_summary[gap_summary['comparison'] == comp].sort_values(variable_alias)
            color = palette.get(comp, 'black')

            ax.plot(sub[variable_alias], sub['mean'],
                    label=comp, color=color, marker='o', linewidth=2)
            ax.fill_between(sub[variable_alias],
                            sub['mean'] - sub['std'],
                            sub['mean'] + sub['std'],
                            color=color, alpha=0.12)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5,
                label='no difference')
        ax.set_xlabel('Forecast horizon (weeks ahead)', fontsize=12)
        ax.set_ylabel(f'Δ {metric.upper()} vs {graph_ref} (identity)\n(positive = graph beats identity)',
                    fontsize=11)
        ax.set_title(f'Graph structure advantage over {graph_ref} across horizons',
                    fontsize=12, fontweight='bold')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        return fig, gap_df, gap_summary

class ExperimentAnalyzerTableMixin:
    """
    Mixin class to ExperimentAnalyzer;
    provides table generation and printing for analysis.

    Methods
    -------
    - `create_results_table()`
    - `print_results_table()`
    - `summarise_for_analysis()`

    See Also
    --------
    ExperimentAnalyzer
    """
    metrics_df:         pd.DataFrame
    experiment_config:  ExperimentConfig

    # ── model display config ──────────────────────────────────────────────
    MODEL_ORDER = [
        'persistence', 'climateology', 'lstm',
        'gcn2_graph1', 'gcn2_graph2', 'gcn2_graph3', 'gcn2_graph4'
    ]
    MODEL_LABELS = {
        'persistence'   : 'Persistence',
        'climateology'  : 'Climatology',
        'lstm'          : 'LSTM',
        'gcn2_graph1'   : 'GCN2 (identity)',
        'gcn2_graph2'   : 'GCN2 (geo)',
        'gcn2_graph3'   : 'GCN2 (random)',
        'gcn2_graph4'   : 'GCN2 (commuter)',
    }
    METRIC_LABELS = {
        'ccc'    : 'CCC',
        'r2'     : 'R²',
        'rmse'   : 'RMSE',
        'mae'    : 'MAE',
        'pearson': 'Pearson r',
        'mda'    : 'MDA',
        'mbe'    : 'MBE',
    }

    def _get_model_key(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add model_key column if not present."""
        df = df.copy()
        if 'model_key' not in df.columns:
            df['model_key'] = df.apply(
                lambda r: f"{r['model_type']}_{r['graph']}"
                if pd.notna(r.get('graph')) and r.get('graph') is not None
                else r['model_type'],
                axis=1
            )
        return df

    def create_results_table(
        self,
        metrics:    list                    = ['ccc', 'r2', 'rmse', 'mae'],
        reference:  str                     = 'persistence',
        varvalues:  list                    = None,
    ) -> pd.DataFrame:
        """
        Create results table with mean ± std per model per variable value,
        plus delta over reference model.

        Parameters
        ----------
        metrics   : list of metric column names to include
        reference : model_key of reference model for delta computation
        varvalues : variable values to include. Defaults to all.

        Returns
        -------
        pd.DataFrame with columns:
            variable_alias, model_key, {metric}_mean, {metric}_std,
            {metric}_delta_mean, {metric}_delta_std for each metric
        """
        variable_alias = self.experiment_config.variable_alias

        df = self._get_model_key(self.metrics_df)

        if varvalues is not None:
            df = df[df[variable_alias].isin(varvalues)].copy()

        # compute reference mean per varvalue per metric
        ref_df = df[df['model_key'] == reference].copy()
        if ref_df.empty:
            raise ValueError(
                f"Reference model '{reference}' not found. "
                f"Available: {df['model_key'].unique().tolist()}"
            )

        for metric in metrics:
            ref_mean = (ref_df
                        .groupby(variable_alias)[metric]
                        .mean()
                        .rename(f'{metric}_ref'))
            df = df.merge(ref_mean, on=variable_alias, how='left')
            df[f'{metric}_delta'] = df[metric] - df[f'{metric}_ref']

        # aggregate: nodes → seeds → summary
        agg_cols = metrics + [f'{m}_delta' for m in metrics]

        per_seed = (df
                    .groupby([variable_alias, 'model_key', 'seed'])[agg_cols]
                    .mean()
                    .reset_index())

        records = []
        for (varval, model_key), grp in per_seed.groupby([variable_alias, 'model_key']):
            row = {variable_alias: varval, 'model': model_key}
            for metric in metrics:
                row[f'{metric}_mean']       = grp[metric].mean()
                row[f'{metric}_std']        = grp[metric].std()
                row[f'{metric}_delta_mean'] = grp[f'{metric}_delta'].mean()
                row[f'{metric}_delta_std']  = grp[f'{metric}_delta'].std()
            records.append(row)

        table = pd.DataFrame(records)

        # apply model order
        present = [m for m in self.MODEL_ORDER if m in table['model'].unique()]
        table['model'] = pd.Categorical(table['model'], categories=present, ordered=True)
        table = table.sort_values(['model', variable_alias]).reset_index(drop=True)

        return table

    def print_results_table(
        self,
        table:      pd.DataFrame,
        metric:     str     = 'ccc',
        style:      str     = 'both',
        precision:  int     = 4,
    ) -> None:
        """
        Print formatted results table.

        Parameters
        ----------
        table     : output of create_results_table()
        metric    : metric to display
        style     : 'absolute', 'delta', or 'both'
        precision : decimal places
        """
        variable_alias  = self.experiment_config.variable_alias
        varvalues       = sorted(table[variable_alias].unique())
        models          = table['model'].cat.categories.tolist()
        metric_label    = self.METRIC_LABELS.get(metric, metric.upper())

        col_width = 16
        name_width = 20

        def fmt(mean, std):
            return f"{mean:+.{precision}f} ± {std:.{precision}f}"

        def header_row():
            return (f"{'model':<{name_width}}"
                    + "".join(f"  {variable_alias}={v:<{col_width-4}}"
                              for v in varvalues))

        def separator():
            return "─" * (name_width + (col_width + 2) * len(varvalues))

        def model_row(model, col_mean, col_std):
            label = self.MODEL_LABELS.get(model, model)
            row   = f"{label:<{name_width}}"
            for v in varvalues:
                sub = table[(table['model'] == model) & (table[variable_alias] == v)]
                if sub.empty:
                    row += f"  {'—':<{col_width}}"
                else:
                    row += f"  {fmt(sub[col_mean].values[0], sub[col_std].values[0]):<{col_width+6}}"
            return row

        if style in ('absolute', 'both'):
            print(f"\n── {metric_label} (mean ± std across seeds) "
                  + "─" * 30)
            print(header_row())
            print(separator())
            for model in models:
                print(model_row(model, f'{metric}_mean', f'{metric}_std'))

        if style in ('delta', 'both'):
            print(f"\n── Δ{metric_label} over reference (mean ± std across seeds) "
                  + "─" * 20)
            print(header_row())
            print(separator())
            for model in models:
                print(model_row(model, f'{metric}_delta_mean', f'{metric}_delta_std'))

    def print_geo_identity_gap(
        self,
        table:      pd.DataFrame,
        metric:     str = 'ccc',
        geo:        str = 'gcn2_graph2',
        identity:   str = 'gcn2_graph1',
    ) -> None:
        """Print geo vs identity gap per variable value."""
        variable_alias = self.experiment_config.variable_alias
        varvalues      = sorted(table[variable_alias].unique())
        metric_label   = self.METRIC_LABELS.get(metric, metric.upper())

        print(f"\n── {metric_label} gap: geo − identity ──────────────────────────────────")
        print(f"  {'varvalue':<12}  {'mean':>10}  {'std':>10}")
        print("  " + "─" * 36)

        for v in varvalues:
            geo_row = table[(table['model'] == geo) & (table[variable_alias] == v)]
            idn_row = table[(table['model'] == identity) & (table[variable_alias] == v)]

            if geo_row.empty or idn_row.empty:
                continue

            gap_mean = geo_row[f'{metric}_mean'].values[0] - idn_row[f'{metric}_mean'].values[0]
            gap_std  = (geo_row[f'{metric}_std'].values[0]**2
                        + idn_row[f'{metric}_std'].values[0]**2) ** 0.5

            print(f"  {variable_alias}={v:<10}  {gap_mean:+.4f}  ±  {gap_std:.4f}")

    def summarise_for_analysis(
        self,
        metrics:    list = ['ccc', 'r2', 'rmse', 'mae'],
        reference:  str  = 'persistence',
        varvalues:  list = None,
        style:      str  = 'both',
    ) -> pd.DataFrame:
        """
        Print full summary for all metrics and return table.
        Paste output here for analysis.

        Parameters
        ----------
        metrics   : metrics to summarise
        reference : reference model for delta computation
        varvalues : variable values to include
        style     : 'absolute', 'delta', or 'both'

        Returns
        -------
        pd.DataFrame — full results table
        """
        variable_alias = self.experiment_config.variable_alias

        table = self.create_results_table(
            metrics   = metrics,
            reference = reference,
            varvalues = varvalues,
        )

        print(f"\n{'='*80}")
        print(f"  RESULTS SUMMARY")
        print(f"  experiment : {self.experiment_config.experiment_name}")
        print(f"  reference  : {reference}")
        print(f"  variable   : {variable_alias}")
        print(f"{'='*80}")

        for metric in metrics:
            self.print_results_table(table, metric=metric, style=style)
            print()

        # geo vs identity gap for ccc
        if 'ccc' in metrics:
            self.print_geo_identity_gap(table, metric='ccc')

        print(f"\n{'='*80}\n")

        return table

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

    dlms:               dict
    experiment_config:  'ExperimentConfig'

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
            varvalue = self.experiment_config.variable_values[0]

        data_orch = self.dlms[varvalue].baseline.dataorchestrator

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
        experiment_name = self.experiment_config.experiment_name
        variable_alias  = self.experiment_config.variable_alias

        print(f"\n{'='*60}")
        print(f"  MORAN'S I ANALYSIS")
        print(f"  experiment : {experiment_name}")
        print(f"  graph_key  : {graph_key}")
        print(f"  dataset    : {dataset}")
        print(f"{'='*60}")

        # use first varvalue only — spatial structure doesn't change with horizon
        varvalue = self.experiment_config.variable_values[0]

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

class ExperimentAnalyzer(ExperimentHandler, 
                         ExperimentAnalyzerPlotterMixin, 
                         ExperimentAnalyzerTableMixin, 
                         ExperimentAnalyzerSpatialAutocorrMixin):
    """ 
    Analyzes experiments
    Subclass of ExperimentHandler, that internally uses another subclass of that,
    namely ExperimentLoader.

    See Also
    --------
    For more information, see ExperimentHandler

    Methods
    -------
    - `compile_metrics()`
    
    For more methods, see ExperimentAnalyzerPlotterMixin
    """
    def __init__(self, 
                 experiment: str):
        self.experiment         = experiment
        self.experiment_dir     = Path(get_project_utilities_env()) / "models" / experiment
        self.experiment_loader  = ExperimentLoader(self.experiment)
        self.experiment_config  = self.experiment_loader.experiment_cfg

        super().__init__(self.experiment_loader.epiconfig, 
                         self.experiment_config)     
        
    # ======= METHODS =========== #
    def _load_models(self) -> Dict[Union[int, str, float], List[BaseModel]]:
        self.models = self.experiment_loader.load_models()

    # ========= HIDDEN METHODS ========= #
    def _load_dataorchs(self):
        self.data_orch_dict = {
            hl: self.experiment_loader.dlms[hl].baseline.dataorchestrator 
                  for hl in self.experiment_loader.dlms}

    def compile_metrics(self
    ) -> pd.DataFrame:
        """
        Evaluate all models across horizons and return flat metrics dataframe.
        Adds columns: horizon, seed, model_type, graph, model_color
        """
        all_metrics     = []
        varvalues_list  = self.experiment_cfg.variable_values
        variable_alias  = self.experiment_cfg.variable_alias

        for varvalue, models in self.models.items():

            # TODO
            if varvalue not in varvalues_list:
                raise ValueError(f'value {varvalue} should not exist!')

            print(f"Compiling metrics for: {variable_alias} = {varvalue}")

            baseline_dlm = self.dlms[varvalue].baseline

            persistence  = PersistenceModel(baseline_dlm,  f'persistence-{variable_alias}{varvalue}')
            climatology  = ClimateologyModel(baseline_dlm, f'climateology-{variable_alias}{varvalue}')

            persistence.forecast('test')
            climatology.forecast('test')

            all_models: List[BaseModel] = [persistence, climatology] + models

            for ml in all_models:
                ml.forecast()

            evaluator  = Evaluator(all_models)
            evaluator.add_evaluation(horizon=0, dataset='test')

            metrics_hl = evaluator.data_compilation.get_data(0, 'test')['metrics'].copy()
            metrics_hl[variable_alias] = varvalue

            # parse model name into components
            def parse_name(name: str) -> dict:
                parts = name.replace('.pt', '').split('-')
                # examples: gcn2-graph1-hl1-s42, lstm-hl1-s42, persistence-hl1, climateology-hl1
                if parts[0] in ('persistence', 'climateology'):
                    return {'model_type': parts[0], 'graph': None, 'seed': None}
                if len(parts) == 4:  # gcn2-graph1-hl1-s42
                    return {'model_type': parts[0], 'graph': parts[1], 'seed': int(parts[3].replace('s', ''))}
                if len(parts) == 3:  # lstm-hl1-s42
                    return {'model_type': parts[0], 'graph': None, 'seed': int(parts[2].replace('s', ''))}
                return {'model_type': parts[0], 'graph': None, 'seed': None}

            parsed = metrics_hl['model'].apply(lambda n: pd.Series(parse_name(n)))
            metrics_hl = pd.concat([metrics_hl, parsed], axis=1)

            # assign colors — differentiate by graph type for GCN
            color_map = {
                ('persistence',  None)     : '#9E9E9E',
                ('climateology', None)     : '#9C27B0',
                ('lstm',         None)     : '#4CAF50',
                ('gcn2',         'graph1') : '#BBDEFB',   # identity — light blue
                ('gcn2',         'graph2') : '#2196F3',   # geo — blue
                ('gcn2',         'graph3') : '#FF9800',   # random — orange
                ('gcn2',         'graph4') : '#F44336',   # commuter — red
            }
            metrics_hl['model_color'] = metrics_hl.apply(
                lambda r: color_map.get((r['model_type'], r['graph']), '#BDBDBD'), axis=1
            )

            # clean model label for plotting
            metrics_hl['model_label'] = metrics_hl.apply(
                lambda r: r['model_type'] if r['graph'] is None else f"{r['model_type']}\n{r['graph']}", axis=1
            )

            all_metrics.append(metrics_hl)
            del evaluator
            print(f"  hl={varvalue} done")

        self.metrics_df = pd.concat(all_metrics, ignore_index=True)
