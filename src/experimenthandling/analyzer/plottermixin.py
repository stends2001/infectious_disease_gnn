import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, Optional, Literal, List
import numpy as np
from scipy.stats import wilcoxon

from ..containers import ExperimentConfig
from ...evaluation import Evaluator

class ExperimentAnalyzerPlotterMixin:
    """
    Mixin class to ExperimentAnalyzer.

    Methods
    -------
    - `plot_metric_over_reference()`
    - `plot_graph_advantage()`
    """
    metrics_df:  pd.DataFrame
    expcfg:      ExperimentConfig
    evaluators:  Dict

    def plot_model_comparison(self,
        models:             List[str]           = ['persistence', 'climateology', 'lstm', 'gcn2-graph1', 'gcn2-graph2'],
        metric:             str                 = 'ccc',
        higher_is_better:   bool                = True,
        reference:          Optional[str]       = None,   # e.g. 'persistence' — if None, plots absolute
        show_significance:  bool                = True,
        interval:           Literal['sem', 'minmax'] = 'sem',
        model_labels:       Dict[str, str]      = {},
    ):
        """
        Plot metric performance across horizons for a set of models.
        Optionally plot delta over a reference model.
        """
        variable_alias  = self.expcfg.variable_alias
        df              = self.metrics_df.copy()

        # build model_key column
        df['model_key'] = df.apply(
            lambda r: f"{r['model_type']}-{r['graph']}" if pd.notna(r['graph']) else r['model_type'],
            axis=1
        )

        df = df[df['model_key'].isin(models)].copy()

        # colors from metrics_df
        colormap = (df[['model_key', 'model_color']]
                    .drop_duplicates()
                    .dropna()
                    .set_index('model_key')['model_color']
                    .to_dict())

        # mean per node x model x hl already done by evaluator
        # aggregate over nodes
        metric_col = metric

        if reference is not None:
            ref_mean = (df[df['model_key'] == reference]
                        .groupby(variable_alias)[metric_col]
                        .mean()
                        .rename('ref_mean'))
            df = df.merge(ref_mean, on=variable_alias)
            delta_sign      = 1 if higher_is_better else -1
            df['plot_val']  = delta_sign * (df[metric_col] - df['ref_mean'])
            ylabel          = f'Δ {metric} vs {model_labels.get(reference, reference)}'
        else:
            df['plot_val']  = df[metric_col]
            ylabel          = metric

        # aggregate over nodes per model per hl
        if interval == 'sem':
            stats = (df.groupby([variable_alias, 'model_key'])['plot_val']
                    .agg(['mean', 'std'])
                    .reset_index())
            n_nodes             = df['node'].nunique()
            stats['sem']        = stats['std'] / np.sqrt(n_nodes)
            stats['ymin']       = stats['mean'] - stats['sem']
            stats['ymax']       = stats['mean'] + stats['sem']
        else:
            stats = (df.groupby([variable_alias, 'model_key'])['plot_val']
                    .agg(['mean', 'min', 'max'])
                    .reset_index()
                    .rename(columns={'min': 'ymin', 'max': 'ymax'}))

        # wilcoxon vs reference if provided
        significance_dict = {}
        if show_significance and reference is not None:
            ref_node_vals = (df[df['model_key'] == reference]
                            .groupby([variable_alias, 'node'])[metric_col]
                            .mean())

            for model_key in models:
                if model_key == reference:
                    continue
                mod_node_vals = (df[df['model_key'] == model_key]
                                .groupby([variable_alias, 'node'])[metric_col]
                                .mean())

                for hl in df[variable_alias].unique():
                    ref_vals = ref_node_vals.get(hl, pd.Series(dtype=float))
                    mod_vals = mod_node_vals.get(hl, pd.Series(dtype=float))

                    common = ref_vals.index.intersection(mod_vals.index)
                    if len(common) < 5:
                        continue

                    x, y = mod_vals.loc[common], ref_vals.loc[common]
                    alt  = 'greater' if higher_is_better else 'less'

                    try:
                        _, p = wilcoxon(x, y, alternative=alt)
                        significance_dict[(model_key, hl)] = p
                    except Exception:
                        pass

        # ── plot ─────────────────────────────────────────────────────────
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        for model_key in models:
            model_stats = stats[stats['model_key'] == model_key].sort_values(variable_alias)
            if model_stats.empty:
                continue

            color = colormap.get(model_key, '#333333')
            label = model_labels.get(model_key, model_key)

            ax.plot(model_stats[variable_alias], model_stats['mean'],
                    color=color, marker='o', linewidth=2, markersize=5, label=label)

            ax.fill_between(model_stats[variable_alias],
                            model_stats['ymin'], model_stats['ymax'],
                            color=color, alpha=0.12)

            if show_significance and reference is not None and model_key != reference:
                y_offset = (model_stats['ymax'] - model_stats['mean']).max() * 1.5

                for _, row in model_stats.iterrows():
                    hl = row[variable_alias]
                    p  = significance_dict.get((model_key, hl), 1.0)
                    y  = row['ymax'] + y_offset

                    if p < 0.001:
                        marker = '***'
                    elif p < 0.01:
                        marker = '**'
                    elif p < 0.05:
                        marker = '*'
                    else:
                        marker = 'ns'

                    ax.annotate(marker, xy=(hl, y),
                                ha='center', va='bottom',
                                fontsize=8, color=color)

        if reference is not None:
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--',
                    alpha=0.5, label=f'reference: {model_labels.get(reference, reference)}')

        ref_label   = model_labels.get(reference, reference) if reference else None
        title_str   = (f'{metric} relative to {ref_label}' if reference 
                    else f'{metric} across horizons')

        ax.set_xlabel('Forecast horizon (weeks ahead)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title_str, fontsize=12, fontweight='bold', loc='left')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        return stats, significance_dict

    def plot_graph_advantage(self,
        graph_ref:  str         = 'graph1',
        model:      str         = 'gcn2',
        metric:     str         = 'ccc',
        higher_is_better: bool  = True,
        show_significance: bool = True,
        interval: Literal['minmax', 'sem'] = 'sem',
        graph_labels: Dict[str, str] = {}
    ):
        """plot graph advantage over reference"""
        variable_alias  = self.expcfg.variable_alias
        varvalues       = self.expcfg.variable_values
        unique_graphs   = [gs for gs in self.metrics_df['graph'].unique() if gs is not None]
        graphcolors_df  = self.metrics_df[['graph','model_color']].drop_duplicates().dropna()
        graphcolors     = {row['graph']: row['model_color'] for index, row in graphcolors_df.iterrows()}
        
        # wide_df: columns: ['node', {variable_alias}, 'graph1', ...]
        wide_df         = (
                            self.metrics_df[self.metrics_df['model_type'] == model]
                        .pivot_table(index = ['node',self.expcfg.variable_alias], values = metric, columns = 'graph')
                        .reset_index()
                        )
        
        # wide_df_delta: the same as wide_df, but now metric has been subtracted from reference
        wide_df_delta_abs   = wide_df.copy()
        wide_df_delta_pct   = wide_df.copy()


        # positive / negative orientation depends on whether higher is better
        delta_sign = 1 if higher_is_better else -1

        ref_values = wide_df[graph_ref].abs().replace(0, np.nan)

        for graph in unique_graphs:
            wide_df_delta_abs[graph] = delta_sign * (wide_df[graph] - wide_df[graph_ref])
            wide_df_delta_pct[graph] = 100 * delta_sign * (wide_df[graph] - wide_df[graph_ref]) / ref_values

        # aggregate over all nodes => get std and mean of metric
        if interval == 'sem':
            stats_abs = (
                wide_df_delta_abs.groupby("hl")[unique_graphs]
                .agg(["mean", "std"])
                .rename_axis(columns=["graph", "stat"])
                .stack(["graph", "stat"])
                .reset_index(name="value")        
            )

            stats_pct = (
                wide_df_delta_pct.groupby("hl")[unique_graphs]
                .agg(["mean", "std"])
                .rename_axis(columns=["graph", "stat"])
                .stack(["graph", "stat"])
                .reset_index(name="value")        
            )

        else:
            stats_abs = (
                wide_df_delta_abs.groupby("hl")[unique_graphs]
                .agg(["mean", "min", 'max'])
                .rename_axis(columns=["graph", "stat"])
                .stack(["graph", 'stat'])
                .reset_index(name="value")        
            )

            stats_pct = (
                wide_df_delta_pct.groupby("hl")[unique_graphs]
                .agg(["mean", "min", 'max'])
                .rename_axis(columns=["graph", "stat"])
                .stack(["graph", "stat"])
                .reset_index(name="value")   
            )     

        # ======== plotting ========== #

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        n_nodes = wide_df_delta_abs['node'].nunique()

        # compute wilcoxon per graph per hl
        significance_dict = {}
        for graph in unique_graphs:
            if graph != graph_ref:
                for hl in wide_df['hl'].unique():
                    comparing = wide_df[wide_df['hl'] == hl][[graph_ref,graph]].dropna()

                    # for higher_isbetter = True => this tests that graph is better than graph_ref
                    if higher_is_better:
                        _, p = wilcoxon(x = comparing[graph], y = comparing[graph_ref], alternative='greater')

                    else:
                        _, p = wilcoxon(x = comparing[graph], y = comparing[graph_ref], alternative='less')

                    significance_dict[(graph, hl)] = p


        significance_df = pd.DataFrame(
            [(k1, k2, v) for (k1, k2), v in significance_dict.items()],
            columns=["graph", variable_alias, 'p']
        )

        stats = stats_abs.copy()
        for graph in unique_graphs:
            if graph != graph_ref:

                graph_df  = stats[stats['graph'] == graph]
                mean      = graph_df[graph_df['stat'] == 'mean'].reset_index(drop=True)

                if interval == 'sem':
                    std       = graph_df[graph_df['stat'] == 'std'].reset_index(drop=True)
                    sem       = std['value'] / np.sqrt(n_nodes) 
                    ymin = mean['value'] - sem
                    ymax = mean['value'] + sem

                else:
                    ymin         = graph_df[graph_df['stat'] == 'min']['value'].reset_index(drop=True)
                    ymax         = graph_df[graph_df['stat'] == 'max']['value'].reset_index(drop=True)                

                color     = graphcolors.get(graph, '#333333')


                ax.plot(mean['hl'], mean['value'], color=color, marker='o', linewidth=2, markersize=5, label=graph_labels[graph])
                ax.fill_between(mean['hl'], ymin, ymax, color=color, alpha=0.15)

                if show_significance:

                    # significance markers above the SEM band
                    y_offset = (ymax.max() - mean['value'].max()) * 1.5

                    for i, row in mean.iterrows():
                        hl = row['hl']
                        p  = significance_dict.get((graph, hl), 1.0)
                        y  = ymax.iloc[i] + y_offset

                        if p < 0.001:
                            marker = '***'
                        elif p < 0.01:
                            marker = '**'
                        elif p < 0.05:
                            marker = '*'
                        else:
                            marker = 'ns'

                        ax.annotate(marker, xy=(hl, y),
                                    ha='center', va='bottom',
                                    fontsize=9, color=color)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5, label=f'reference: {graph_labels[graph_ref]}')
        ax.set_xlabel('Forecast horizon (weeks ahead)', fontsize=12)
        ax.set_ylabel(f'Δ {metric} improvment', fontsize=11)
        fig.suptitle(f'Graph structure advantage in {metric} over {graph_labels[graph_ref]}', 
                    fontsize=12, fontweight='bold', x=ax.get_position().x0 - 0.0375, y=0.94,
                ha='left', transform=fig.transFigure)
        ax.set_title(f'Interval: {interval}', fontsize=11, loc = 'left', pad=2)  
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        return significance_df
