import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import pandas as pd

from ..containers import ExperimentConfig

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
    expcfg:             ExperimentConfig

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

        variable_alias  = self.expcfg.variable_alias
        varvalues       = self.expcfg.variable_values

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
        variable_alias  = self.expcfg.variable_alias
        varvalues       = self.expcfg.variable_values        

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
