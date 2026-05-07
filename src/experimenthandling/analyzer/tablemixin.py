from typing import List, Union, Optional
import pandas as pd

from ..containers import ExperimentConfig

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
        metrics:    List[str]                               = ['ccc', 'r2', 'rmse', 'mae'],
        reference:  str                                     = 'persistence',
        varvalues:  Optional[List[Union[str, float, int]]]  = None,
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
