from typing import Optional, List, Dict, Tuple, Literal
import pandas as pd
import math 
from scipy.stats import wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt

from ..exceptions import MetricsException
from ..containers import ExperimentConfig
from ...dataloading.epidataorchestration import EpiDataOrchestrator

class ResultsMixin:
    """
    Mixinclass to ExperimentAnalyzer that deals with the saving of metrics_df. 

    Methods
    -------
    - `summarize_absolute_metric()`
    - `summarize_graph_advantage()`
    - `plot_datasplit()`
    - `plot_graph_advantage()`
    """
    metrics_df:  Optional[pd.DataFrame]
    expcfg:      ExperimentConfig
    evaluators:  Dict
    model_names: List[str]
    variable_alias: str
    epidataorchestrators: Dict[int | str | float, EpiDataOrchestrator]

    def _set_modelcolors(self):
        self.model_colors = {
            # baselines
            'climateology'  : "#A6A6A6",
            'persistence'   : "#4D4D4D",

            # gcns
            'gcngraph1'     : "black",
            'gcngraph2'     : "#1F78B4",

            'gcngraph3'     : "#8C4312", 
            'gcngraph4'     : "#B8571A", 
            'gcngraph5'     : "#D97C38",  
            'gcngraph6'     : "#E8A97A", 

            # gats
            'gatgraph1'     : 'black',    
            'gatgraph2'     : "#1B9E77",

            'gatgraph3'     : "#4A296C",  
            'gatgraph4'     : "#6A3D9A",  
            'gatgraph5'     : "#9570B8",  
            'gatgraph6'     : "#C3AEDC", 
        } 
        
        for ml in self.model_names:
            if ml not in self.model_colors:
                print(f'no color found for model {ml}: will be put to black for now')
                self.model_colors[ml] = 'black'

    def summarize_absolute_metric(self, metric: str = 'ccc') -> pd.DataFrame:
        if self.metrics_df is None:
            raise MetricsException('no attribute metrics_df. Run `compile_metrics()` first.')

        df                  = self.metrics_df.copy()
        exp_metric          = df
        exp_metric          = exp_metric[['node',self.variable_alias,'model',metric]]
        summarisation       = exp_metric.groupby([self.variable_alias,'model'])[metric].agg(['mean','std']).reset_index(drop = False)
        summarisation['sem']= summarisation['std'] / math.sqrt(df['node'].nunique())
        return summarisation

    def summarize_graph_advantage(self, metric: str = 'ccc', model_type: str = 'gcn') -> Tuple[pd.DataFrame, pd.DataFrame]:
        df                              = self.metrics_df.copy()
        graph_models                    = [ml for ml in self.model_names if 'graph' in ml and ml.startswith(model_type)]
        baseline                        = next(s for s in graph_models if s.endswith('graph1'))

        graph_advantage                 = df[df['model'].isin(graph_models)].reset_index(drop = True)
        graph_advantage                 = graph_advantage[['node',self.variable_alias,'model',metric]]
        graph_advantage                 = graph_advantage.pivot(index = ['node',self.variable_alias], columns='model', values=metric).reset_index(drop = False)

        graph_advantage_delta           = graph_advantage.copy()
        for ml in graph_models:
            graph_advantage_delta[ml]   = graph_advantage[ml] - graph_advantage[baseline]

        graph_advantage_delta           = graph_advantage_delta.melt(id_vars = ['node',self.variable_alias], value_name= metric, value_vars=graph_models).reset_index(drop = False)
        summarisation                   = graph_advantage_delta.groupby(['hl','model'])[metric].agg(['mean','std']).reset_index(drop = False)
        summarisation['sem']            = summarisation['std'] / math.sqrt(df['node'].nunique())

        return summarisation, graph_advantage

    def plot_graph_advantage(self, 
                                model_type: str,
                                metric: str, 
                                stats: bool = True, 
                                higher_is_better: bool = True, 
                                titles: bool = True):
        self._set_modelcolors()

        df1, df2        = self.summarize_graph_advantage(metric, model_type)
        graphmodels: List[str]     = list(df1['model'].unique())
        graphmodels     = [gm for gm in graphmodels if gm.startswith(model_type)]
        baseline        = f"{model_type}graph1"

        # DETERMINE STATS #
        if stats:
            results = []
            for ml in graphmodels:

                if ml == baseline:
                    continue

                for varvalue in df2[self.variable_alias].unique():

                    subset = df2[df2[self.variable_alias] == varvalue]

                    x = subset[ml]
                    y = subset[baseline]

                    mask = ~(x.isna() | y.isna())
                    x, y = x[mask], y[mask]

                    if len(x) == 0:
                        continue

                    stat, p = wilcoxon(x, y, alternative = 'greater' if higher_is_better else 'less')

                    results.append({
                        self.variable_alias: varvalue,
                        "model":    ml,
                        "p":        p,
                        "mean_diff": (x - y).mean()
                    })

            stats_df = pd.DataFrame(results)
            stats_df["sig"] = stats_df["p"].apply(self._get_stars)

            # get y positions from your summary
            y_pos       = df1.groupby([self.variable_alias, "model"])["mean"].max().reset_index()
            y_pos       = y_pos.rename(columns={"mean": "y"})
            stats_df    = stats_df.merge(y_pos, on=[self.variable_alias, "model"], how="left")

            # offset so stars sit above error bars
            stats_df["y"] = stats_df["y"] + 0.02

        
        # SETUP PLOT
        fig, ax = plt.subplots(1,1, figsize = (10, 4))
        ax.grid()

        # Plot lines
        for ml in df1['model'].unique():
            subset = df1[df1['model'] == ml] 
            sns.lineplot(subset, x = 'hl', y = 'mean', c = self.model_colors[ml], ax = ax, marker = 'o', label = ml)
            plt.fill_between(
                x   =subset['hl'],
                y1   =subset['mean'] - subset['sem'],
                y2   =subset['mean'] + subset['sem'],
                color = self.model_colors[ml],
                alpha = 0.2
            )    

        # Plot stats
        if stats:
            for _, row in stats_df.iterrows():

                if row["sig"] == "":
                    continue

                ax.text(
                    x=row["hl"],
                    y=row["y"],
                    s=row["sig"],
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color=self.model_colors[row["model"]]
                )

        ax.legend(title = 'graph')
        if titles:
            ax.set_title('Mean ± SEM across nodes', loc = 'left', fontsize = 10)
            fig.suptitle(f'{metric} improvement over Identity Graph across horizons:  {self.expcfg.experiment_name}', fontweight = 'bold', x = 0.375, fontsize = 12)
        ax.set_ylabel(metric)
        ax.set_xlabel('')     

    def plot_datasplit(self, node: int = 0):
        fig, ax = plt.subplots(1,1, figsize = (10,2.5))
        df = self.epidataorchestrators[1].data_final.data_denorm
        df = df[df['node'] == node]

        ax.grid()
        sns.lineplot(df[df['train']], x = 'timestamp', y ='incidence_lag0', label = 'train', color = "#6B6767",   ax = ax)
        sns.lineplot(df[df['val']],   x = 'timestamp', y ='incidence_lag0', label = 'val',  color  = "#6696CE",    ax = ax)
        sns.lineplot(df[df['test']],  x = 'timestamp', y ='incidence_lag0', label = 'test', color  = "#E76464",         ax = ax)
        ax.set_title(f'Influenza incidence in Berlin', loc = 'left', fontsize = 10)
        fig.suptitle(f'Data split experiment {self.expcfg.experiment_name}', fontweight = 'bold', fontsize = 12)
        ax.set_ylabel('incidence')
        ax.set_xlim([pd.to_datetime('2012-06-01'),pd.to_datetime('2025-06-01')])
        ax.set_xlabel('')
        fig.tight_layout()

    def compare_modeltypes(self, metric: str, graph: str | None = None):

        df = self.summarize_absolute_metric(metric)

        model_types = ['gcn','gat']
        graphs      = [graph] if graph is not None else ['graph1','graph2','graph3','graph4']

        models = [x + y for x in model_types for y in graphs]

        if graph is None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)


            for idx, ax in enumerate(axes):

                model_type = model_types[idx]

                for ml in models:
                # Plot lines
                    if ml.startswith(model_type):
                        subset = df[df['model'] == ml] 
                        sns.lineplot(subset, x = 'hl', y = 'mean', c = self.model_colors[ml], ax = ax, marker = 'o', label = ml)
                        ax.errorbar(
                            x   =subset['hl'],
                            y   =subset['mean'],
                            yerr=subset['sem'],
                            fmt ='none', 
                            capsize=3,
                            color = self.model_colors[ml]
                        )       

                ax.set_title('')
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.legend()
                ax.grid()

        if graph is not None:
            fig, ax = plt.subplots(1,1, figsize = (8,4))
            ax.grid()

            for ml in models:
                colors = {models[0]:'#2196F3',
                        models[1]:'#FF9800'}
                # Plot lines
                subset = df[df['model'] == ml] 
                sns.lineplot(subset, x = 'hl', y = 'mean', c = colors[ml], ax = ax, marker = 'o', label = ml)
                ax.errorbar(
                    x   =subset['hl'],
                    y   =subset['mean'],
                    yerr=subset['sem'],
                    fmt ='none', 
                    capsize=3,
                    color = colors[ml]
                )       

                ax.set_title('')
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.legend()

            plt.tight_layout()
            plt.show()

    def _get_stars(self, p) -> str:
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""        

    