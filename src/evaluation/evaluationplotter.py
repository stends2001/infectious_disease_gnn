import matplotlib.pyplot as plt 
import seaborn as sns
from typing import Literal, Optional, List, Union, TYPE_CHECKING, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import numpy as np

from ..utils import testcolor, check_dataset
from ..plotting import convert_managedfigure, ManagedFigure, calculate_subplot_layout

if TYPE_CHECKING:
    from .evaluator import Evaluator

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .issues import MetricError

class EvaluationPlotter:

    def __init__(self, evaluator: 'Evaluator'):
        self.evaluator = evaluator

    @convert_managedfigure
    def plot_single_metric(self, 
                    metric:         str, 
                    horizon:        int,
                    dataset:        str = 'test',
                    plot_type:      Literal['violin', 'box', 'map'] = 'violin', 
                    highlight_node: Optional[float] = None,
                    log:            bool = False,
                    reverse_scale:  bool = False,
                    vmin:           Optional[float] = None,
                    vmax:           Optional[float] = None) -> ManagedFigure:
        """
        Plot specified metric across models.
        
        Parameters
        -----------
        metric : str
            Metric name (corr, mse, rmse, ccc, lag_corr, neighbor_corr, spatial_autocorr)
        horizon : int
            Which horizon to plot
        plot_type : str
            Type of plot (violin, box, map)
        highlight_node : Optional[float]
            If provided, a red dot will be placed at this value on each distribution.
        """       
        horizon_str         = f'horizon_{horizon}'

        self._validate_metric(metric)

        # Get colors
        model_class_colors = {ml.model_class: ml.model_color for ml in self.evaluator.evaluated_models.values()}
        model_name_colors  = {ml.name: ml.model_color  for ml in self.evaluator.evaluated_models.values()}
        
        # Prepare data
        metrics_df = self.evaluator.data_compilation.get_data(horizon,dataset)['metrics'][[self.evaluator.id_col,'model',metric]]
        
        if plot_type in ['violin', 'box']:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            self._plot_distribution_on_ax(
                ax, 
                metrics_df, 
                metric, 
                plot_type, 
                model_name_colors, 
                model_class_colors,
                highlight_node, 
                add_legend=True,
                log = log,
                vmin = vmin ,
                vmax = vmax
            )
            plt.tight_layout()
        elif plot_type == 'map':
            fig = self._plot_map(metrics_df, 
                                 metric, 
                                 model_name_colors,
                                 log,
                                 reverse_scale,
                                 vmin = vmin, vmax = vmax)
        else:
            raise ValueError("plot_type must be 'violin', 'box', or 'map'")
        
        plt.close()
        return fig

    def _plot_distribution_on_ax(self, 
                                 ax: Axes, 
                                 df: pd.DataFrame, 
                                 metric: str, plot_type: str,
                                 model_name_colors: dict, model_class_colors: dict,
                                 highlight_node: Optional[int] = None,
                                 add_legend: bool = True,
                                 log: bool = False,
                                 vmin: Optional[float] = None, 
                                 vmax: Optional[float] = None) -> None:
        """
        Plot violin or box plot on a given axes object.
        
        Parameters
        ----------
        ax : Axes
            Matplotlib axes to plot on
        df : pd.DataFrame
            Long-format dataframe with columns: node, model, value
        metric : str
            Name of the metric being plotted
        plot_type : str
            'violin' or 'box'
        model_name_colors : dict
            Mapping of model names to colors
        model_class_colors : dict
            Mapping of model classes to colors
        highlight_node : Optional[int]
            If provided, highlights this specific node
        add_legend : bool
            Whether to add legend to this plot
        """
        plot_func = sns.violinplot if plot_type == 'violin' else sns.boxplot

        if log:
            df[metric] = np.log1p(df[metric])

        plot_func(
            data=df, x='model', y=metric, hue='model',
            ax=ax, palette=model_name_colors,
            **(dict(cut=0) if plot_type == 'violin' else {}),
            legend=False
        )
        
        ylab = f"{metric} [log]" if log else f"{metric}"

        # ax limits
        if not vmin:
            vmin = df[metric].min()
        if not vmax:
            vmax = df[metric].max()


        ax.set_title(f'{metric.upper()}')
        ax.set_ylabel(ylab)
        ax.set_xlabel('Model')
        ax.grid(alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='center', fontsize=8)
        ax.set_ylim(vmin, vmax)

        if highlight_node:
            # Plot red dot for the specified node for all models
            node_values = df[df[self.evaluator.id_col] == highlight_node]  # Get values for the specified node
            
            # Loop over each model and plot a red dot for the specified node
            for i, model in enumerate(model_name_colors.keys()):
                # Find the corresponding value for the node for each model
                node_value = node_values[node_values['model'] == model][metric].values
                if len(node_value) > 0:
                    # Plot red dot at the value of the node for the current model
                    ax.scatter(
                        x=i,  # i is the x-position of the model in the distribution
                        y=node_value[0],  # the y-position is the value for the specified node
                        color='red', 
                        zorder=10, 
                        s=100, 
                        label=f'Node {highlight_node}' if i == 0 else ""
                    )

        # Legend
        if add_legend:
            handles = [mpatches.Patch(color=c) for c in model_class_colors.values()]
            ax.legend(handles, model_class_colors.keys(), title='Model Class', loc='best')
    

    def _plot_distribution(self, df: pd.DataFrame, 
                           metric: str, 
                           plot_type: str,
                           model_name_colors: dict, 
                           model_class_colors: dict,
                           node: Optional[int]) -> Figure:
        """
        Plot violin or box plot (legacy wrapper for backward compatibility).
        Creates a new figure and calls _plot_distribution_on_ax.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        self._plot_distribution_on_ax(
            ax, df, metric, plot_type, 
            model_name_colors, model_class_colors,
            node, add_legend=True
        )
        plt.tight_layout()
        plt.close()
        return fig

    def _plot_map(self, metric_df: pd.DataFrame, metric: str, model_name_colors: dict, log: bool, reverse_scale: bool,
                                 vmin: Optional[float] = None, 
                                 vmax: Optional[float] = None):
        """Plot spatial map of metric values."""       
    
        if reverse_scale:
            cmap = "coolwarm_r"
        else:
            cmap = "coolwarm"

        models = list(self.evaluator.evaluated_models.values())
        # Calculate layout
        n_models = len(self.evaluator.evaluated_models)
        nrows, ncols, figsize = calculate_subplot_layout(n_models+1, target_width=8, target_height=6)
        
        ctx = models[0].dataloadermanager.dataorchestrator.data_context

        gdf = ctx.shapedata_node

        map_df = gpd.GeoDataFrame(pd.merge(metric_df, gdf, on=['nuts_node']))
        if log:
            map_df[metric] =  np.log1p(pd.to_numeric(map_df[metric]))

        if not vmin:
            vmin = map_df[metric].min()
        if not vmax:
            vmax = map_df[metric].max()

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten()

        idx = -1

        for model_name, model in self.evaluator.evaluated_models.items():

            idx += 1

            data = map_df[map_df['model'] == model_name].reset_index(drop = False)


            data.plot(
                ax=axes[idx],
                column=metric,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                legend=False       # no legend on any subplot
            )

            ctx.shapedata_nuts2.plot(ax = axes[idx], facecolor = 'none', linewidth = 1, edgecolor = 'grey')
            ctx.shapedata_nuts1.plot(ax = axes[idx], facecolor = 'none', linewidth = 1, edgecolor = 'black')
            ctx.shapedata_nuts0.plot(ax = axes[idx], facecolor = 'none', linewidth = 2, edgecolor = 'black')    
            axes[idx].set_title(model_name)

        # --- colorbar in the last (empty) axis ---
        last_ax = axes[-1]
        last_ax.set_visible(False)          # hide the map frame/ticks

        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])                    # required boilerplate

        cbar = fig.colorbar(
            sm,
            ax=last_ax,                     # anchor to the empty axis
            fraction=0.5,                   # how much of the axis the cbar fills
            shrink=0.8,
            pad=0.05
        )
        cbar_title = f"{metric} [log]" if log else f"{metric}"
        cbar.set_label(f'{metric}', fontsize=12)

        fig.suptitle(f'{metric} distribution',fontweight = 'bold')
        plt.tight_layout()
        plt.show()

    def _validate_metric(self, metric: str) -> None:
        if metric not in self.evaluator.metric_calculator.supported_metrics:
            raise MetricError(f'invalid metric {metric}. Supported metrics are {self.evaluator.metric_calculator.supported_metrics}')