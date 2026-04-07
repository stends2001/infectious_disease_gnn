from typing import Literal, Optional, List, Union, TYPE_CHECKING, assert_never
import pandas as pd
import geopandas as gpd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .issues import MetricError

from ..plotting import ManagedFigure, calculate_subplot_layout

if TYPE_CHECKING:
    from .evaluator import Evaluator

class EvaluationPlotter:
    """
    Plotting extension for Evaluator

    Parameters
    ----------
    evaluator: 'Evaluator'

    Methods
    -------
    - plot_metric()

    hidden methods

    - _plot_distribution_on_ax()
    - _plot_kde_on_ax()
    - _plot_metric_map()
    - _plot_metric_best_map()
    - _validate_metric()
    
    """
    def __init__(self, evaluator: 'Evaluator'):
        self.evaluator = evaluator

    def plot_metric(self, 
                    horizon:        int,
                    dataset:        Literal['train','val','test'],
                    metric:         str, 
                    plot_type:      Literal['box', 'violin', 'kde', 'map','best_map'] = 'box', 
                    highlight_node: Optional[int] = None,
                    legend:         bool = True,
                    log:            bool = False,
                    reverse_scale:  bool = False,
                    vmin:           Optional[float] = None,
                    vmax:           Optional[float] = None,
                    margin_fraction:Optional[float] = None
                    ) -> ManagedFigure:
        """
        Plot specified metric across models.
        
        Parameters
        -----------
        horizon: int
            number of horizon to plot
        dataset: Literal['train','val','test']
            dataset of metric to be shown
        metric: str
            name of metric to be shown
        plot_type: Literal['box','violin','kde','map','best_map'] = 'box'
            type of plot shown.
            - box: model by model distribution            
            - violin: model by model distribution
            - kde: distribution on one axis
            - map: geographical plot of metric per nuts-level
            - best_map: geographical plot of best model per nuts-level
        highlight_node: Optional[int] = None
            specific node - token to be higlighed
        log: bool = False
            whether metric - scale is to be logged
        reverse_scale: bool = False
            whether scale should be reversed
        vmin: Optional[float] = None
            the minimal value for the metric. When None, will be inferred
        vmax: Optional[float] = None
            the maximaml value for the metric. When None, will be inferred.
        margin_fraction: Optional[float] = None
            # TODO
        """       
        horizon_str         = f'horizon_{horizon}'
        self._validate_metric(metric)

        # Get colors
        model_class_colors = {ml.model_class: ml.model_color for ml in self.evaluator.evaluated_models.values()}
        
        # when double colors, should be adjusted to have a brighter and darker version TODO
        model_name_colors  = {ml.name: ml.model_color  for ml in self.evaluator.evaluated_models.values()}
        
        # Prepare data
        # df: | node | model | {metric_name} |
        metrics_df = self.evaluator.data_compilation.get_data(horizon,dataset)['metrics'].copy()[[self.evaluator.id_col,'model',metric]]

        # ax limits
        if vmin is None:
            vmin = float(metrics_df[metric].min())
        if vmax is None:
            vmax = float(metrics_df[metric].max())
        
        match plot_type:
            case 'violin' | 'box' | 'kde':
                fig, ax = plt.subplots(1, 1, figsize=(8, 5) if plot_type =='kde' else (12, 5))
                self._plot_distribution_on_ax(
                    ax                  = ax, 
                    metric_df           = metrics_df, 
                    metric              = metric, 
                    plot_type           = plot_type, 
                    model_name_colors   = model_name_colors, 
                    model_class_colors  = model_class_colors,
                    add_legend          = legend,
                    log                 = log,
                    vmin                = vmin,
                    vmax                = vmax,
                    highlight_node      = highlight_node,                     
                )
            case 'best_map':
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                self._plot_metric_best_map(
                    ax                  = ax,
                    metric_df           = metrics_df,
                    metric              = metric,
                    model_name_colors   = model_name_colors,
                    vmin                = vmin,
                    vmax                = vmax,                    
                    reverse_scale       = reverse_scale,
                    highlight_node      = highlight_node,
                    legend              = legend,
                    margin_fraction     = margin_fraction
                )

            case 'map': 
                fig = self._plot_metric_map(
                    metric_df           = metrics_df,
                    metric              = metric,
                    reverse_scale       = reverse_scale,
                    legend              = legend,
                    log                 = log,
                    vmin                = vmin, 
                    vmax                = vmax,
                    highlight_node      = highlight_node,                    
                )

            case _:
                assert_never(plot_type)

        plt.close()                
        return ManagedFigure(fig)

    def _plot_distribution_on_ax(self, 
                                 ax: Axes, 
                                 metric_df: pd.DataFrame, 
                                 metric: str, 
                                 plot_type: Literal['violin','box','kde'],
                                 model_name_colors: dict, 
                                 model_class_colors: dict,
                                 add_legend: bool,
                                 log: bool,
                                 vmin: float, 
                                 vmax: float,
                                 highlight_node: Optional[int] = None
                                 ):
        """
        Plot violin or box plot on a given axes object.
        
        Parameters
        ----------
        # TODO
        """
        df = metric_df.copy()
        metric_label = f"{metric}"

        if log:
            df[metric] = np.log1p(df[metric])
            metric_label += " [log]"

        ground_arguments = {'data':df,'ax':ax,'legend':False,'hue':'model','palette':model_name_colors}

        match plot_type:
            case 'violin':
                plot_func = sns.violinplot 
                arg_adds  = {"x"        : 'model',
                             "y"        : metric,
                             "cut"      : 0
                             }
                xlab    = "model"
                ylab    = metric_label
                ax.tick_params(axis='x', rotation=30)
                ax.set_ylim(ymin = vmin, ymax = vmax)

            case 'box':
                plot_func = sns.boxplot
                arg_adds  = {"x"        : 'model',
                             "y"        : metric,
                             }     
                xlab    = "model"
                ylab    = metric_label      
                ax.tick_params(axis='x', rotation=30)
                ax.set_ylim(ymin = vmin, ymax = vmax)          

            case 'kde':
                plot_func = sns.kdeplot
                arg_adds  = {"x"            : metric,
                             "common_norm"  : False,
                             "common_grid"  : True,
                             "fill"         : True,
                             "alpha"        : 0.3,
                             "linewidth"    : 1.5
                             }                  
                ylab    = "density"
                xlab    = metric_label            
                ax.set_xlim(xmin = vmin, xmax = vmax)
                    
            case _:
                assert_never(plot_type)
        
        plot_arguments = ground_arguments | arg_adds # | is dict union

        plot_func(**plot_arguments)
        
        ax.set_title(f'{metric.upper()}')
        ax.set_ylabel(ylab)
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.3)
        

    # --- Highlight node ---
        if highlight_node is not None:
            node_rows = df[df[self.evaluator.id_col] == highlight_node]

            match plot_type:
                case 'violin' | 'box':
                    for i, model in enumerate(model_name_colors.keys()):
                        vals = node_rows[node_rows['model'] == model][metric].values
                        if len(vals) > 0:
                            ax.scatter(
                                x=i, y=vals[0],
                                color='red', edgecolors='darkred',
                                zorder=10, s=80,
                                label=f'Node {highlight_node}' if i == 0 else ""
                            )

                case 'kde':
                    for model, color in model_name_colors.items():
                        vals = node_rows[node_rows['model'] == model][metric].values
                        if len(vals) > 0:
                            ax.axvline(
                                x=vals[0],
                                color=color,
                                alpha=0.85,
                                linewidth=1.5,
                                linestyle='--',
                            )

                    dot_y = ax.get_ylim()[1] * 0.03  # ← after all vlines, ymax is stable
                    for model, color in model_name_colors.items():
                        vals = node_rows[node_rows['model'] == model][metric].values
                        if len(vals) > 0:
                            ax.scatter(
                                x=vals[0], y=dot_y,
                                color=color, edgecolors='black',
                                zorder=10, s=80,
                            )

        if add_legend:
            handles: list[Union[mpatches.Patch, Line2D]]
            handles = [mpatches.Patch(color=color, label=model_class)
                    for model_class, color in model_class_colors.items()]

            if highlight_node is not None:
                handles.append(
                    Line2D([0], [0],
                        marker='o',
                        linestyle='',
                        markersize=10,
                        markeredgecolor='black',
                        markeredgewidth=1,
                        markerfacecolor='red',
                        label=f'Node {highlight_node}')
                )

            ax.legend(handles=handles, title='Model Class', loc='best')
  
    def _plot_metric_map(self, 
                         metric_df:         pd.DataFrame,
                         metric:            str, 
                         reverse_scale:     bool,
                         legend:            bool,
                         log:               bool,
                         vmin:              float,
                         vmax:              float,
                         highlight_node:    Optional[int] = None) -> Figure:
        """Plot spatial map of metric values."""       
        axes_np: np.ndarray
        axes:    List[Axes]        
        
        df  = metric_df.copy()
        cmap= 'coolwarm'

        # good metric value : blue
        if not reverse_scale:
            cmap += "_r"

        models = list(self.evaluator.evaluated_models.values())

        # Calculate layout
        n_models                = len(self.evaluator.evaluated_models)
        nrows, ncols, figsize   = calculate_subplot_layout(n_models+1, target_width=8, target_height=6)
        
        # reference data
        ctx     = models[0].dataloadermanager.dataorchestrator.data_context
        gdf     = ctx.local_shapedata

        # map data
        map_df  = gpd.GeoDataFrame(pd.merge(df, gdf, on=[self.evaluator.id_col]))

        # make metric log
        if log:
            map_df[metric] =  np.log1p(pd.to_numeric(map_df[metric]))

        fig, axes_np    = plt.subplots(nrows, ncols, figsize=figsize)
        axes            = axes_np.flatten().tolist()

        # axes filling with model metrics
        for idx, model_name in enumerate(self.evaluator.evaluated_models.keys()):

            data = map_df[map_df['model'] == model_name].reset_index(drop = False)
            
            # gdf.plot() color map
            data.plot(
                ax      = axes[idx],
                column  = metric,
                cmap    = cmap,
                vmin    = vmin,
                vmax    = vmax,
                legend  = False  # legend will come in the final axis
            )
            # gdf.plot() context nuts values
            ctx.global_shapedata[ctx.global_shapedata['level'] == 'nuts2'].plot(ax=axes[idx], facecolor='none', linewidth=0.5, edgecolor='grey')
            ctx.global_shapedata[ctx.global_shapedata['level'] == 'nuts1'].plot(ax=axes[idx], facecolor='none', linewidth=1.0, edgecolor='black')
            ctx.global_shapedata[ctx.global_shapedata['level'] == 'nuts0'].plot(ax=axes[idx], facecolor='none', linewidth=1.5, edgecolor='black')
            
            # gdf.plot() highlight node - edge
            if highlight_node is not None:
                highlight = gdf[gdf[self.evaluator.id_col] == highlight_node]
                highlight.plot(ax=axes[idx], facecolor='none', linewidth=2, edgecolor='darkgreen')

            # basic axis layout
            axes[idx].set_title(model_name)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])

        # --- colorbar in the last (empty) axis ---
        last_ax = axes[-1]

        sm      = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])                    # required boilerplate

        cbar = fig.colorbar(
            sm,
            ax      = last_ax,                     # anchor to the empty axis
            fraction= 0.5,                   # how much of the axis the cbar fills
            shrink  = 0.8,
            pad     = 0.05 
        )  

        cbar_title = f"{metric} [log]" if log else f"{metric}"
        
        cbar.set_label(f'{cbar_title}', fontsize=12)

        if highlight_node is not None:
            handle = mpatches.Patch(facecolor='none', edgecolor='darkgreen', 
                                    linewidth=1, label=f'Node {highlight_node}')
            last_ax.legend(handles=[handle], loc='lower center', frameon=False, fontsize=9)
        last_ax.set_visible(False)  # hides frame/ticks but legend still renders     
        fig.suptitle(f'{metric} distribution',fontweight = 'bold')

        plt.tight_layout()
        return fig

    def _plot_metric_best_map(self,
                              ax: Axes, 
                              metric_df: pd.DataFrame, 
                              metric: str, 
                              model_name_colors: dict,
                              vmin: float, 
                              vmax: float,
                              reverse_scale: bool = False,
                              highlight_node: Optional[int] = None,
                              legend: bool = True,
                              margin_fraction: Optional[float] = None,   # set None to disable      
                            ):
        """plots map with best model per geographical unit onto supplied axis"""
        df = metric_df.copy()
        model_cols   = list(df['model'].unique())
        metrics_df_l = df[[self.evaluator.id_col, 'model', metric]]
        metrics_df_w = (metrics_df_l
                        .pivot_table(index=self.evaluator.id_col, columns='model', values=metric)
                        .reset_index())

        # Best model per node: if reverse_scale, the lowest value is best. else max is best
        if reverse_scale:
            metrics_df_w['best'] = metrics_df_w[model_cols].idxmin(axis=1)
        else:
            metrics_df_w['best'] = metrics_df_w[model_cols].idxmax(axis=1)   

        # Margin masking: gap between top-2 scores
        if margin_fraction is not None:
            sorted_vals         = np.sort(metrics_df_w[model_cols].values, axis=1)
            # get gap: difference between best and second best
            if reverse_scale:
                gap = sorted_vals[:, 1] - sorted_vals[:, 0]
            else:
                gap                 = sorted_vals[:, -1] - sorted_vals[:, -2]       

            metric_range        = vmax - vmin
            threshold           = metric_range * margin_fraction            
            metrics_df_w['best']= np.where(gap < threshold, 'uncertain', metrics_df_w['best'])

        # Map color column
        color_map               = {**model_name_colors, 'uncertain': "#ffffff"}
        metrics_df_w['color']   = metrics_df_w['best'].map(color_map)

        # Merge with geodata
        model0   = list(self.evaluator.evaluated_models.values())[0]
        ctx      = model0.dataloadermanager.dataorchestrator.data_context
        gdf      = ctx.local_shapedata
        map_data = gpd.GeoDataFrame(pd.merge(gdf, metrics_df_w, on=self.evaluator.id_col))

        map_data.plot(ax        = ax, 
                      color     = map_data['color'], 
                      linewidth = 0, 
                      edgecolor ='white')
            
        ctx.global_shapedata[ctx.global_shapedata['level'] == 'nuts2'].plot(ax=ax, facecolor='none', linewidth=0.5, edgecolor='black')
        ctx.global_shapedata[ctx.global_shapedata['level'] == 'nuts1'].plot(ax=ax, facecolor='none', linewidth=1.0, edgecolor='black')
        ctx.global_shapedata[ctx.global_shapedata['level'] == 'nuts0'].plot(ax=ax, facecolor='none', linewidth=1.5, edgecolor='black')

        # Legend
        if legend:
            # model patches
            handles = [mpatches.Patch(color=c, label=m, ec='black',lw = 0.5) for m, c in color_map.items()]

            if highlight_node is not None:
                # single red patch for highlight-node
                handles.append(mpatches.Patch(color='red', label=f'node {highlight_node}', ec = 'black', lw = 0.5))
                highlight = map_data[map_data[self.evaluator.id_col] == highlight_node]
                highlight.plot(ax           = ax,
                               facecolor    = 'none',
                               linewidth    = 2,
                               edgecolor    = 'red')

            ax.legend(handles=handles, title='Best model', loc='best', fontsize=8)

        ax.set_xticks([])
        ax.set_yticks([])        
        ax.set_title(f'Best model by {metric}', fontweight='bold')

    def _validate_metric(self, metric: str) -> None:
        """validate that metric is a function of the metric- calculator - class"""
        if metric not in self.evaluator.metric_calculator.supported_metrics:
            raise MetricError(f'invalid metric {metric}. Supported metrics are {self.evaluator.metric_calculator.supported_metrics}')