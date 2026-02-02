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

from src.utils import testcolor
from src.plotting import convert_managedfigure, ManagedFigure, calculate_subplot_layout

if TYPE_CHECKING:
    from src.evluation.evaluator import Evaluator  # Only imported for type checking, not at runtime


class EvaluationPlotter:

    def __init__(self, evaluator: 'Evaluator'):
        self.evaluator = evaluator

    @convert_managedfigure
    def plot_timeseries(self, nodes: Union[int, str, List[int], List[str]], horizon: int = 0, dataset: Literal['train','val','test'] = 'test') -> ManagedFigure:

        horizon = f'horizon_{horizon}'
        # get nodes in right format
        if isinstance(nodes, int):
            nodes = [nodes]
        elif isinstance(nodes, str):
            nodes = [int(nodes)]            

        nodes = [int(node) for node in nodes]

        nrows, ncols, figsize = calculate_subplot_layout(len(nodes), target_width=9, target_height=6)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if len(nodes) > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        # get model colors and linestyles
        models_to_plot          = self.evaluator.evaluated_models
        models_plotting_info    = {model.name : {"color": model.model_color} for model in models_to_plot.values()}

        # Iterate through models and assign linestyles based on color duplication
        for model in models_to_plot.values():
            if any(info["color"] == model.model_color for info in models_plotting_info.values()):
                models_plotting_info[model.name]['linestyle'] = '--'  # Example: dashed line for duplicate color
            else:
                models_plotting_info[model.name]['linestyle'] = '-'  # Example: solid line for unique color

        compilation_predictions = self.evaluator.prediction_compilations.get_compilation(horizon, dataset)

        for idx, node_id in enumerate(nodes):  
            node_data = compilation_predictions[compilation_predictions['node'] == node_id]
            ax        = axes[idx]

            
            for modelname, plotting_info in models_plotting_info.items():
                sns.lineplot(node_data, x = 'timestamp', y = f'pred_{modelname}', color = plotting_info['color'],  linestyle = plotting_info['linestyle'],  marker = '.', label  = f'{modelname}', ax = ax)      
            sns.lineplot(node_data, x = 'timestamp', y = 'target',                  color = testcolor,  label = 'target', marker = 'o',                  ax = ax)
            ax.grid()
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(f'node: {node_id}')     
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # 2-month interval
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))        

        plt.tight_layout()
        plt.close()
        return fig

    @convert_managedfigure
    def plot_metric_compilation(self, dataset: Literal['train','val','test'] = 'test', horizon: int = 0, metrics: List[str] = ['pearson_corr','spearman_corr','ccc','rmse'], plot_types: List[str]=['box','box','box','box'], subfig_size: Tuple[float,float] = (8,6)) -> ManagedFigure:
        """
        Create a compilation of multiple metrics in subplots.
        """
        horizon_name = f'horizon_{horizon}'
        n_subfigures = len(metrics)

        nrow, ncol, figsize = calculate_subplot_layout(n_subfigures, subfig_size[0], subfig_size[1])
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize)
        axes = axes.flatten()

        # Get colors
        model_class_colors = {ml.model_class: ml.model_color for ml in self.evaluator.evaluated_models.values()}
        model_name_colors  = {ml.name: ml.model_color  for ml in self.evaluator.evaluated_models.values()}
        
        for ii, metric in enumerate(metrics):
            ax          = axes[ii]
            plot_type   = plot_types[ii]

            # Prepare data
            metric_df = self.evaluator.metric_compilations.get_metric(horizon_name, dataset, metric)
            metric_df_long = pd.melt(
                metric_df, 
                id_vars='node', 
                value_vars=list(model_name_colors.keys()),
                var_name='model', 
                value_name='value'
            )
        
            if plot_type in ['violin', 'box']:
                self._plot_distribution_on_ax(
                    ax, metric_df_long, metric, plot_type, 
                    model_name_colors, model_class_colors,
                    highlight_node=None,
                    add_legend=(ii == 0)  # Only add legend to first subplot
                )
            else:
                raise ValueError("plot_type must be 'violin' or 'box'")
        
        # Hide extra subplots
        for jj in range(len(metrics), len(axes)):
            axes[jj].axis('off')
        
        plt.tight_layout()
        plt.close()
        return fig

    @convert_managedfigure
    def plot_single_metric(self, 
                    metric:         str, 
                    horizon:        int,
                    dataset:        str = 'test',
                    plot_type:      Literal['violin', 'box', 'map'] = 'violin', 
                    highlight_node: Optional[float] = None) -> ManagedFigure:
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
        horizon = f'horizon_{horizon}'

        # Get colors
        model_class_colors = {ml.model_class: ml.model_color for ml in self.evaluator.evaluated_models.values()}
        model_name_colors  = {ml.name: ml.model_color  for ml in self.evaluator.evaluated_models.values()}
        
        # Prepare data
        metric_df = self.evaluator.metric_compilations.get_metric(horizon, dataset, metric)
        metric_df_long = pd.melt(
            metric_df, 
            id_vars='node', 
            value_vars=list(model_name_colors.keys()),
            var_name='model', 
            value_name='value'
        )
        
        if plot_type in ['violin', 'box']:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            self._plot_distribution_on_ax(
                ax, metric_df_long, metric, plot_type, 
                model_name_colors, model_class_colors,
                highlight_node, add_legend=True
            )
            plt.tight_layout()
        elif plot_type == 'map':
            fig = self._plot_map(metric_df_long, metric, model_name_colors)
        else:
            raise ValueError("plot_type must be 'violin', 'box', or 'map'")
        
        plt.close()
        return fig

    def _plot_distribution_on_ax(self, ax: Axes, df: pd.DataFrame, metric: str, plot_type: str,
                          model_name_colors: dict, model_class_colors: dict,
                          highlight_node: Optional[int] = None,
                          add_legend: bool = True) -> None:
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
        plot_func(
            data=df, x='model', y='value', hue='model',
            ax=ax, palette=model_name_colors,
            **(dict(cut=0) if plot_type == 'violin' else {}),
            legend=False
        )
        
        ax.set_title(f'{metric.upper()}')
        ax.set_ylabel(metric.upper())
        ax.set_xlabel('Model')
        ax.grid(alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='center', fontsize=8)

        if highlight_node:
            # Plot red dot for the specified node for all models
            node_values = df[df['node'] == highlight_node]  # Get values for the specified node
            
            # Loop over each model and plot a red dot for the specified node
            for i, model in enumerate(model_name_colors.keys()):
                # Find the corresponding value for the node for each model
                node_value = node_values[node_values['model'] == model]['value'].values
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

    def _plot_distribution(self, df: pd.DataFrame, metric: str, plot_type: str,
                          model_name_colors: dict, model_class_colors: dict,
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

    def _plot_map(self, metric_df: pd.DataFrame, metric: str, model_name_colors: dict):
        """Plot spatial map of metric values."""
        shapedata = list(self.evaluator.evaluated_models.values())[0].dataloadermanager.dataorchestrator.data_context.shapedata
        
        # Calculate layout
        n_models = len(self.evaluator.evaluated_models)
        nrows, ncols, figsize = calculate_subplot_layout(n_models, target_width=8, target_height=6)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten()
        
        # Merge with shapefile
        merged_data = gpd.GeoDataFrame(pd.merge(metric_df, shapedata, on='node'))
        merged_data['value'] = pd.to_numeric(merged_data['value'], errors='coerce')
        
        # Set color scale
        if metric in ['ccc', 'corr', 'lag_corr', 'neighborhood_ccc']:
            delta = 0.001
            vmin, vmax = 0 - delta, 1 + delta
        else:
            vmin, vmax = merged_data['value'].min(), merged_data['value'].max()

        # Plot each model
        for ii, model_name in enumerate(merged_data['model'].unique()):
            df = merged_data[merged_data['model'] == model_name]
            df.plot(column='value', cmap='Blues', legend=False, edgecolor='black', linewidth=0.5,
                        missing_kwds={
                            'color': 'lightgrey',
                            'label': 'No Data'
                        },
                   vmax=vmax, vmin=vmin, ax=axes[ii])
            axes[ii].set_title(model_name)
            axes[ii].axis('off')
        
        # Hide extra subplots
        for jj in range(ii + 1, len(axes)):
            axes[jj].axis('off')
        
        # Add colorbar to the right of ALL subplots
        sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        
        # Create colorbar with better positioning
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', 
                           fraction=0.046, pad=0.04, aspect=50, shrink = 0.6)
        
        # Set colorbar label
        cbar.set_label(metric, rotation=90, labelpad=20)
        
        # Set specific ticks at 0 and 1 (and optionally 0.5)
        tickrange = vmax - vmin
        ticks = [0, 0.25 * tickrange, 0.5 * tickrange, 0.75 * tickrange, tickrange]
        ticks = [vmin + t for t in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{t:.1f}' for t in ticks])
        
        plt.suptitle(f'{metric}')
        return fig

    def plot_metric_horizons_separate(self,
                                    metric: str,
                                    horizons: list[int],
                                    dataset: str = 'test',
                                    plot_type: Literal['violin', 'box'] = 'violin',
                                    highlight_node: Optional[int] = None):
        """
        Plot specified metric with separate subplots for each horizon.
        Allows easier comparison of model performance within each horizon.
        
        Parameters
        -----------
        metric : str
            Metric name (corr, mse, rmse, ccc, lag_corr, neighbor_corr, spatial_autocorr)
        horizons : list[int]
            List of horizons to compare (e.g., [0, 1, 2])
        dataset : str
            Which dataset to evaluate on ('test', 'val', 'train')
        plot_type : str
            Type of plot ('violin' or 'box')
        highlight_node : Optional[int]
            If provided, highlights this specific node across all plots
        """
        # Get colors
        model_name_colors = {ml.name: ml.model_color for ml in self.evaluator.evaluated_models.values()}
        model_class_colors = {ml.model_class: ml.model_color for ml in self.evaluator.evaluated_models.values()}
        
        horizons_str = [f'horizon_{h}' for h in horizons]

        # Calculate layout
        n_horizons = len(horizons)
        nrows = n_horizons
        ncols = 1
        figsize = (7 * (ncols+1), 5 * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
        if n_horizons == 1:
            axes = [axes]
        
        # Plot each horizon
        for idx, horizon in enumerate(horizons):
            ax = axes[idx]
            
            # Get data for this horizon
            metric_df = self.evaluator.metric_compilations.get_metric(horizons_str[idx], dataset, metric)
            metric_df_long = pd.melt(
                metric_df,
                id_vars='node',
                value_vars=list(model_name_colors.keys()),
                var_name='model',
                value_name='value'
            )
            
            # Use the new axes-based plotting function
            self._plot_distribution_on_ax(
                ax, metric_df_long, metric, plot_type,
                model_name_colors, model_class_colors,
                highlight_node=highlight_node,
                add_legend=False  # We'll add a single legend for the whole figure
            )
            
            # Styling
            ax.set_title(f'Horizon {horizon}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Model', fontsize=10)
            if idx == 0:
                ax.set_ylabel(metric.upper(), fontsize=10)
            else:
                ax.set_ylabel('')
            
            # Add node-specific legend only to first subplot if node is highlighted
            if highlight_node is not None and idx == 0:
                # Add a small legend for the highlighted node
                node_patch = mpatches.Patch(color='red', label=f'Node {highlight_node}')
                ax.legend(handles=[node_patch], loc='upper right', frameon=True, fontsize=8)
        
        # Add main title
        fig.suptitle(f'Model Evaluation Across Horizons: {metric.upper()}', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        # Add legend for model classes
        handles = [mpatches.Patch(color=c) for c in model_class_colors.values()]
        fig.legend(handles, model_class_colors.keys(), 
                title='Model Class', loc='upper right', 
                bbox_to_anchor=(0.99, 0.98), frameon=True)
        
        plt.tight_layout()
        return fig