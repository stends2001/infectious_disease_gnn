import matplotlib.pyplot as plt 
import seaborn as sns
from typing import Literal, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.plotting.format import calculate_subplot_layout
import geopandas as gpd
import matplotlib.patches as mpatches

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluator import Evaluator  # Only imported for type checking, not at runtime


class EvaluationPlotter:

    def __init__(self, evaluator: 'Evaluator'):
        self.evaluator = evaluator

    def plot_metric(self, 
                    metric:         str, 
                    horizon:        str,
                    dataset:        str = 'test',
                    plot_type:      Literal['violin', 'box', 'map'] = 'violin', 
                    highlight_node: Optional[float] = None):
        """
        Plot specified metric across models.
        
        Parameters:
        -----------
        metric : str
            Metric name (corr, mse, rmse, ccc, lag_corr, neighbor_corr, spatial_autocorr)
        horizon : int
            Which horizon to plot
        plot_type : str
            Type of plot (violin, box, map)
        highlight_value : Optional[float]
            If provided, a red dot will be placed at this value on each distribution.
        """       
        # Get colors
        model_class_colors = {ml.model_class: ml.model_color for ml in self.evaluator.evaluated_models.values()}
        model_name_colors  = {ml.clean_name: ml.model_color  for ml in self.evaluator.evaluated_models.values()}
        
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
            self._plot_distribution(
                metric_df_long, metric, plot_type, 
                model_name_colors, model_class_colors,
                highlight_node
            )
        elif plot_type == 'map':
            self._plot_map(metric_df_long, metric, model_name_colors)
        else:
            raise ValueError("plot_type must be 'violin', 'box', or 'map'")

    def _plot_distribution(self, df: pd.DataFrame, metric: str, plot_type: str,
                          model_name_colors: dict, model_class_colors: dict,
                          node: Optional[int]):
        """Plot violin or box plot and optionally add a red dot for the specified node."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        
        plot_func = sns.violinplot if plot_type == 'violin' else sns.boxplot
        plot_func(
            data=df, x='model', y='value', hue='model',
            ax=ax, palette=model_name_colors,
            **(dict(cut=0) if plot_type == 'violin' else {}),
            legend=False
        )
        
        ax.set_title(f'Model Evaluation: {metric.upper()}')
        ax.set_ylabel(metric.upper())
        ax.set_xlabel('Model')
        ax.grid(alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='center', fontsize=8)

        if node:
        
            # Plot red dot for the specified node for all models
            node_values = df[df['node'] == node]  # Get values for the specified node
            
            # Loop over each model and plot a red dot for the specified node
            for i, model in enumerate(model_name_colors.keys()):
                # Find the corresponding value for the node for each model
                node_value = node_values[node_values['model'] == model]['value'].values[0]
                
                # Plot red dot at the value of the node for the current model
                ax.scatter(
                    x=i,  # i is the x-position of the model in the distribution
                    y=node_value,  # the y-position is the value for the specified node
                    color='red', 
                    zorder=10, 
                    s=100, 
                    label=f'Node {node}: {node_value}' if i == 0 else ""  # Label only the first red dot for clarity
                )

        # Legend
        handles = [mpatches.Patch(color=c) for c in model_class_colors.values()]
        ax.legend(handles, model_class_colors.keys(), title='Model Class', loc='best')
        
        plt.tight_layout()

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
