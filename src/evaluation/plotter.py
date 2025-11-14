from typing import Literal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import geopandas as gpd
import matplotlib.patches as mpatches

from ..plotting.format import calculate_subplot_layout


class EvaluationPlotter:

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def plot_metric(self, 
                    metric: str, 
                    plot_type: Literal['violin', 'box', 'map'] = 'violin', 
                    horizon: int = 0):
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
        """
        horizon_dataset = f'horizon_{horizon}'
        
        # Get colors
        model_class_colors = {m.model_class: m.model_color for m in self.evaluator.evaluated_models}
        model_name_colors = {m.name: m.model_color for m in self.evaluator.evaluated_models}
        
        # Prepare data
        metric_df = self.evaluator.evaluation_entries[horizon_dataset][metric]
        metric_df_long = pd.melt(
            metric_df, 
            id_vars='node', 
            value_vars=list(model_name_colors.keys()),
            var_name='model', 
            value_name='value'
        )
        
        if plot_type in ['violin', 'box']:
            self._plot_distribution(metric_df_long, metric, plot_type, 
                                   model_name_colors, model_class_colors)
        elif plot_type == 'map':
            self._plot_map(metric_df_long, metric, model_name_colors)
        else:
            raise ValueError("plot_type must be 'violin', 'box', or 'map'")
        
    def _plot_distribution(self, df: pd.DataFrame, metric: str, plot_type: str,
                          model_name_colors: dict, model_class_colors: dict):
        """Plot violin or box plot."""
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
        
        # Legend
        handles = [mpatches.Patch(color=c) for c in model_class_colors.values()]
        ax.legend(handles, model_class_colors.keys(), title='Model Class', loc='best')
        
        plt.tight_layout()

    def _plot_map(self, metric_df: pd.DataFrame, metric: str, model_name_colors: dict):
        """Plot spatial map of metric values."""
        shapedata = self.evaluator.evaluated_models[0].dataloader.data['context']['shapedata']
        
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
