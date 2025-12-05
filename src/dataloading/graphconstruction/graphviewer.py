import pandas as pd
import geopandas as gpd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Union, TYPE_CHECKING
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator

# import not at runtime, prevents circular import
if TYPE_CHECKING: 
    from .graphconstruction_orchestrator import GraphStructure, GraphRegistry

class GraphViewer:
    """
    Visualizes graph structures overlaid on geographical shapes.
    Heavy lifting is done here, documentation found in GraphOrchestrator
    
    Parameters
    ----------
    graph_entry: 'GraphEntry'
        The entry in the attribute .graph_registry
    """
    
    def __init__(self, graph_registry: 'GraphRegistry', shapefile: gpd.GeoDataFrame, figsize: Tuple[float,float] = (14,9)):
        self.graph_registry = graph_registry
        self.shapes         = shapefile
        self.palette_blues  = sns.color_palette("Blues", n_colors=100)
        self.palette_reds   = sns.color_palette("Reds", n_colors=100)
        self.figsize        = figsize

    def view(self, graphname: str, node_idx: Optional[int] = None, subplots: bool = True, title: Optional[str] = None):
        """
        """
        if not title:
            title = ""

        if graphname == 'empty':
            return self._create_empty_view()
        
        graph_structure = self.graph_registry.get_entry(graphname).structure
        if graph_structure is None:
            return

        if node_idx is None:
            return self._create_global_view(graph_structure, subplots, title, graphname)
        
        return self._create_node_view(graph_structure, int(node_idx), subplots, title, graphname)

    # ============ Main View Creators ============
    
    def _create_empty_view(self) -> Figure:
        """
        Create view showing only node contours and centroids
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        self._plot_shape(ax, self.shapes)
        self._plot_centroids(ax, self.shapes, self.palette_blues[10:], size=7)
        self._style_axis(ax, title="Germany")
        plt.tight_layout()
        plt.close(fig)
        return fig

    def _create_global_view(self, graph_structure: 'GraphStructure', subplots: bool, title: str, graphname: str) -> Figure:
        """Create view showing all graph connections with statistics."""
        
        if not subplots:
            fig, ax_main = plt.subplots(1, 1, figsize=self.figsize)

        else:
            fig     = plt.figure(figsize=self.figsize)
            gs      = fig.add_gridspec(3, 2, width_ratios=[3, 1], height_ratios=[1, 1, 1])
            ax_main = fig.add_subplot(gs[:, 0])
            # Distribution histograms
            ax_ori = fig.add_subplot(gs[0, 1])
            self._plot_connection_histogram(ax_ori, graph_structure.edge_index[0])
            self._style_axis(ax_ori, title=f"Origin connections {graphname}", 
                            xlabel='number of connections', ylabel='frequency', show_spines=True, title_fontsize=10, label_fontsize=8)
            
            ax_dst = fig.add_subplot(gs[1, 1])
            self._plot_connection_histogram(ax_dst, graph_structure.edge_index[1])
            self._style_axis(ax_dst, title=f"Destination connections {graphname}", 
                            xlabel='number of connections', ylabel='frequency', show_spines=True, title_fontsize=10, label_fontsize=8)
            
            ax_weights = fig.add_subplot(gs[2, 1])
            self._plot_weight_histogram(ax_weights, graph_structure.edge_weight)
            self._style_axis(ax_weights, title=f"Edge weights {graphname}", 
                            xlabel='weight', ylabel='frequency', show_spines=True, title_fontsize=10, label_fontsize=8)            
        
        # Main map with all connections
        self._plot_shape(ax_main, self.shapes)
        self._plot_connection_lines(ax_main, graph_structure.edge_index, self.shapes)
        self._plot_centroids(ax_main, self.shapes, self.palette_blues[70:71], size=4)
        self._style_axis(ax_main, title=title)
        
        plt.tight_layout()
        plt.close(fig)
        return fig

    def _create_node_view(self, graph_structure: 'GraphStructure', node_idx: int, subplots: bool, title: str, graphname: str) -> Figure:
        """Create view focused on a specific node and its neighborhood."""
        # Extract neighborhood data
        neighborhood = self._get_neighborhood(graph_structure, node_idx)
        
        if not subplots:
            return self._create_simple_node_view(neighborhood, title)
        
        return self._create_detailed_node_view(graph_structure, node_idx, neighborhood, title, graphname)

    def _create_simple_node_view(self, neighborhood: dict, title: str) -> Figure:
        """Create simple map view of node neighborhood."""
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        self._plot_neighborhood_map(ax, neighborhood)
        self._style_axis(ax, title=title)
        plt.tight_layout()
        plt.close(fig)
        return fig

    def _create_detailed_node_view(self, graph_structure: 'GraphStructure', node_idx: int, neighborhood: dict, title: str, graphname: str) -> Figure:
        """Create detailed view with map and statistical subplots."""
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
        
        # Global map showing neighborhood
        ax_global = fig.add_subplot(gs[:, 0])
        self._plot_neighborhood_map(ax_global, neighborhood)
        self._style_axis(ax_global, title=title)
        
        # Weight distribution
        ax_hist = fig.add_subplot(gs[1, 1])
        max_weight = graph_structure.edge_weight.max().item()
        self._plot_weight_histogram(ax_hist, neighborhood['edge_weights'], 
                                    limits=(0, max_weight))
        ax_hist.set_xlim(0, max_weight)
        self._style_axis(ax_hist, title='Local edge weights', 
                        xlabel='weight', ylabel='frequency', show_spines=True, title_fontsize=10, label_fontsize=8)
        
        # Zoomed local map with colored weights
        ax_local = fig.add_subplot(gs[0, 1])
        self._plot_shape(ax_local, self.shapes) # background
        self._plot_local_weighted_map(graph_structure, ax_local, node_idx, neighborhood, max_weight)
        
        plt.tight_layout()
        plt.close(fig)
        return fig

    # ============ Data Extraction ============
    
    def _get_neighborhood(self, graph_structure: 'GraphStructure', node_idx: int) -> dict:
        """Extract all data related to a node's neighborhood."""
        mask = graph_structure.edge_index[0] == node_idx
        
        neighbor_nodes = graph_structure.edge_index[1][mask]
        all_nodes = torch.cat([torch.tensor([node_idx]), neighbor_nodes]).unique()
        
        # Create local index mapping
        node_mapping = {global_idx: local_idx 
                       for local_idx, global_idx in enumerate(all_nodes.tolist())}
        
        # Remap edge indices to local coordinates
        local_edges = graph_structure.edge_index[:, mask].clone()
        local_edges[0] = torch.tensor([node_mapping[idx.item()] 
                                       for idx in local_edges[0]])
        local_edges[1] = torch.tensor([node_mapping[idx.item()] 
                                       for idx in local_edges[1]])
        
        connected_nodes = torch.unique(graph_structure.edge_index[:, mask])
        
        return {
            'node_idx': node_idx,
            'edge_weights': graph_structure.edge_weight[mask],
            'local_edge_index': local_edges,
            'connected_nodes': connected_nodes,
            'shapes': self.shapes[self.shapes['node'].isin(connected_nodes.numpy())].reset_index(drop=True),
            'node_shape': self.shapes[self.shapes['node'] == node_idx]
        }

    # ============ Plotting Components ============
    
    def _plot_neighborhood_map(self, ax: Axes, neighborhood: dict):
        """Plot the main neighborhood map."""
        self._plot_shape(ax, self.shapes)
        self._plot_shape(ax, neighborhood['shapes'], linewidth=1)
        self._plot_shape(ax, neighborhood['node_shape'], linewidth=1, 
                        color='darkblue', alpha=0.7)
        self._plot_connection_lines(ax, neighborhood['local_edge_index'], 
                                   neighborhood['shapes'])
        self._plot_centroids(ax, neighborhood['shapes'], 
                           self.palette_blues[70:71], size=5)
        self._style_axis(ax)

    def _plot_local_weighted_map(self, graph_structure: 'GraphStructure', ax: Axes, node_idx: int, 
                                 neighborhood: dict, max_weight: float):
        """Plot zoomed map with weight-colored nodes."""
        shapes = neighborhood['shapes']
        bounds = shapes.total_bounds
        margin = 0.1 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        
        ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
        ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
        
        # Merge weight data with shapes
        weight_df = pd.DataFrame({
            'node': graph_structure.edge_index[1][graph_structure.edge_index[0] == node_idx].numpy(),
            'weight': neighborhood['edge_weights'].numpy()
        })
        shapes_with_weights = gpd.GeoDataFrame(
            pd.merge(shapes, weight_df, on='node')
        )
        
        self._plot_centroids(ax, neighborhood['node_shape'], 
                           self.palette_blues[75:76], size=5)
        shapes_with_weights.plot(column='weight', cmap='Reds', 
                                edgecolor='black', linewidth=1, ax=ax,
                                vmin=0, vmax=max_weight)
        self._style_axis(ax, title=f"Local graph - node {node_idx}", show_spines=True, show_ticks=False)

    def _plot_shape(self, ax: Axes, df: gpd.GeoDataFrame, color: str = 'lightgrey',
                   edgecolor: str = 'black', linewidth: float = 0.075, 
                   alpha: float = 1):
        """Plot geographical shapes."""
        df.plot(color=color, edgecolor=edgecolor, linewidth=linewidth, 
               alpha=alpha, ax=ax)

    def _plot_centroids(self, ax: Axes, df: pd.DataFrame, 
                       colorpalette: List, size: int = 7):
        """Plot centroids with random colors from palette."""
        centroids = df.geometry.centroid
        colors = [colorpalette[np.random.randint(0, len(colorpalette))] 
                 for _ in range(len(centroids))]
        
        for centroid, color in zip(centroids, colors):
            ax.plot(centroid.x, centroid.y, 'o', color=color, 
                   markersize=size, markeredgecolor='black', 
                   markeredgewidth=1)

    def _plot_connection_lines(self, ax: Axes, edge_list: torch.Tensor, 
                              df: pd.DataFrame):
        """Draw lines between connected nodes."""
        for i, j in zip(edge_list[0].tolist(), edge_list[1].tolist()):
            point_i = df.iloc[i]['geometry'].centroid
            point_j = df.iloc[j]['geometry'].centroid
            ax.plot([point_i.x, point_j.x], [point_i.y, point_j.y], 
                   color='black', linewidth=1.5, alpha=0.8)

    def _plot_connection_histogram(self, ax: Axes, edge_index: torch.Tensor):
        """Plot histogram of connection degrees."""
        degree_series = pd.Series(edge_index.tolist()).value_counts().sort_index()
        degree_counts = degree_series.value_counts().sort_index()

        ax.bar(degree_counts.index.to_numpy(), degree_counts.to_numpy(), 
            color='lightgray', edgecolor='black', width=1)
        
        min_deg, max_deg = degree_counts.index.min(), degree_counts.index.max()
        ax.set_xlim(min_deg - 1, max_deg + 1)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def _plot_weight_histogram(self, ax: Axes, weights: torch.Tensor, 
                              limits: Optional[Tuple[float, float]] = None):
        """Plot histogram of edge weights."""
        weight_array = weights.detach().cpu().numpy()
        bins = 15
        
        if limits:
            ax.hist(weight_array, bins=bins, range=limits, 
                   color='lightgray', edgecolor='black')
        else:
            ax.hist(weight_array, bins=bins, 
                   color='lightgray', edgecolor='black')
        
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def _style_axis(self, ax: Axes, title: Optional[str] = None,
                    xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                    show_spines: bool = False, show_ticks: bool = True, title_fontsize: float = 12.0, label_fontsize: float = 10.0):
        """Apply consistent styling to axes."""
        if not show_spines:
            ax.axis('off')
        else:
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=label_fontsize)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=label_fontsize)
            if not show_ticks:
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        
        if title:
            ax.set_title(title, fontsize=title_fontsize)