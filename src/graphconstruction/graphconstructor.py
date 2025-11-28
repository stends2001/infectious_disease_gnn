import os
from typing import Optional, Tuple, List, Union, Literal, Dict
from dataclasses import dataclass, asdict
from matplotlib.figure import Figure
from ..utils.textformatting import checkmark
from shapely.geometry import LineString
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
import torch
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from ..dataloading.epidataloader import EpiDataLoader
from ..dataloading.deepdataloader import DeepDataLoader

from ..graphconstruction.graphconstructor_edgeweightnormalizer import GraphEdgeWeightNormalizer
from ..graphconstruction.graphconstructor_generation import GraphGeneration
from ..graphconstruction.graphconstructor_selfloops import GraphAddSelfLoops

@dataclass
class GraphConfig:
    method:         str
    name_addition:  Optional[str]
    self_connection:str
    scaling_method: Optional[str]
    kwargs:         dict

palette_blues = sns.color_palette("Blues", n_colors=100)
palette_reds  = sns.color_palette("Reds", n_colors=100)


class GraphConstructor:

    """
    calculates and saves graphs in edges and weights
    extensive preview-functionality.

    graphs are saved into `self.graph_registry`

    Parameters:
    ----------
    epidata: EpiDataLoader
        Data object, should have tokenized nodes and when applicable, population data
    id_col: str = 'node'
        Name of the column associated with the node-id
    graph_dir: str = data/graphs/
        directory in which graphs will be saved. Internally, the epidata.nuts_level will be the subfolder
        in this graph_dir folder in which the graph will be saved.

    Examples:
    --------
    >>> epiloader = EpiDataLoader(disease_name, get_data_env(), nuts_level=nuts_level, min_date=min_date,max_date=max_date, include_population=False, horizon_size = horizon_size, horizon_leadtime = horizon_leadtime, sequence_length=sequence_length, split_berlin=split_berlin)
    >>> epiloader.add_time_features()
    >>> epiloader.log_transform_target()
    >>> epiloader.set_splits(split_trainval, split_valtest)
    >>> epiloader.normalize()
    >>> epiloader.add_lagged_features(lags = lags)
    >>> epiloader.finalize()

    >>> epidata_loader_basis = DeepDataLoader(disease_name, get_data_env(), nuts_level=nuts_level, min_date=min_date,max_date=max_date, include_population=False, horizon_size = horizon_size, horizon_leadtime = horizon_leadtime, sequence_length=sequence_length, split_berlin=split_berlin)
    >>> epidata_loader_basis.add_time_features()
    >>> epidata_loader_basis.log_transform_target()
    >>> epidata_loader_basis.set_splits(split_trainval, split_valtest)
    >>> epidata_loader_basis.normalize()
    >>> epidata_loader_basis.add_lagged_features(lags = lags)
    >>> epidata_loader_basis.finalize()


    # initating class
    >>> graphconstruction = GraphConstructor(epidata = epidata_loader_bn)

    # generate boolean neighbors - graph
    >>> graphconstruction.generate_graph(
    >>>     method='boolean_neighbors', 
    >>>     name_addition='',
    >>> )

    # generate commuter-based graphs
    >>> graphconstruction.generate_graph(
    >>>     method='commuter', 
    >>>     self_connection='mean',
    >>>     scaling_method='symmetric',
    >>>     name_addition='t2000',
    >>>     commuter_type = 'static',
    >>>     commuting_threshold = 2000
    >>> )

    # preview graphs
    # preview local graph
    >>> graphconstruction.preview_graph(graphname='commuter_t2000_selfmean_symmetric', node_idx= 15)
    # preview global graph
    >>> graphconstruction.preview_graph(graphname='commuter_t2000_selfmean_symmetric', node_idx= 'all')

    # save graphs
    >>> graphconstruction.save_graph('all')

    """

    def __init__(self,                  
                 epidata:         EpiDataLoader,
                 id_col:          str = 'node',
                 graph_dir:       str = "data/graphs/"):
        
        # extract metadata
        shapes               = epidata.data['context']['shapedata']    
        epidemiological_data = epidata.data['context']['epidemiological_data']
        self.tokens          = epidata.tokens['id_idx']
        
        # mean population data
        self.population_data = epidemiological_data.groupby(id_col)['population_size'].mean().reset_index()
        self.shapes          = shapes.copy()
        self.id_col          = id_col 
        self.nuts_level      = epidata.nuts_level
        self.graph_dir       = os.path.join(graph_dir, f'{self.nuts_level}')

        # registry of graphs
        self.graph_registry = {}
        os.makedirs(self.graph_dir, exist_ok=True)

        self.graph_methods          = ['boolean_neighbors', 
                                       'identity', 
                                       'mesh', 
                                       'distance_threshold',
                                       'k_nearest', 
                                       'population_weighted', 
                                       'gravity_model', 
                                       'commuter']
        
        self.num_nodes              = epidata.data['context']['epidemiological_data'][self.id_col].nunique()
        
    def generate_graph(self, 
                       method: str                                         = 'boolean_neighbors',
                       name_addition:   Optional[str]                      = None,
                       self_connection:  Literal['max','0','mean']          = 'mean',
                       scaling_method:  Optional[Literal['minmax','log','zscore','symmetric','rowwise']] = None,
                       **kwargs) -> None:
        """
        Generates a graph structure based on the method. Depending on the method, additional kwargs may be required.
        A graph structure and config are created and saved into the dictionary `self.graph_registry` under the key
        corresponding to `graph_name`, which is equal to:

            method + name_addition + self{self_connection} + scaling_method

        where '_' is used as separator

    
        Parameters:
        ----------
        method: str 

        name_addition: Optional[str]

        self_connection: Literal['max','0','mean']

        scaling_method: Optional[]

        kwargs

        See also:
        --------
        The heavy lifting is done through the following classes. Each of these contains further information.
            - GraphGeneration
            - GraphEdgeWeightNormalizer
            - GraphAddSelfLoops
        """

        if method not in self.graph_methods:
            raise ValueError(f'{method} not a valid graph method. Please choose a method from this list:\n{self.graph_methods}')

        graphconfig = GraphConfig(
            method          = method,
            name_addition   = name_addition,
            self_connection = self_connection,
            scaling_method  = scaling_method,
            kwargs          = kwargs
            )

        
        graphname = f'{method}_{name_addition}'         if name_addition    else f'{method}'
        graphname = f'{graphname}_self{self_connection}'if self_connection  else f'{graphname}'
        graphname = f'{graphname}_{scaling_method}'     if scaling_method   else f'{graphname}'

        # Ensure IDs are integers and no missing
        shapes_cp               = self.shapes.dropna(subset=[self.id_col])
        shapes_cp[self.id_col]  = shapes_cp[self.id_col].astype(int)
        node_ids                = shapes_cp[self.id_col].dropna().astype(int).values

        ##########################
        ##### Create Graphs ######
        ##########################     
        graph_generator = GraphGeneration(
            gdf     = shapes_cp,
            tokens  = self.tokens,
            popdata = self.population_data,
            id_col  = self.id_col
        )

        # Generate the graph with whatever method and kwargs
        edges, weights = graph_generator.generate_graph(method=method, **kwargs)

        # if weights is undefined, give 1 everywhere
        if weights is None:
            weights = [1 for _ in edges]

        ##########################
        ##### Add self-loops #####
        ##########################        
        if method not in ['identity', 'mesh']:
            edges, weights = GraphAddSelfLoops(edge_indices=edges, edge_weights=weights, num_nodes = self.num_nodes, node_ids=node_ids).add_loops(self_connection)

        # remove zero valued loops        
        edges   = [edge for edge, weight in zip(edges, weights) if weight != 0]
        weights = [weight for weight in weights if weight != 0]   

        # transform into torch objects        
        edge_weight = torch.tensor(weights, dtype=torch.float)
        edge_index  = torch.tensor(edges, dtype=torch.long).t().contiguous()

        ##########################
        # Normalize edge-weights #
        ##########################
        if scaling_method:
           edge_weight = GraphEdgeWeightNormalizer(edge_indices=edge_index, edge_weights=edge_weight, num_nodes = self.num_nodes).normalize(scaling_method)

        # save config
        graphdict = {'structure': {'edge_index': edge_index, 'edge_weight': edge_weight},
                     'config'   : asdict(graphconfig),
                     'summary'  : self._get_graph_summary(edge_index,edge_weight)}   

        self.graph_registry[graphname] = graphdict
        
        print(f'{checkmark} graph generated: {graphname}')

    def preview_graph(self, 
                        graphname: str = 'empty',
                        node_idx: Union[int, str] = 'all',
                        map_only: bool = False) -> Figure:
            """ 
            Previews a graph structure.

            Parameters:
            ----------
            map_only: bool = False
                if True and node_idx is an integer, returns only the main map without subplots
            
            graphname: str = 'empty'
                the graphname under which the graph structure is self into `self.graph_registry`
                NOTE: the graphname is printed upon generation.
                NOTE: if graphname == 'empty', the emtpy graphstructure is shown, that is, the underlyng map with centroids.
            
            node_idx: Union[int, str] = 'all' 
                the node of which the neighborhood will be shown. when node_idx == 'all', the global graph is shown.
            """

            def plot_shape(ax: Axes, df: gpd.GeoDataFrame, color = 'lightgrey', edgecolor = 'black', linewidth = 0.075) -> Axes:
                df.plot(color=color, edgecolor=edgecolor, linewidth=linewidth, ax=ax)
                return ax
            
            def basic_plot_makeup(ax: Axes, 
                                title: Optional[str] = None, 
                                xlabel: Optional[str] = None, 
                                ylabel: Optional[str] = None, 
                                ticks: bool = True,
                                vspine: bool = True) -> Axes:
                if not vspine:
                    ax.axis('off')  # turns off everything (ticks, labels, box)
                else:
                    if not ticks:
                        ax.tick_params(
                            axis='both',
                            which='both',
                            bottom=False,
                            top=False,
                            left=False,
                            right=False,
                            labelbottom=False,
                            labelleft=False
                        )
                    if xlabel:
                        ax.set_xlabel(xlabel)
                    if ylabel:
                        ax.set_ylabel(ylabel)            

                if title:
                    ax.set_title(title)         

                return ax

            def plot_histogram_connection_degree(ax: Axes, edge_index, node_idx=None, self_color=None) -> Axes: 
                nodes = edge_index.tolist()

                degree_series = pd.Series(nodes).value_counts().sort_index()
                degree_counts = degree_series.value_counts().sort_index()

                bars = ax.bar(degree_counts.index, degree_counts.values, color='lightgray', edgecolor='black', width=1)

                # Set x-axis limits: one below min, one above max
                min_degree = degree_counts.index.min()
                max_degree = degree_counts.index.max()
                ax.set_xlim(min_degree - 1, max_degree + 1)

                if node_idx:
                    node_degree = degree_series.get(int(node_idx))
                    if node_degree in degree_counts.index:
                        idx = list(degree_counts.index).index(node_degree)
                        bars[idx].set_color(self_color)
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                return ax

            def plot_centroids(ax: Axes, df: pd.DataFrame, colorpalette: List, column: str, size: int = 7) -> Axes:
                centroids   = df.geometry.centroid
                n_centroids = len(centroids)

                if column == 'random':
                    colors = [colorpalette[np.random.randint(0, len(colorpalette))] for _ in range(n_centroids)]
                
                for centroid, color in zip(centroids, colors):
                    ax.plot(centroid.x, centroid.y, 'o', 
                            color=color, 
                            markersize=size, 
                            markeredgecolor='black', 
                            markeredgewidth=1)

                return ax
            
            def plot_connection_lines(ax: Axes, edge_list: torch.Tensor, df: pd.DataFrame) -> Axes:
                for i, j in zip(edge_list[0].tolist(), edge_list[1].tolist()):
                    point_i = df.iloc[i]['geometry'].centroid
                    point_j = df.iloc[j]['geometry'].centroid
                    
                    ax.plot([point_i.x, point_j.x], [point_i.y, point_j.y], 
                            color='black', linewidth=2, alpha = 0.8)
                return ax
            
            def plot_colored_nodes(ax: Axes, df: pd.DataFrame, column: str,vmin, vmax, colorpalette=None) -> Axes:
                cmap = colorpalette if colorpalette else None
                df.plot(column=column, cmap=cmap, edgecolor='black', linewidth=1, ax=ax, vmin = vmin, vmax = vmax)
                return ax

            def zoom_on_plot(ax: Axes, xlim: Optional[Tuple[float, float]] = None, ylim: Optional[Tuple[float, float]] = None) -> Axes:

                if xlim:
                    ax.set_xlim(xlim)

                if ylim:
                    ax.set_ylim(ylim)

                return ax

            def plot_selfloop(ax: Axes, df: pd.DataFrame, loop_radius : float = 2*10E3) -> Axes:
                geom = df.iloc[0].geometry
    

                centroid: Point = geom.centroid
                cx, cy = centroid.x, centroid.y

                # Define control points for the curve (from and to centroid, but offset slightly)
                start = (cx - loop_radius, cy)
                end = (cx + loop_radius, cy + loop_radius)  # very slightly offset so matplotlib draws it

                # Draw curved arrow from and to the centroid using an arc connection
                arrow = FancyArrowPatch(
                    posA=start,
                    posB=end,
                    connectionstyle=f"arc3,rad=1.0",  # big curve
                    arrowstyle='->',
                    color='black',
                    linewidth=1.5,
                    mutation_scale=10,
                    zorder=5
                )

                ax.add_patch(arrow)
                return ax

            def plot_histogram_edge_weights(ax: Axes, edge_weights: torch.Tensor, limits: Optional[Tuple[float,float]] = None) -> Axes:
                weights = edge_weights.detach().cpu().numpy()
                if limits:
                    ax.hist(weights, bins=15, range=limits, color='lightgray', edgecolor='black')
                else:
                    ax.hist(weights, bins=15, color='lightgray', edgecolor='black')
                
                # Force y-axis to use integer ticks only
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                
                return ax

            # emtpy graph => single plot with centroids in the shapefile
            if graphname == 'empty':
                fig, ax = plt.subplots(1, 1, figsize=(18, 8))
                plot_shape(ax, self.shapes)
                plot_centroids(ax, self.shapes, palette_blues[10:], column = 'random', size = 7)
                basic_plot_makeup(ax, title = f"Germany - {self.nuts_level}", vspine=False)

            # graphstructure => three plots, depending on whether local or global graph is shown.
            else:
                graph               = self.graph_registry[graphname]['structure']
                global_edge_index   = graph['edge_index']

                # global graph =>
                #   ax_main:        map showing all connected nodes
                #   ax_hist_quali:  histogram showing distribution of number of connections per node
                #   ax_hist_quanti: histogram showing distribution of edge_weights
                if node_idx == 'all':
                    fig = plt.figure(figsize=(18, 12))
                    gs  = fig.add_gridspec(3, 2, width_ratios=[3, 1], height_ratios=[1, 1, 1])
                    ax_main         = fig.add_subplot(gs[:, 0])
                    ax_hist_quali1  = fig.add_subplot(gs[0, 1])
                    ax_hist_quali2  = fig.add_subplot(gs[1, 1])
                    ax_hist_quanti  = fig.add_subplot(gs[2, 1])        

                    edge_index      = self.graph_registry[graphname]['structure']['edge_index']    
                    edge_weights    = self.graph_registry[graphname]['structure']['edge_weight']    

                    # ax_main: map showing all connected nodes
                    plot_shape(ax_main, self.shapes)               
                    plot_connection_lines(ax_main, edge_index, self.shapes)
                    plot_centroids(ax_main, self.shapes, palette_blues[70:71], column='random', size=5)
                    basic_plot_makeup(ax_main, f"{graphname} [qualitatively, global]", vspine=False)

                    # ax_hist_quali: histogram showing distribution of number of connections per node
                    plot_histogram_connection_degree(ax_hist_quali1, global_edge_index[0])
                    basic_plot_makeup(ax_hist_quali1, title = f"Global distribution of src connections per node {graphname}", vspine=True, xlabel='connections', ylabel = 'frequency')                    

                    plot_histogram_connection_degree(ax_hist_quali2, global_edge_index[1])
                    basic_plot_makeup(ax_hist_quali2, title = f"Global distribution of dst connections per node {graphname}", vspine=True, xlabel='connections', ylabel = 'frequency')   

                    # ax_hist_quanti: histogram showing distribution of edge_weights                
                    plot_histogram_edge_weights(ax_hist_quanti, edge_weights)
                    basic_plot_makeup(ax_hist_quanti, title = f"Global distribution of edge weights {graphname}", vspine=True, xlabel='weight', ylabel = 'frequency')  

                    plt.tight_layout()
                    plt.close(fig)
                    return fig

                # local graph =>
                #   ax_global:      map showing all connected nodes to node_idx (qualitatively)
                #   ax_local:       map showing all connected nodes to node_idx (quantitatively, zoomed in)
                #   ax_hist:        histogram showing distribution of edge_weights with node_idx as source
                elif isinstance(node_idx, int):  # node_idx is int
                    
                    # Get global graph info
                    graph              = self.graph_registry[graphname]['structure']
                    global_edge_index  = graph['edge_index']
                    global_edge_weight = graph['edge_weight']   
                    max_weight         = global_edge_weight.max().item()             
                    
                    # Mask edges connected to node_idx
                    mask                    = global_edge_index[0] == node_idx
                    local_edge_index_mask   = global_edge_index[:, mask]
                    local_edge_weights_mask = global_edge_weight[mask]

                    local_nodes             = global_edge_index[1][mask] # nodes that make up the neighborhood   
                    all_local_nodes         = torch.cat([torch.tensor([node_idx]), local_nodes]).unique()
                    global_to_local         = {global_idx: local_idx for local_idx, global_idx in enumerate(all_local_nodes.tolist())} # remap them.
                    
                    # Remap edge_index to local indices
                    local_edge_index_selected   = global_edge_index[:, mask].clone()
                    local_edge_index_selected[0]= torch.tensor([global_to_local[idx.item()] for idx in local_edge_index_selected[0]])
                    local_edge_index_selected[1]= torch.tensor([global_to_local[idx.item()] for idx in local_edge_index_selected[1]])                
                    connected_nodes             = torch.unique(local_edge_index_mask)

                    # Extract local shapes and node shape
                    local_shapes = self.shapes[self.shapes['node'].isin(connected_nodes.numpy())].reset_index(drop=True)# neighborhood shapes
                    node_shape   = self.shapes[self.shapes['node'] == node_idx]                                         # shape of node_idx
                    
                    # MAP ONLY MODE: Just return the global map
                    if map_only:
                        fig, ax_global = plt.subplots(1, 1, figsize=(18, 12))
                        plot_shape(ax_global, self.shapes)
                        plot_shape(ax_global, local_shapes, linewidth=1)
                        plot_connection_lines(ax_global, local_edge_index_selected, local_shapes)
                        plot_centroids(ax_global, local_shapes, palette_blues[70:71], column='random', size=5)
                        basic_plot_makeup(ax_global, f"{graphname} [qualitatively, node {node_idx}]", vspine=False)
                        plt.tight_layout()
                        plt.close(fig)
                        return fig
                    
                    # FULL MODE: Create subplots as before
                    fig = plt.figure(figsize=(18, 12))
                    gs  = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
                    ax_global = fig.add_subplot(gs[:, 0])
                    ax_local  = fig.add_subplot(gs[0, 1])
                    ax_hist   = fig.add_subplot(gs[1, 1])
                    
                    # ax_global: map showing all connected nodes to node_idx (qualitatively)
                    plot_shape(ax_global, self.shapes)
                    plot_shape(ax_local,  self.shapes)
                    plot_shape(ax_global, local_shapes, linewidth=1)
                    plot_connection_lines(ax_global, local_edge_index_selected, local_shapes)
                    plot_centroids(ax_global, local_shapes, palette_blues[70:71], column='random', size=5)
                    basic_plot_makeup(ax_global, f"{graphname} [qualitatively, node {node_idx}]", vspine=False)

                    # ax_hist: histogram showing distribution of edge_weights with node_idx as source
                    plot_histogram_edge_weights(ax_hist, local_edge_weights_mask, limits = (0, max_weight))
                    zoom_on_plot(ax_hist, xlim = (0, max_weight))
                    basic_plot_makeup(ax_hist, vspine=True, ticks=True, ylabel = 'frequency', xlabel = 'weight', title = 'local edge weights')                


                    # ax_local: map showing all connected nodes to node_idx (quantitatively, zoomed in)

                    # determine coordinates for zooming in on local neighborhood:
                    bounds  = local_shapes.total_bounds
                    margins = 0.1 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
                    xlim    = (bounds[0] - margins, bounds[2] + margins)
                    ylim    = (bounds[1] - margins, bounds[3] + margins)


                    # prepare df with the edge weights to plot the nodes in colors
                    stacked_array                   = np.vstack([local_edge_index_mask.numpy(), local_edge_weights_mask.numpy()]).astype(float)
                    edge_information                = pd.DataFrame(stacked_array.T)
                    edge_information.drop(labels    = 0, axis = 1, inplace=True)
                    edge_information.rename(columns = {1:'node',2:'weight'}, inplace = True)
                    edge_information['node']        = edge_information['node'].astype(int)

                    local_shapes_with_weight        = gpd.GeoDataFrame(pd.merge(local_shapes,edge_information, on ='node'))        
                    
                    zoom_on_plot(ax_local, xlim = xlim, ylim = ylim)
                    basic_plot_makeup(ax_local, title=f"Local graph zoomed on node {node_idx}", vspine=True, ticks= False)
                    plot_centroids(ax_local , node_shape, colorpalette=palette_blues[75:76], column='random')
                    plot_colored_nodes(ax_local, local_shapes_with_weight, column = 'weight', colorpalette='Reds', vmin = 0, vmax = max_weight)
    
                else:
                    raise ValueError(f'node_idx needs to be either the string "all" or an integer.\n{node_idx} is an invalid input.')
                
            plt.tight_layout()
            plt.close(fig)
            return fig


    def rename_graph(self, old_graphname: str, new_graphname: str) -> None:
        """ 
        Rename a graph in the registry (the key by which the graph is saved)
        the old graph is copied into the `new graphname` and the `old_graphname` is removed.
        """
        self.graph_registry[new_graphname] = self.graph_registry[old_graphname]
        del self.graph_registry[old_graphname]
        print(f'{old_graphname} has been replaced by {new_graphname}')

    def save_graph(self, graphname: Union[str,List[str]] = 'all') -> None:
        """
        Save edge index and weight from registry. 
        If graphname == 'all', all graphs are saved.
        """

        if graphname == ['all']:
            graphname = 'all'

        if graphname == 'all':
            graphnames = list(self.graph_registry.keys())

        elif isinstance(graphname, str):
            graphnames = [graphname]

        else:
            raise ValueError(f'Please provide a list or string for the graphname.')

        for graphname in graphnames:

            graph = self.graph_registry[graphname]['structure']
            edge_index = graph['edge_index']
            edge_weight= graph['edge_weight']

            torch.save(edge_index, os.path.join(self.graph_dir, f'{graphname}_edge_index.pt'))
            print(f'{checkmark} graph saved: edge index {graphname} saved to {self.graph_dir}')

            if edge_weight is not None:
                torch.save(edge_weight, os.path.join(self.graph_dir, f'{graphname}_edge_weight.pt'))
                print(f'{checkmark} graph saved: edge weight {graphname} saved to {self.graph_dir}')

    def _get_graph_summary(self, global_edge_index: torch.Tensor, global_edge_weight: torch.Tensor) -> Dict[str, float]:

        """ 
        validate the graph created
        """

        # statistics
        num_edges           = global_edge_index.shape[1]
        num_nodes           = int(global_edge_index.max().item())+1
        edge_density        = num_edges / (num_nodes * (num_nodes - 1))
        edge_weight_np      = global_edge_weight.cpu().numpy()
        edge_weight_mean    = edge_weight_np.mean()
        edge_weight_min     = edge_weight_np.min()
        edge_weight_max     = edge_weight_np.max()

        # isolated nodes:
        edges_out, edges_in = global_edge_index[0], global_edge_index[1] 
        out_degree          = torch.bincount(edges_out, minlength=num_nodes)
        in_degree           = torch.bincount(edges_in, minlength=num_nodes)
        isolated_mask       = (out_degree == 0) & (in_degree == 0)
        num_isolated        = isolated_mask.sum().item()

        summary = {
            'num_nodes'     : num_nodes,
            'num_edges'     : num_edges,
            'edge_density'  : edge_density,
            'num_isolated'  : num_isolated,
            'edge_weight_mean'  : edge_weight_mean,
            'edge_weight_min'   : edge_weight_min,
            'edge_weight_max'   : edge_weight_max
        }   

        return summary

    def __repr__(self) -> str:
        return f'<GraphConstructor> level {self.nuts_level}. Registry: {list(self.graph_registry.keys())}'