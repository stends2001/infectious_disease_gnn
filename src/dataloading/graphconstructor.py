import os
from typing import Optional, Literal, Tuple, List, Union
import geopandas as gpd
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from torch_geometric.utils import k_hop_subgraph
from typing import Optional, Literal
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from .epidataloader import EpiDataLoader
from statistics import mean
from numpy.typing import NDArray
import matplotlib.cm as cm
from dataclasses import dataclass, asdict
from .gnndataloader import GNNDataLoader
from ..utils import get_data_env


import pandas as pd 
import os

from typing import Optional

class PendlerDatenProcessor:
    """
    >>> graphobject = GraphConstructor(epidata=epidata_nuts3)
    >>> # graphobject.preview_shape_object()
    >>> graphobject.generate_graph(method = 'commuter',
    >>>                        self_connection='mean',
    >>>                        commuter_type = 'static',
    >>>                        commuting_threshold = 1_500,
    >>>                        scaling_method='rowwise',
    >>>                        name_addition='t1500')
    >>> # graphobject.preview_graph(graphname = 'commuter_t1500_selfmean_rowwis', node_idx = 26, qualitative= False)
    >>> graphobject.generate_graph(method = 'commuter',
    >>>                         self_connection='mean',
    >>>                         commuter_type = 'static',
    >>>                         commuting_threshold = 1_000,
    >>>                        scaling_method='rowwise',
    >>>                         name_addition='t1500')
    >>> graphobject.generate_graph(method = 'identity')
    >>> graphobject.generate_graph(method = 'boolean_neighbors',
    >>>                         self_connection='mean')
    >>> graphobject.save_graph(graphname = 'all')
    """
    def __init__(self, 
                 raw_folder_path: str,
                 processed_folder_path: str):
        
        self.raw_folder_path = raw_folder_path
        self.processed_folder_path = processed_folder_path

        self.harmfile = pd.read_csv(os.path.join(get_data_env(),'processed/germany/geospatial/harmonization/german_nuts_harmonization.tsv'), sep ="\t", dtype=str)

        self.dtypes = {'Regionalschlüssel'  : 'str', 
                       'Regionalschlüssel.1': 'str'}
        
        self.rename_cols = {'Regionalschlüssel'  :'nuts3_work',
                            'Regionalschlüssel.1':'nuts3_residence',
                            'Insgesamt'          : 'commuters'}
        
        self.columns     = list(self.rename_cols.values())
        self.data = {}

    def import_raw_data(self,
                        year: str):
        
        rawfolder = os.path.join(self.raw_folder_path, year)

        all_data = []  # to accumulate all processed DataFrames

        for file in os.listdir(rawfolder):  # iterate files in the folder
            path = os.path.join(rawfolder, file)


            trimmed_data = clean_csv_file(path)

            if trimmed_data is None:
                raise ImportError(f'No data found inside {path}')

            trimmed_csv = pd.read_csv(trimmed_data, sep=";", dtype=self.dtypes).rename(columns=self.rename_cols)
            trimmed_csv = trimmed_csv[self.columns]
            
            filtered_csv= trimmed_csv[trimmed_csv['nuts3_work'].isin(list(self.harmfile['nuts3'].unique()))]
            filtered_csv=filtered_csv[trimmed_csv['nuts3_residence'].isin(list(self.harmfile['nuts3'].unique()))]

            all_data.append(filtered_csv)  # append the processed dataframe

        self.data[year] = pd.concat(all_data, ignore_index=True)
        

        return self
    

from io import StringIO
def clean_csv_file(path: str) -> Optional[StringIO]:

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    start_index = next((i for i, line in enumerate(lines) if line.count(';') > 3), None)
        
    if start_index is not None:
        # Slice and remove trailing garbage lines
        trimmed_lines = lines[start_index:]
        valid_lines   = [line for line in trimmed_lines if line.count(';') > 3]

        cleaned_lines = [line.replace('.','') for line in valid_lines]

        # Create a temporary in-memory CSV
        return StringIO(''.join(cleaned_lines))

    else:
        return None



@dataclass
class GraphConfig:
    method:         str
    name_addition:  Optional[str]
    self_connection:str
    scaling_method: Optional[str]
    kwargs:         dict

cmap_red = cm.get_cmap('Reds')

class GraphConstructor:

    """
    calculates and saves graphs, edges and weights (if applicabe)

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

    >>> epidata = EpiDataLoader('influenza', data_env, aggr_level= '03', min_date='2012-06-01',max_date='2020-06-01')
    >>> epidata.add_time_features()
    >>> epidata.log_transform_target()
    >>> epidata.normalize('2018-06-01','2019-06-01','zscore')
    >>> epidata.add_lagged_features(range(4,8))

    # initating class
    >>> graphconstruction = GraphConstructor(graph_dir = 'data/graphs', population_data=epidata.population_by_node, shapes = epidata.shapedata, id_col='node')

    # generate graph
    >>> graphconstruction.generate_graph(
    >>>    method='boolean_neighbors', 
    >>>    name_addition='',
    >>> )

    # preview graph
    >>> graphconstruction.preview_graph('boolean_neighbors',391, 1)

    # save graph
    >>> graphconstruction.save_graph('boolean_neighbors')

    """

    def __init__(self,                  
                 epidata:         EpiDataLoader,
                 id_col:          str = 'node',
                 graph_dir:       str = "data/graphs/"):
        
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

        self.graph_methods          = ['boolean_neighbors', 'identity', 'mesh', 'distance_threshold','k_nearest', 'population_weighted', 'gravity_model', 'commuter']
        self.num_nodes              = epidata.data['context']['epidemiological_data'][self.id_col].nunique()
        
    def generate_graph(self, 
                       method: str                                         = 'boolean_neighbors',
                       name_addition:   Optional[str]                      = None,
                       self_connection: Literal['max','0','mean']          = 'mean',
                       scaling_method:  Optional[str] = None,
                       **kwargs) -> 'GraphConstructor':
        
        """
        generates graph structure on specific method. the produced edge_index and edge_weight,
        when applicable, are saved lunder the attribute `dict_graphs`, a dictionary with
        graphname: {'edge_index': ..., 'edge_weight' : ...}

        the graphname will be:
            method + name_addition + scaling_method 

        with "_" as separator. It is possible that either, or both, of `name_addition` 
        and `scaling_method` are None.
        """

        if method not in self.graph_methods:
            raise ValueError(f'{method} not a valid graph method. Please choose a method from this list:\n{self.graph_methods}')
    
        # Then in generate_graph():
        graphconfig = GraphConfig(
            method=method,
            name_addition=name_addition,
            self_connection=self_connection,
            scaling_method=scaling_method,
            kwargs=kwargs
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
            gdf=shapes_cp,
            tokens = self.tokens,
            popdata=self.population_data,
            id_col=self.id_col
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
                     'config'   : asdict(graphconfig)}   

        self.graph_registry[graphname] = graphdict
        
        print(f'{graphname} generated')

        return self
        
    # def preview_graph(self, 
    #                   graphname: str,
    #                   node_idx: int    = 11,
    #                   num_hops: int    = 1) -> 'GraphConstructor':

    #     """
    #     previews a subset of the graph, specific to the node_idx,
    #     with the level of neighborhoods as put in.
    #     """

    #     graph             = self.graph_registry[graphname]['structure']
    #     global_edge_index = graph['edge_index']
    #     global_edge_weight= graph['edge_weight']

    #     mask              = global_edge_index[0] == node_idx
    #     local_nodes       = global_edge_index[1][mask]

    #     local_shapes      = self.shapes[self.shapes['node'].isin(local_nodes.numpy())]
    #     node_shape        = self.shapes[self.shapes['node'] == node_idx]

    #     quanti_graph = len(torch.unique(global_edge_weight)) > 2

    #     if quanti_graph:
    #         # Create figure with subplot layout: main plot on left, 2 subplots on right
    #         fig = plt.figure(figsize=(18, 8))
    #         gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])

    #         # Main plot that spans both rows in the left column
    #         ax_main = fig.add_subplot(gs[:, 0])

    #         # Local plot (top right)
    #         ax_local = fig.add_subplot(gs[0, 1])

    #         # Histogram plot (bottom right)
    #         ax_hist = fig.add_subplot(gs[1, 1])

    #         # Set example titles and labels
    #         ax_main.set_title("Main Plot")
    #         ax_local.set_title("Local Plot")
    #         ax_hist.set_title("Histogram")
    #         ax_local.set_aspect('auto')
    #         ax_hist.set_aspect('auto')

    #         # Create custom colormap: lightgrey for 0, then Reds for positive values
    #         max_val = global_edge_weight.max().item()
            
    #         # Create boundaries for discrete mapping
    #         boundaries = [0, 0.001] + list(np.linspace(0.001, max_val, 10))

    #         colors = ['lightgrey'] + [cmap_red(i) for i in np.linspace(0.3, 1, len(boundaries)-2)]
    #         custom_cmap = mcolors.ListedColormap(colors)
    #         norm = mcolors.BoundaryNorm(boundaries, custom_cmap.N)

    #         # Background plot for both main and local
    #         self.shapes.plot(color='lightgrey', edgecolor='white', linewidth=0.1, ax=ax_main)
    #         self.shapes.plot(color='lightgrey', edgecolor='white', linewidth=0.1, ax=ax_local)

    #         if global_edge_weight is not None:
    #             local_edge_weight = global_edge_weight[mask]

    #             local_connectivity = pd.DataFrame({
    #                 "node": local_nodes.tolist(),
    #                 "weight": local_edge_weight.tolist()
    #             })
                
    #             merged_data = pd.merge(local_shapes, local_connectivity, on='node')
                
    #             # Plot on main axis with colorbar
    #             im_main = merged_data.plot(column='weight', cmap=custom_cmap, norm=norm,
    #                                     edgecolor='black', linewidth=0.1, ax=ax_main)
                
    #             # Plot on local axis
    #             merged_data.plot(column='weight', cmap=custom_cmap, norm=norm,
    #                         edgecolor='black', linewidth=0.1, ax=ax_local)
                
    #             # Add colorbar to main plot (full height)
    #             sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    #             sm.set_array([])
    #             cbar = fig.colorbar(sm, ax=ax_main, shrink=1.0)
    #             cbar.set_label('Edge Weight', rotation=270, labelpad=20)
                
    #             # Add custom patch for 0 value in colorbar
    #             from matplotlib.patches import Rectangle
    #             cbar.ax.add_patch(Rectangle((0, 0), 1, 0.001/max_val, facecolor='lightgrey', 
    #                                     edgecolor='black', linewidth=0.5))
                
    #             # Node shape with its own connection strength on local plot
    #             # First check if node has outgoing connections
    #             if len(local_connectivity) > 0:
    #                 # Check if the node_idx itself appears in the connectivity (has self-loop or is in local_nodes)
    #                 node_in_local = node_idx in local_connectivity['node'].values
    #                 if node_in_local:
    #                     node_connectivity = local_connectivity[local_connectivity['node'] == node_idx]
    #                     node_connectivity_shapes = pd.merge(node_shape, node_connectivity, on='node')
    #                     node_connectivity_shapes.plot(column='weight', cmap=custom_cmap, norm=norm,
    #                                                 edgecolor='black', linewidth=2, ax=ax_local)
    #                 else:
    #                     # Node not in connectivity, color by its connection strength to others or default
    #                     # Use the mean weight of its connections as a proxy
    #                     mean_weight = local_edge_weight.mean().item()
    #                     node_color = custom_cmap(norm(mean_weight))
    #                     node_shape.plot(color=node_color, edgecolor='black', linewidth=2, ax=ax_local)
    #             else:
    #                 # If no connections, use minimum color (lightgrey)
    #                 node_shape.plot(color='lightgrey', edgecolor='black', linewidth=2, ax=ax_local)
                
    #             # Histogram (horizontal)
    #             sns.histplot(y=global_edge_weight.numpy(), ax=ax_hist, bins=20, color='red', alpha=0.7)
    #             ax_hist.set_title('Global Edge Weights Distribution', fontsize=10)
    #             ax_hist.set_ylabel('Edge Weight', fontsize=9)
    #             ax_hist.set_xlabel('Count', fontsize=9)

    #         # Plot node in blue on main plot
    #         node_shape.plot(color='blue', edgecolor='black', linewidth=2, ax=ax_main)
            
    #         # Set limits for local plot to zoom in
    #         if len(local_shapes) > 0:
    #             bounds = local_shapes.total_bounds
    #             margin = 0.1 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    #             ax_local.set_xlim(bounds[0] - margin, bounds[2] + margin)
    #             ax_local.set_ylim(bounds[1] - margin, bounds[3] + margin)
            
    #         # Match the width of ax_bar to ax_local by getting the xlim
    #         bar_xlim = ax_local.get_xlim()
    #         bar_width = bar_xlim[1] - bar_xlim[0]
            
    #         # Remove ticks and labels
    #         for ax in [ax_main, ax_local]:
    #             ax.tick_params(left=False, right=False, bottom=False, top=False,
    #                         labelleft=False, labelbottom=False)
            
    #         ax_local.set_title('Local Neighborhood', fontsize=10)

    #     else:
    #         # Non-quantitative graph
    #         fig = plt.figure(figsize=(18, 8))
    #         gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])

    #         # Main plot that spans both rows in the left column
    #         ax_main = fig.add_subplot(gs[:, 0])

    #         # Local plot (top right)
    #         ax_local = fig.add_subplot(gs[0, 1])

    #         # Histogram plot (bottom right)
    #         ax_bar = fig.add_subplot(gs[1, 1])

    #         # Set example titles and labels
    #         ax_main.set_title("Main Plot")
    #         ax_local.set_title("Local Plot")
    #         ax_bar.set_title("Histogram")
    #         ax_local.set_aspect('auto')
    #         ax_bar.set_aspect('auto')

    #         # Background plot for both
    #         self.shapes.plot(color='lightgrey', edgecolor='white', linewidth=0.1, ax=ax_main)
    #         self.shapes.plot(color='lightgrey', edgecolor='white', linewidth=0.1, ax=ax_local)

    #         # Plot local connections
    #         local_shapes.plot(color='red', edgecolor='white', linewidth=0.1, alpha=0.6, ax=ax_main)
    #         local_shapes.plot(color='red', edgecolor='white', linewidth=0.1, alpha=0.6, ax=ax_local)

    #         # Plot node
    #         node_shape.plot(color='blue', edgecolor='black', linewidth=2, ax=ax_main)

    #         # Add legend patches for main plot
    #         from matplotlib.patches import Patch
    #         legend_elements = [
    #             Patch(facecolor='lightgrey', label='No Connection (0)'),
    #             Patch(facecolor='red', alpha=0.6, label='Connection (1)')
    #         ]
    #         ax_main.legend(handles=legend_elements, loc='upper right')

    #         # Set limits for local plot
    #         if len(local_shapes) > 0:
    #             bounds = local_shapes.total_bounds
    #             margin = 0.1 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    #             ax_local.set_xlim(bounds[0] - margin, bounds[2] + margin)
    #             ax_local.set_ylim(bounds[1] - margin, bounds[3] + margin)

    #         # Bar plot showing number of connections (horizontal bars)
    #         connection_counts = {'No Connection': len(self.shapes) - len(local_shapes) - 1, 
    #                         'Connected': len(local_shapes)}
            
    #         # Create horizontal bar plot with same width as local neighborhood
    #         bars = ax_bar.barh(list(connection_counts.keys()), list(connection_counts.values()), 
    #                         color=['lightgrey', 'red'], alpha=0.7, height=0.4)
    #         ax_bar.set_title('Connection Summary', fontsize=10)
    #         ax_bar.set_xlabel('Count', fontsize=9)
            
    #         # Set the x-axis limits to match the proportional width
    #         max_count = max(connection_counts.values())
    #         ax_bar.set_xlim(0, max_count * 1.2)  # Add some padding
            
    #         # Add value labels
    #         for bar, count in zip(bars, connection_counts.values()):
    #             width = bar.get_width()
    #             ax_bar.text(width + max_count * 0.02, bar.get_y() + bar.get_height()/2.,
    #                     f'{count}', ha='left', va='center', fontsize=9)

    #         # Remove ticks for spatial plots
    #         for ax in [ax_main, ax_local]:
    #             ax.tick_params(left=False, right=False, bottom=False, top=False,
    #                         labelleft=False, labelbottom=False)
            
    #         ax_local.set_title('Local Neighborhood', fontsize=10)

    #     title = f'Subgraph of {graphname}, node {node_idx}' if num_hops==1 else f'Subgraph of {graphname}, node {node_idx}\nneighborhood level {num_hops}'
    #     ax_main.set_title(title, fontsize=12)

    #     plt.tight_layout()
    #     fig.show()
    #     return self

    # def preview_shape_object(self, local_nodes: Optional[Union[List[int],int]] = None) -> 'GraphConstructor':


    #     global_shapes = self.shapes
    #     fig, ax_main = plt.subplots(figsize = (10,8))

    #     global_shapes.plot(color = 'lightgrey',
    #                                 edgecolor='white',
    #                                 linewidth= 0.1,
    #                                 ax = ax_main)
    #     # get point - coordinates
    #     points   = global_shapes.geometry.centroid
    #     n_points = len(points)
    #     palette  = sns.color_palette("Blues", n_points)
    #     np.random.shuffle(palette)      
        
    #     # Plot points with random blue shades
    #     for i, point in enumerate(points):
    #         x, y = point.coords[0]  # Get coordinates from the point
    #         ax_main.plot(x, y, 'o', color=palette[i], markersize=5.5, 
    #                     markeredgecolor='black', markeredgewidth=0.6)

    #     ax_main.set_title('Shape object')

    #     ax_main.tick_params(
    #         left=False, right=False, bottom=False, top=False,  # no ticks
    #         labelleft=False, labelbottom=False                 # no labels
    #     )        

    #     if local_nodes:
    #         if isinstance(local_nodes, int):
    #             local_nodes = [local_nodes]

    #         local_shapes = global_shapes[global_shapes['node'].isin(local_nodes)]
    #         points_local = local_shapes.geometry.centroid
    #         for i, point in enumerate(points_local):
    #             x, y = point.coords[0]  # Get coordinates from the point
    #             ax_main.plot(x, y, 'o', color='red', markersize=6, 
    #                         markeredgecolor='black', markeredgewidth=0.6)            

    #     fig.show()
    #     return self

    def preview_graph(self, 
                    graphname: str,
                    node_idx: Union[int, str] = 11,  # allow 'all' as str
                    num_hops: int = 1,
                    qualitative: bool = False) -> 'GraphConstructor':

        graph = self.graph_registry[graphname]['structure']
        global_edge_index = graph['edge_index']
        global_edge_weight = graph.get('edge_weight', None)

        if qualitative:
            # Qualitative graph: ignore weights, plot connections as black lines
            
            if node_idx == 'all':
                # Show all connected regions in the graph
                connected_nodes = set(global_edge_index.flatten().tolist())
                local_shapes = self.shapes[self.shapes['node'].isin(connected_nodes)]
                node_shape = None
            else:
                # Show neighborhood of a single node
                mask = global_edge_index[0] == node_idx
                local_nodes = global_edge_index[1][mask]
                local_shapes = self.shapes[self.shapes['node'].isin(local_nodes.numpy())]
                node_shape = self.shapes[self.shapes['node'] == node_idx]

            fig, (ax_main, ax_local) = plt.subplots(1, 2, figsize=(18, 8))

            # Plot base shapes
            self.shapes.plot(color='lightgrey', edgecolor='white', linewidth=0.1, ax=ax_main)
            self.shapes.plot(color='lightgrey', edgecolor='white', linewidth=0.1, ax=ax_local)

            # Draw black lines for edges on main plot
            for i, j in zip(global_edge_index[0].tolist(), global_edge_index[1].tolist()):
                point_i = self.shapes.loc[self.shapes['node'] == i, 'geometry'].centroid.values[0]
                point_j = self.shapes.loc[self.shapes['node'] == j, 'geometry'].centroid.values[0]
                ax_main.plot([point_i.x, point_j.x], [point_i.y, point_j.y], color='black', linewidth=0.6, alpha=0.6)

            # Highlight connected local shapes with black edges (no fill) on local plot
            if len(local_shapes) > 0:
                local_shapes.plot(color='none', edgecolor='black', linewidth=1.2, ax=ax_local)

            # Highlight the selected node in blue if applicable
            if node_shape is not None and not node_shape.empty:
                node_shape.plot(color='blue', edgecolor='black', linewidth=2, ax=ax_local)

            # Titles and aesthetics
            ax_main.set_title(f"Qualitative Graph: {graphname} (All nodes)" if node_idx == 'all' else
                            f"Qualitative Graph: {graphname} - Node {node_idx}")
            ax_local.set_title("Local Neighborhood")
            for ax in [ax_main, ax_local]:
                ax.axis('off')

            plt.tight_layout()
            plt.show()

            return self

        else:
            # Quantitative graph plotting (your original code below)

            mask = global_edge_index[0] == node_idx
            local_nodes = global_edge_index[1][mask]

            local_shapes = self.shapes[self.shapes['node'].isin(local_nodes.numpy())]
            node_shape = self.shapes[self.shapes['node'] == node_idx]

            quanti_graph = global_edge_weight is not None and len(torch.unique(global_edge_weight)) > 2

            # Create figure and gridspec
            fig = plt.figure(figsize=(18, 8))
            gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])

            ax_main = fig.add_subplot(gs[:, 0])
            ax_local = fig.add_subplot(gs[0, 1])
            ax_hist = fig.add_subplot(gs[1, 1])

            ax_main.set_title("Main Plot")
            ax_local.set_title("Local Plot")
            ax_hist.set_title("Histogram")
            ax_local.set_aspect('auto')
            ax_hist.set_aspect('auto')

            # Background plot
            self.shapes.plot(color='lightgrey', edgecolor='white', linewidth=0.1, ax=ax_main)
            self.shapes.plot(color='lightgrey', edgecolor='white', linewidth=0.1, ax=ax_local)

            if global_edge_weight is not None:
                local_edge_weight = global_edge_weight[mask]

                local_connectivity = pd.DataFrame({
                    "node": local_nodes.tolist(),
                    "weight": local_edge_weight.tolist()
                })

                merged_data = pd.merge(local_shapes, local_connectivity, on='node')

                # Create custom colormap (lightgrey for 0, Reds for positive)
                max_val = global_edge_weight.max().item()
                boundaries = [0, 0.001] + list(np.linspace(0.001, max_val, 10))
                colors = ['lightgrey'] + [plt.cm.Reds(i) for i in np.linspace(0.3, 1, len(boundaries)-2)]
                custom_cmap = mcolors.ListedColormap(colors)
                norm = mcolors.BoundaryNorm(boundaries, custom_cmap.N)

                # Plot merged data on main and local axes
                im_main = merged_data.plot(column='weight', cmap=custom_cmap, norm=norm,
                                        edgecolor='black', linewidth=0.1, ax=ax_main)

                merged_data.plot(column='weight', cmap=custom_cmap, norm=norm,
                                edgecolor='black', linewidth=0.1, ax=ax_local)

                # Colorbar on main plot
                sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax_main, shrink=1.0)
                cbar.set_label('Edge Weight', rotation=270, labelpad=20)

                # Patch for zero in colorbar
                from matplotlib.patches import Rectangle
                cbar.ax.add_patch(Rectangle((0, 0), 1, 0.001/max_val, facecolor='lightgrey',
                                        edgecolor='black', linewidth=0.5))

                # Highlight node in local plot with its own connection strength
                if len(local_connectivity) > 0:
                    node_in_local = node_idx in local_connectivity['node'].values
                    if node_in_local:
                        node_connectivity = local_connectivity[local_connectivity['node'] == node_idx]
                        node_connectivity_shapes = pd.merge(node_shape, node_connectivity, on='node')
                        node_connectivity_shapes.plot(column='weight', cmap=custom_cmap, norm=norm,
                                                    edgecolor='black', linewidth=2, ax=ax_local)
                    else:
                        mean_weight = local_edge_weight.mean().item()
                        node_color = custom_cmap(norm(mean_weight))
                        node_shape.plot(color=node_color, edgecolor='black', linewidth=2, ax=ax_local)
                else:
                    node_shape.plot(color='lightgrey', edgecolor='black', linewidth=2, ax=ax_local)

                # Histogram of global edge weights
                import seaborn as sns
                sns.histplot(y=global_edge_weight.numpy(), ax=ax_hist, bins=20, color='red', alpha=0.7)
                ax_hist.set_title('Global Edge Weights Distribution', fontsize=10)
                ax_hist.set_ylabel('Edge Weight', fontsize=9)
                ax_hist.set_xlabel('Count', fontsize=9)

            # Plot node in blue on main plot
            node_shape.plot(color='blue', edgecolor='black', linewidth=2, ax=ax_main)

            # Zoom local plot limits
            if len(local_shapes) > 0:
                bounds = local_shapes.total_bounds
                margin = 0.1 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
                ax_local.set_xlim(bounds[0] - margin, bounds[2] + margin)
                ax_local.set_ylim(bounds[1] - margin, bounds[3] + margin)

            # Remove ticks on spatial plots
            for ax in [ax_main, ax_local]:
                ax.tick_params(left=False, right=False, bottom=False, top=False,
                            labelleft=False, labelbottom=False)

            title = f'Subgraph of {graphname}, node {node_idx}' if num_hops == 1 else \
                    f'Subgraph of {graphname}, node {node_idx}\nneighborhood level {num_hops}'
            ax_main.set_title(title, fontsize=12)

            plt.tight_layout()
            plt.show()

            return self



    def rename_graph(self, old_graphname:str, 
                     new_graphname: str) -> 'GraphConstructor':

        self.graph_registry[new_graphname] = self.graph_registry[old_graphname]

        del self.graph_registry[old_graphname]

        print(f'{old_graphname} has been replaced by {new_graphname}')

        return self 

    def save_graph(self, graphname: Union[str,List[str]] = 'all') -> 'GraphConstructor':

        """Save edge index (and if applicable edge weight) from dictionary"""

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
            print(f'edge index {graphname} saved to {self.graph_dir}')

            if edge_weight is not None:
                torch.save(edge_weight, os.path.join(self.graph_dir, f'{graphname}_edge_weight.pt'))
                print(f'edge weight {graphname} saved to {self.graph_dir}')

        return self

class GraphAddSelfLoops:
    def __init__(self,
                 edge_indices: List[Tuple[int,int]],
                 edge_weights: List[float],
                 num_nodes   : int,
                 node_ids    : NDArray):
        
        self.edge_weights_ls = edge_weights 
        self.edge_indices_ls = edge_indices
        self.num_nodes       = num_nodes
        self.node_ids        = node_ids


        self.SELFLOOP_FUNCS = {
                                    '0'     : self._add0,
                                    'max'   : self._addmax,
                                    'mean'  : self._addmean,
        }    

    def add_loops(self, method: str) -> Tuple[List[Tuple[int,int]], List[float]]:

        if method not in self.SELFLOOP_FUNCS:
            raise ValueError(f"Unknown selfloop addition method: {method}")

        # add the indices
        self_loops              = [(nid, nid) for nid in self.node_ids]   
        updated_indices         = self.edge_indices_ls + self_loops

        # add the weights
        updated_weights        = self.SELFLOOP_FUNCS[method]() 

        return (updated_indices, updated_weights)
    
                
    def _add0(self):
        return self.edge_weights_ls + [0 for _ in self.node_ids]

    def _addmax(self):
        max_weight = max(self.edge_weights_ls) if self.edge_weights_ls else 1
        return self.edge_weights_ls + [max_weight for _ in self.node_ids]        

    def _addmean(self):
        mean_weight = mean(self.edge_weights_ls)
        return self.edge_weights_ls + [mean_weight for _ in self.node_ids]        
       
class GraphEdgeWeightNormalizer:

    def __init__(self,
                 edge_indices: torch.Tensor,
                 edge_weights: torch.Tensor,
                 num_nodes   : int):
        
        self.edge_weights = edge_weights 
        self.edge_indices = edge_indices
        self.num_nodes    = num_nodes


        self.NORMALIZATION_FUNCS = {
                                    'minmax'    : self._minmax,
                                    'log'       : self._log,
                                    'zscore'    : self._zscore,
                                    'symmetric' : self._symmetric,
                                    'rowwise'   : self._rowwise
        }

    def normalize(self, method: str) -> torch.Tensor:

        if method not in self.NORMALIZATION_FUNCS:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return self.NORMALIZATION_FUNCS[method]()
        
    def _minmax(self) -> torch.Tensor:
        min_w = self.edge_weights.min()
        max_w = self.edge_weights.max()
        if max_w > min_w:
            edge_weights = (self.edge_weights - min_w) / (max_w - min_w)
        else:
            edge_weights = torch.zeros_like(self.edge_weights)

        return edge_weights 
    
    def _log(self) -> torch.Tensor:
        edge_weights = torch.log1p(self.edge_weights)
        max_w = edge_weights.max()
        if max_w > 0:
            edge_weights = edge_weights / max_w
        else:
            edge_weights = torch.zeros_like(edge_weights)
        return edge_weights

    def _zscore(self) -> torch.Tensor:
        mean_w = self.edge_weights.mean()
        std_w  = self.edge_weights.std()
        if std_w > 0:
            edge_weights = (self.edge_weights - mean_w) / std_w
        else:
            edge_weights = torch.zeros_like(self.edge_weights)

        return edge_weights

    def _symmetric(self) -> torch.Tensor:
            # Build adjacency matrix in sparse form
            # D_ii = sum of weights connected to node i
            row, col = self.edge_indices[0], self.edge_indices[1]

            deg = torch.zeros(self.num_nodes, dtype=self.edge_weights.dtype, device=self.edge_weights.device)
            deg.scatter_add_(0, row, self.edge_weights)
            
            # Compute D^{-1/2} for each node, avoid division by zero
            deg_inv_sqrt                                = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')]  = 0
            
            # For each edge (i,j), normalized weight is w_ij * D_i^{-1/2} * D_j^{-1/2}
            edge_weights = self.edge_weights * deg_inv_sqrt[row] * deg_inv_sqrt[col]    
            return edge_weights

    def _rowwise(self) -> torch.Tensor:
            # Row-normalize: divide each edge weight by sum of weights in the source node's row
            row, _ = self.edge_indices
            deg = torch.zeros(self.num_nodes, dtype=self.edge_weights.dtype, device=self.edge_weights.device)
            deg.scatter_add_(0, row, self.edge_weights)
            
            # avoid division by zero
            deg_inv = torch.zeros_like(deg)
            nonzero_mask = deg > 0
            deg_inv[nonzero_mask] = 1.0 / deg[nonzero_mask]
            
            edge_weights = self.edge_weights * deg_inv[row]    
            return edge_weights

class GraphGeneration:
    def __init__(self,
                 gdf:       gpd.GeoDataFrame,
                 tokens:    dict,
                 popdata:   pd.DataFrame,
                 id_col:    str = 'node',
                 ):
        
        self.gdf    = gdf
        self.popdata= popdata
        self.id_col = id_col
        self.tokens = tokens

    # VALIDATION
        if self.id_col not in self.gdf.columns.tolist():
            raise ValueError(
                f'Yikes! id column not found in gdf. '
                f'gdf columns are: {self.gdf.columns.tolist()} '
                f'while id_col is supposedly {self.id_col}'
            )
        
        if 'geometry' not in self.gdf.columns:
            raise ValueError(f'GeoDataFrame must have a "geometry" column')    

        self.GENERATION_FUNCS = {
                                    'boolean_neighbors'     : self._boolean_neighbors,
                                    'identity'              : self._identity,
                                    'mesh'                  : self._mesh,
                                    'distance_threshold'    : self._distance_threshold,
                                    'k_nearest'             : self._k_nearest,
                                    'population_weighted'   : self._population_weighted,
                                    'gravity_model'         : self._gravity_model,
                                    'commuter'              : self._commuter,

        }        

    def generate_graph(self, method: str, **kwargs) -> Tuple[List[Tuple[int,int]], Optional[List[float]]]:

        if method not in self.GENERATION_FUNCS:
            available = ', '.join(self.GENERATION_FUNCS.keys())
            raise ValueError(
                f"Unknown graph generation method: '{method}'. "
                f"Available methods: {available}"
            )
        
        edge_indices, edge_weights =  self.GENERATION_FUNCS[method](**kwargs) 


        return (edge_indices, edge_weights)

    def _commuter(self, commuter_type: Literal['static'], commuting_threshold: int = 1_000) -> Tuple[List[Tuple[int,int]], List[float]]:
        if commuter_type == 'static':

            commuter_data_object = PendlerDatenProcessor(raw_folder_path = os.path.join(get_data_env(),'raw/germany/mobility/commuter_data/auspendler/'), processed_folder_path= os.path.join(get_data_env(),'processed/germany/mobility/commuter_data/')).import_raw_data('2024')
            commuter_data        = commuter_data_object.data['2024']
            commuter_data.loc[:, 'nuts3_work'] = commuter_data.loc[:, 'nuts3_work'].map(self.tokens)
            commuter_data.loc[:, 'nuts3_residence'] = commuter_data.loc[:, 'nuts3_residence'].map(self.tokens)

        commuter_data = commuter_data[commuter_data['commuters'] > commuting_threshold]
        node_weights = []
        node_edges = []
        for i in range(len(commuter_data)):


            node_edges.append((int(commuter_data.iloc[i]['nuts3_work']), int(commuter_data.iloc[i]['nuts3_residence'])))
            node_weights.append((int(commuter_data.iloc[i]['commuters'])))

        return (node_edges, node_weights)
                
    def _boolean_neighbors(self) -> Tuple[List[Tuple[int,int]], None]:
        dfc         = self.gdf[[self.id_col,'geometry']].copy()
        dfc         = dfc.sort_values(self.id_col).reset_index(drop=True)
        neighbors   = gpd.sjoin(dfc, dfc, how='inner', predicate='touches').reset_index(drop=False)
        neighbors   = neighbors[neighbors[f'{self.id_col}_left'] != neighbors[f'{self.id_col}_right']]
        edges       = list(zip(neighbors[f'{self.id_col}_left'], neighbors[f'{self.id_col}_right']))
        edges      += [(t, s) for s, t in edges]
        edges       = list(set(edges))
        return edges, None

    def _identity(self) -> Tuple[List[Tuple[int,int]], None]:
        dfc         = self.gdf[[self.id_col]].sort_values(self.id_col).reset_index(drop=True)
        node_ids    = dfc[self.id_col].dropna().astype(int).values
        edges       = [(int(nid), int(nid)) for nid in node_ids]
        return edges, None

    def _mesh(self) -> Tuple[List[Tuple[int,int]], None]:
        dfc         = self.gdf[[self.id_col]].sort_values(self.id_col).reset_index(drop=True)
        node_ids    = dfc[self.id_col].dropna().astype(int).values
        edges       = [(int(s), int(t)) for s in node_ids for t in node_ids]
        return edges, None
            
    def _distance_threshold(self, max_distance: float) -> Tuple[List[Tuple[int,int]], None]:
        dfc         = self.gdf[[self.id_col, 'geometry']].sort_values(self.id_col).reset_index(drop=True)
        centroids   = dfc.geometry.centroid
        coords      = np.array([[p.x, p.y] for p in centroids])
        distances   = euclidean_distances(coords)
        edges       = []
        for i in range(len(dfc)):
            for j in range(len(dfc)):
                if distances[i, j] <= max_distance:
                    edges.append((int(dfc.iloc[i][self.id_col]), int(dfc.iloc[j][self.id_col])))
        return edges, None

    def _k_nearest(self, k: int) -> Tuple[List[Tuple[int,int]], None]:
        dfc         = self.gdf[[self.id_col, 'geometry']].sort_values(self.id_col).reset_index(drop=True)
        centroids   = dfc.geometry.centroid
        coords      = np.array([[p.x, p.y] for p in centroids])
        distances   = euclidean_distances(coords)
        edges       = []
        for i in range(len(dfc)):
            nearest_indices = np.argsort(distances[i])[1:k+1]
            for j in nearest_indices:
                edges.append((int(dfc.iloc[i][self.id_col]), int(dfc.iloc[j][self.id_col])))
        return edges, None

    def _population_weighted(self, max_distance: float) -> Tuple[List[Tuple[int,int]], List[float]]:
        dfc                     = self.gdf[[self.id_col, 'geometry']].sort_values(self.id_col).reset_index(drop=True)
        dfc                     = dfc.merge(self.popdata[[self.id_col, 'population_size']], on=self.id_col, how='left')
        dfc['population_size']  = dfc['population_size'].fillna(dfc['population_size'].mean())
        centroids               = dfc.geometry.centroid
        coords                  = np.array([[p.x, p.y] for p in centroids])
        distances               = euclidean_distances(coords)
        edges                   = []
        weights                 = []

        for i in range(len(dfc)):
            for j in range(len(dfc)):
                if distances[i, j] <= max_distance and i != j:
                    edges.append((int(dfc.iloc[i][self.id_col]), int(dfc.iloc[j][self.id_col])))
                    pop_i = dfc.iloc[i]['population_size']
                    pop_j = dfc.iloc[j]['population_size']
                    dist = distances[i, j]
                    weight = (pop_i * pop_j) / (dist + 1)
                    weights.append(weight)       
        return edges, weights

    def _gravity_model(self, max_distance: float, alpha: float=2.0, 
                       density_control: str ='top_k', k : int=10, 
                       weight_threshold: Optional[int]=None, 
                       distance_decay_factor: float=1.0) -> Tuple[List[Tuple[int,int]], List[float]]:
        """
        Generate gravity model graph with density control options.
        
        Parameters:
        - density_control: 'top_k', 'threshold', 'adaptive', or 'distance_bands'
        - k: number of top connections per node (for top_k method)
        - weight_threshold: minimum weight to keep edge (for threshold method)
        - distance_decay_factor: additional distance penalty factor
        """
        dfc                     = self.gdf[[self.id_col, 'geometry']].sort_values(self.id_col).reset_index(drop=True)
        dfc                     = dfc.merge(self.popdata[[self.id_col, 'population_size']], on=self.id_col, how='left')
        dfc['population_size']  = dfc['population_size'].fillna(dfc['population_size'].mean())
        
        centroids               = dfc.geometry.centroid
        coords                  = np.array([[point.x, point.y] for point in centroids])
        distances               = euclidean_distances(coords)

        # Calculate all potential weights first
        all_weights = []
        all_edges = []
        
        for i in range(len(dfc)):
            node_weights = []
            node_edges = []
            
            for j in range(len(dfc)):
                if distances[i, j] <= max_distance and i != j:
                    pop_i = dfc.iloc[i]['population_size']
                    pop_j = dfc.iloc[j]['population_size']
                    dist = distances[i, j]
                    
                    # Enhanced gravity model with additional distance decay
                    weight = (pop_i * pop_j) / ((dist * distance_decay_factor)**alpha + 1e-6)
                    
                    node_weights.append((weight, j))
                    node_edges.append((int(dfc.iloc[i][self.id_col]), int(dfc.iloc[j][self.id_col])))
            
            # Apply density control
            if density_control == 'top_k':
                # Keep only top k connections per node
                node_weights.sort(reverse=True)
                selected = node_weights[:k]
                
            elif density_control == 'threshold':
                # Keep connections above weight threshold
                if weight_threshold is None:
                    weight_threshold = np.percentile([w[0] for w in node_weights], 75)
                selected = [(w, j) for w, j in node_weights if w >= weight_threshold]
                
            elif density_control == 'adaptive':
                # Adaptive threshold based on node's maximum weight
                if node_weights:
                    max_weight = max(w[0] for w in node_weights)
                    adaptive_threshold = max_weight * 0.1  # Keep top 10% of weights
                    selected = [(w, j) for w, j in node_weights if w >= adaptive_threshold]
                else:
                    selected = []
                    
            elif density_control == 'distance_bands':
                # Prioritize closer connections, limit distant ones
                node_weights_with_dist = [(w, j, distances[i, j]) for w, j in node_weights]
                node_weights_with_dist.sort(key=lambda x: x[2])  # Sort by distance
                
                # Take more from closer distance bands
                close_band  = [x for x in node_weights_with_dist if x[2] <= max_distance * 0.3]
                medium_band = [x for x in node_weights_with_dist if max_distance * 0.3 < x[2] <= max_distance * 0.7]
                far_band    = [x for x in node_weights_with_dist if x[2] > max_distance * 0.7]
                
                selected_items = close_band[:6] + medium_band[:3] + far_band[:1]  # 6+3+1=10 connections max
                selected = [(w, j) for w, j, d in selected_items]
            
            else:
                selected = node_weights  # No density control
            
            # Add selected edges and weights
            for idx, (weight, j) in enumerate(selected):
                all_edges.append(node_edges[node_weights.index((weight, j))])
                all_weights.append(weight)
            
        return all_edges, all_weights