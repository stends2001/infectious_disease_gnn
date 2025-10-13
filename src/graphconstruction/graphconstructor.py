import os
from typing import Optional, Tuple, List, Union, Literal, Dict
from dataclasses import dataclass, asdict

import pandas as pd
import geopandas as gpd
import numpy as np
from numpy.typing import NDArray

import torch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns

from ..dataloading.epidataloader import EpiDataLoader
from ..dataloading.gnndataloader import GNNDataLoader

from .graphconstructor_edgeweightnormalizer import GraphEdgeWeightNormalizer
from .graphconstructor_generation import GraphGeneration
from .graphconstructor_selfloops import GraphAddSelfLoops

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

    epidata_loader_bn = GNNDataLoader(disease_name, get_data_env(), nuts_level=nuts_level, min_date=min_date,max_date=max_date, include_population=False, horizon_size = horizon_size, horizon_leadtime = horizon_leadtime, sequence_length=sequence_length, split_berlin=split_berlin)
    epidata_loader_bn.add_time_features()
    epidata_loader_bn.log_transform_target()
    epidata_loader_bn.set_splits(split_trainval, split_valtest)
    epidata_loader_bn.normalize()
    epidata_loader_bn.add_lagged_features(lags = lags)
    epidata_loader_bn.finalize()
    epidata_loader_bn.retrieve_graph(graphtype)
    epidata_loader_bn.construct_dataloaders()


    # initating class
    graphconstruction = GraphConstructor(epidata = epidata_loader_bn)

    # generate graph
    graphconstruction.generate_graph(
        method='boolean_neighbors', 
        name_addition='',
    )

    graphconstruction.generate_graph(
        method='commuter', 
        self_connection='mean',
        scaling_method='symmetric',
        name_addition='t2000',
        commuter_type = 'static',
        commuting_threshold = 2000
    )

    graphconstruction.generate_graph(
        method='commuter', 
        self_connection='mean',
        scaling_method='rowwise',
        name_addition='t2000',
        commuter_type = 'static',
        commuting_threshold = 2000
    )

    graphconstruction.generate_graph(
        method='commuter', 
        self_connection='mean',
        scaling_method='zscore',
        name_addition='t2000',
        commuter_type = 'static',
        commuting_threshold = 2000
    )

    graphconstruction.generate_graph(
        method='commuter', 
        self_connection='mean',
        scaling_method='symmetric',
        name_addition='t1000',
        commuter_type = 'static',
        commuting_threshold = 1000
    )

    graphconstruction.generate_graph(
        method='commuter', 
        self_connection='mean',
        scaling_method='rowwise',
        name_addition='t1000',
        commuter_type = 'static',
        commuting_threshold = 1000
    )

    graphconstruction.generate_graph(
        method='commuter', 
        self_connection='mean',
        scaling_method='zscore',
        name_addition='t1000',
        commuter_type = 'static',
        commuting_threshold = 1000
    )

    graphconstruction.save_graph('all')

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
        when applicable, are saved under the attribute `dict_graphs`, a dictionary with
        graphname: {'edge_index': ..., 'edge_weight' : ...}

        the graphname will be:
            method + name_addition + scaling_method 

        with "_" as separator. It is possible that either, or both, of `name_addition` 
        and `scaling_method` are None.

        for renaming a graph structure afterewards, please adjust using the method:
        rename_graph
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
                     'config'   : asdict(graphconfig),
                     'summary'  : self._get_graph_summary(edge_index,edge_weight)}   

        self.graph_registry[graphname] = graphdict
        
        print(f'{graphname} generated')

        return self
        
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
