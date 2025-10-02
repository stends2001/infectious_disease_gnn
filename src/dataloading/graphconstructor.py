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

import matplotlib.cm as cm

cmap_red = cm.get_cmap('Reds')

class GraphConstructor:

    """
    calculates and saves graphs, edges and weights (if applicabe)


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
                 graph_dir:       str,
                 epidata:         EpiDataLoader,
                 id_col:          str = 'node'):  # <-- Added id_col for dynamic id field
        shapes               = epidata.data['context']['shapedata']    
        epidemiological_data = epidata.data['context']['epidemiological_data']
        self.population_data = epidemiological_data.groupby(id_col)['population_size'].mean().reset_index()

        self.shapes          = shapes.copy()
        self.edge_index      = None
        self.edge_weight     = None
        self.id_col          = id_col
        self.nuts_level      = epidata.nuts_level

        self.graph_dir       = os.path.join(graph_dir, f'{self.nuts_level}')
        self.dict_graphs     = {}

        os.makedirs(self.graph_dir, exist_ok=True)
        
    def generate_graph(self, 
                method: Literal['boolean_neighbors', 'identity_graph', 'mesh_graph', 
                                'distance_threshold', 'k_nearest', 'population_weighted', 
                                'gravity_model'] = 'boolean_neighbors',
                name_addition:   Optional[str]    = None,
                self_connection: Literal['1','0','mean'] = '1',
                scaling_method:  Optional[Literal['log', 'minmax']] = None,
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
        
        graphname = f'{method}_{name_addition}' if name_addition else f'{method}'
        graphname = f'{graphname}_{scaling_method}' if scaling_method else f'{graphname}'

        # Ensure IDs are integers and no missing
        shapes_cp              = self.shapes.dropna(subset=[self.id_col])
        shapes_cp[self.id_col] = shapes_cp[self.id_col].astype(int)

        node_ids = shapes_cp[self.id_col].dropna().astype(int).values

        if method == 'boolean_neighbors':
            edges, weights = generate_boolean_neighbors(shapes_cp, id_col=self.id_col)

        elif method == 'identity_graph':
            edges, weights = generate_identity_graph(shapes_cp, id_col=self.id_col)

        elif method == 'mesh_graph':
            edges, weights = generate_mesh_graph(shapes_cp, id_col=self.id_col)

        elif method == 'distance_threshold':
            max_distance = kwargs.get('max_distance', 100_000)
            edges, weights = generate_distance_threshold_graph(shapes_cp, max_distance, id_col=self.id_col)

        elif method == 'k_nearest':
            k = kwargs.get('k', 5)
            edges, weights = generate_k_nearest_graph(shapes_cp, k, id_col=self.id_col)

        elif method == 'population_weighted':
            max_distance = kwargs.get('max_distance', 100_000)
            if self.population_data is None:
                raise ValueError("population_data required for population_weighted graph")
            edges, weights = generate_population_weighted_graph(
                shapes_cp, self.population_data, max_distance, id_col=self.id_col)
            
        elif method == 'gravity_model':
            max_distance         = kwargs.get('max_distance', 300_000)
            alpha                = kwargs.get('alpha', 2.0)
            density_control      = kwargs.get('density_control', 'top_k')
            k                    = kwargs.get('k', 10)
            weight_threshold     = kwargs.get('weight_threshold', None)
            distance_decay_factor= kwargs.get('distance_decay_factor', 1.0)
            if self.population_data is None:
                raise ValueError("population_data required for gravity_model graph")
                
            edges, weights       = generate_gravity_model_graph(
                shapes_cp, self.population_data, max_distance, 
                alpha, id_col=self.id_col, density_control=density_control, 
                k=k, weight_threshold=weight_threshold,
                distance_decay_factor=distance_decay_factor)
        else:
            raise ValueError(f"Unknown method '{method}'")

        # if weights is undefined, give 1 everywhere
        if weights is None:
            weights = [1 for _ in edges]


        # Self loops
        if method not in ['identity_graph', 'mesh_graph']:
            self_loops = [(nid, nid) for nid in node_ids]
            edges += self_loops

            # Handle self-connection weights
            if self_connection == '1':
                # Use max weight value for self-connections
                max_weight = max(weights) if weights else 1
                weights += [max_weight for _ in node_ids]
            elif self_connection == '0':
                # Use 0 for self-connections
                weights += [0 for _ in node_ids]
            elif self_connection == 'mean':
                mean_weight = mean(weights)
                weights += [mean_weight for _ in node_ids]
                
              
        # remove zero-valued connections
        edges = [edge for edge, weight in zip(edges, weights) if weight != 0]
        weights = [weight for weight in weights if weight != 0]
        
        edge_weight = torch.tensor(weights, dtype=torch.float)

        edge_index  = torch.tensor(edges, dtype=torch.long).t().contiguous()


        # Scale weights if requested and edge_weight is not None
        if scaling_method is not None:
            if scaling_method == 'minmax':
                min_w = edge_weight.min()
                max_w = edge_weight.max()
                if max_w > min_w:
                    edge_weight = (edge_weight - min_w) / (max_w - min_w)
                else:
                    edge_weight = torch.zeros_like(edge_weight)

            elif scaling_method == 'log':
                edge_weight = torch.log1p(edge_weight)
                max_w = edge_weight.max()
                if max_w > 0:
                    edge_weight = edge_weight / max_w
                else:
                    edge_weight = torch.zeros_like(edge_weight)
            else:
                raise ValueError(f"Unknown scaling_method '{scaling_method}'")            

        self.dict_graphs[graphname] = {'edge_index': edge_index,
                                       'edge_weight': edge_weight}
        
        print(f'{graphname} generated')
        return self
        
    def preview_graph(self, 
                    graphname: str,
                    node_idx: int    = 11,
                    num_hops: int    = 1) -> 'GraphConstructor':

        """
        previews a subset of the graph, specific to the node_idx,
        with the level of neighborhoods as put in.
        """

        graph             = self.dict_graphs[graphname]
        global_edge_index = graph['edge_index']
        global_edge_weight= graph['edge_weight']

        mask              = global_edge_index[0] == node_idx
        local_nodes       = global_edge_index[1][mask]

        local_shapes      = self.shapes[self.shapes['node'].isin(local_nodes.numpy())]
        node_shape        = self.shapes[self.shapes['node'] == node_idx]

        quanti_graph = len(torch.unique(global_edge_weight)) > 2

        if quanti_graph:
            # Create figure with subplot layout: main plot on left, 2 subplots on right
            fig = plt.figure(figsize=(18, 8))
            gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])

            # Main plot that spans both rows in the left column
            ax_main = fig.add_subplot(gs[:, 0])

            # Local plot (top right)
            ax_local = fig.add_subplot(gs[0, 1])

            # Histogram plot (bottom right)
            ax_hist = fig.add_subplot(gs[1, 1])

            # Set example titles and labels
            ax_main.set_title("Main Plot")
            ax_local.set_title("Local Plot")
            ax_hist.set_title("Histogram")
            ax_local.set_aspect('auto')
            ax_hist.set_aspect('auto')

            # Create custom colormap: lightgrey for 0, then Reds for positive values
            max_val = global_edge_weight.max().item()
            
            # Create boundaries for discrete mapping
            boundaries = [0, 0.001] + list(np.linspace(0.001, max_val, 10))

            colors = ['lightgrey'] + [cmap_red(i) for i in np.linspace(0.3, 1, len(boundaries)-2)]
            custom_cmap = mcolors.ListedColormap(colors)
            norm = mcolors.BoundaryNorm(boundaries, custom_cmap.N)

            # Background plot for both main and local
            self.shapes.plot(color='lightgrey', edgecolor='white', linewidth=0.1, ax=ax_main)
            self.shapes.plot(color='lightgrey', edgecolor='white', linewidth=0.1, ax=ax_local)

            if global_edge_weight is not None:
                local_edge_weight = global_edge_weight[mask]

                local_connectivity = pd.DataFrame({
                    "node": local_nodes.tolist(),
                    "weight": local_edge_weight.tolist()
                })
                
                merged_data = pd.merge(local_shapes, local_connectivity, on='node')
                
                # Plot on main axis with colorbar
                im_main = merged_data.plot(column='weight', cmap=custom_cmap, norm=norm,
                                        edgecolor='black', linewidth=0.1, ax=ax_main)
                
                # Plot on local axis
                merged_data.plot(column='weight', cmap=custom_cmap, norm=norm,
                            edgecolor='black', linewidth=0.1, ax=ax_local)
                
                # Add colorbar to main plot (full height)
                sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax_main, shrink=1.0)
                cbar.set_label('Edge Weight', rotation=270, labelpad=20)
                
                # Add custom patch for 0 value in colorbar
                from matplotlib.patches import Rectangle
                cbar.ax.add_patch(Rectangle((0, 0), 1, 0.001/max_val, facecolor='lightgrey', 
                                        edgecolor='black', linewidth=0.5))
                
                # Node shape with its own connection strength on local plot
                # First check if node has outgoing connections
                if len(local_connectivity) > 0:
                    # Check if the node_idx itself appears in the connectivity (has self-loop or is in local_nodes)
                    node_in_local = node_idx in local_connectivity['node'].values
                    if node_in_local:
                        node_connectivity = local_connectivity[local_connectivity['node'] == node_idx]
                        node_connectivity_shapes = pd.merge(node_shape, node_connectivity, on='node')
                        node_connectivity_shapes.plot(column='weight', cmap=custom_cmap, norm=norm,
                                                    edgecolor='black', linewidth=2, ax=ax_local)
                    else:
                        # Node not in connectivity, color by its connection strength to others or default
                        # Use the mean weight of its connections as a proxy
                        mean_weight = local_edge_weight.mean().item()
                        node_color = custom_cmap(norm(mean_weight))
                        node_shape.plot(color=node_color, edgecolor='black', linewidth=2, ax=ax_local)
                else:
                    # If no connections, use minimum color (lightgrey)
                    node_shape.plot(color='lightgrey', edgecolor='black', linewidth=2, ax=ax_local)
                
                # Histogram (horizontal)
                sns.histplot(y=global_edge_weight.numpy(), ax=ax_hist, bins=20, color='red', alpha=0.7)
                ax_hist.set_title('Global Edge Weights Distribution', fontsize=10)
                ax_hist.set_ylabel('Edge Weight', fontsize=9)
                ax_hist.set_xlabel('Count', fontsize=9)

            # Plot node in blue on main plot
            node_shape.plot(color='blue', edgecolor='black', linewidth=2, ax=ax_main)
            
            # Set limits for local plot to zoom in
            if len(local_shapes) > 0:
                bounds = local_shapes.total_bounds
                margin = 0.1 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
                ax_local.set_xlim(bounds[0] - margin, bounds[2] + margin)
                ax_local.set_ylim(bounds[1] - margin, bounds[3] + margin)
            
            # Match the width of ax_bar to ax_local by getting the xlim
            bar_xlim = ax_local.get_xlim()
            bar_width = bar_xlim[1] - bar_xlim[0]
            
            # Remove ticks and labels
            for ax in [ax_main, ax_local]:
                ax.tick_params(left=False, right=False, bottom=False, top=False,
                            labelleft=False, labelbottom=False)
            
            ax_local.set_title('Local Neighborhood', fontsize=10)

        else:
            # Non-quantitative graph
            fig = plt.figure(figsize=(18, 8))
            gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])

            # Main plot that spans both rows in the left column
            ax_main = fig.add_subplot(gs[:, 0])

            # Local plot (top right)
            ax_local = fig.add_subplot(gs[0, 1])

            # Histogram plot (bottom right)
            ax_bar = fig.add_subplot(gs[1, 1])

            # Set example titles and labels
            ax_main.set_title("Main Plot")
            ax_local.set_title("Local Plot")
            ax_bar.set_title("Histogram")
            ax_local.set_aspect('auto')
            ax_bar.set_aspect('auto')

            # Background plot for both
            self.shapes.plot(color='lightgrey', edgecolor='white', linewidth=0.1, ax=ax_main)
            self.shapes.plot(color='lightgrey', edgecolor='white', linewidth=0.1, ax=ax_local)

            # Plot local connections
            local_shapes.plot(color='red', edgecolor='white', linewidth=0.1, alpha=0.6, ax=ax_main)
            local_shapes.plot(color='red', edgecolor='white', linewidth=0.1, alpha=0.6, ax=ax_local)

            # Plot node
            node_shape.plot(color='blue', edgecolor='black', linewidth=2, ax=ax_main)

            # Add legend patches for main plot
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightgrey', label='No Connection (0)'),
                Patch(facecolor='red', alpha=0.6, label='Connection (1)')
            ]
            ax_main.legend(handles=legend_elements, loc='upper right')

            # Set limits for local plot
            if len(local_shapes) > 0:
                bounds = local_shapes.total_bounds
                margin = 0.1 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
                ax_local.set_xlim(bounds[0] - margin, bounds[2] + margin)
                ax_local.set_ylim(bounds[1] - margin, bounds[3] + margin)

            # Bar plot showing number of connections (horizontal bars)
            connection_counts = {'No Connection': len(self.shapes) - len(local_shapes) - 1, 
                            'Connected': len(local_shapes)}
            
            # Create horizontal bar plot with same width as local neighborhood
            bars = ax_bar.barh(list(connection_counts.keys()), list(connection_counts.values()), 
                            color=['lightgrey', 'red'], alpha=0.7, height=0.4)
            ax_bar.set_title('Connection Summary', fontsize=10)
            ax_bar.set_xlabel('Count', fontsize=9)
            
            # Set the x-axis limits to match the proportional width
            max_count = max(connection_counts.values())
            ax_bar.set_xlim(0, max_count * 1.2)  # Add some padding
            
            # Add value labels
            for bar, count in zip(bars, connection_counts.values()):
                width = bar.get_width()
                ax_bar.text(width + max_count * 0.02, bar.get_y() + bar.get_height()/2.,
                        f'{count}', ha='left', va='center', fontsize=9)

            # Remove ticks for spatial plots
            for ax in [ax_main, ax_local]:
                ax.tick_params(left=False, right=False, bottom=False, top=False,
                            labelleft=False, labelbottom=False)
            
            ax_local.set_title('Local Neighborhood', fontsize=10)

        title = f'Subgraph of {graphname}, node {node_idx}' if num_hops==1 else f'Subgraph of {graphname}, node {node_idx}\nneighborhood level {num_hops}'
        ax_main.set_title(title, fontsize=12)

        plt.tight_layout()
        fig.show()
        return self

    def preview_shape_object(self) -> 'GraphConstructor':

        global_shapes = self.shapes
        fig, ax_main = plt.subplots(figsize = (10,8))

        global_shapes.plot(color = 'lightgrey',
                                    edgecolor='white',
                                    linewidth= 0.1,
                                    ax = ax_main)
        # get point - coordinates
        points   = global_shapes.geometry.centroid
        n_points = len(points)
        palette  = sns.color_palette("Blues", n_points)
        np.random.shuffle(palette)      
        
        # Plot points with random blue shades
        for i, point in enumerate(points):
            x, y = point.coords[0]  # Get coordinates from the point
            ax_main.plot(x, y, 'o', color=palette[i], markersize=5.5, 
                        markeredgecolor='black', markeredgewidth=0.6)

        ax_main.set_title('Shape object')

        ax_main.tick_params(
            left=False, right=False, bottom=False, top=False,  # no ticks
            labelleft=False, labelbottom=False                 # no labels
        )        

        fig.show()
        return self

    def rename_graph(self, old_graphname:str, 
                     new_graphname: str) -> 'GraphConstructor':

        self.dict_graphs[new_graphname] = self.dict_graphs[old_graphname]

        del self.dict_graphs[old_graphname]

        print(f'{old_graphname} has been replaced by {new_graphname}')

        return self 

    def save_graph(self, graphname: Union[str,List[str]] = 'all') -> 'GraphConstructor':

        """Save edge index (and if applicable edge weight) from dictionary"""

        if graphname == ['all']:
            graphname = 'all'

        if graphname == 'all':
            graphnames = list(self.dict_graphs.keys())

        elif isinstance(graphname, str):
            graphnames = [graphname]

        else:
            raise ValueError(f'Please provide a list or string for the graphname.')

        for graphname in graphnames:

            graph = self.dict_graphs[graphname]
            edge_index = graph['edge_index']
            edge_weight= graph['edge_weight']

            torch.save(edge_index, os.path.join(self.graph_dir, f'{graphname}_edge_index.pt'))
            print(f'edge index {graphname} saved to {self.graph_dir}')

            if edge_weight is not None:
                torch.save(edge_weight, os.path.join(self.graph_dir, f'{graphname}_edge_weight.pt'))
                print(f'edge weight {graphname} saved to {self.graph_dir}')

        return self

def generate_boolean_neighbors(df, id_col='node') -> Tuple[List, None]:
    df = df[[id_col,'geometry']]
    df = df.sort_values(id_col).reset_index(drop=True)
    neighbors = gpd.sjoin(df, df, how='inner', predicate='touches').reset_index(drop=False)
    neighbors = neighbors[neighbors[f'{id_col}_left'] != neighbors[f'{id_col}_right']]
    edges = list(zip(neighbors[f'{id_col}_left'], neighbors[f'{id_col}_right']))
    edges += [(t, s) for s, t in edges]
    edges = list(set(edges))
    return edges, None

def generate_identity_graph(df, id_col='id') -> Tuple[List, None]:
    df = df[[id_col]].sort_values(id_col).reset_index(drop=True)
    node_ids = df[id_col].dropna().astype(int).values
    edges = [(int(nid), int(nid)) for nid in node_ids]
    return edges, None

def generate_mesh_graph(df, id_col='id') -> Tuple[List, None]:
    df = df[[id_col]].sort_values(id_col).reset_index(drop=True)
    node_ids = df[id_col].dropna().astype(int).values
    edges = [(int(s), int(t)) for s in node_ids for t in node_ids]
    return edges, None
    
def generate_distance_threshold_graph(df, max_distance, id_col='id') -> Tuple[List, None]:
    df = df[[id_col, 'geometry']].sort_values(id_col).reset_index(drop=True)
    centroids = df.geometry.centroid
    coords = np.array([[p.x, p.y] for p in centroids])
    distances = euclidean_distances(coords)
    edges = []
    for i in range(len(df)):
        for j in range(len(df)):
            if distances[i, j] <= max_distance:
                edges.append((int(df.iloc[i][id_col]), int(df.iloc[j][id_col])))

    return edges, None

def generate_k_nearest_graph(df, k, id_col='id') -> Tuple[List, None]:
    df = df[[id_col, 'geometry']].sort_values(id_col).reset_index(drop=True)
    centroids = df.geometry.centroid
    coords = np.array([[p.x, p.y] for p in centroids])
    distances = euclidean_distances(coords)
    edges = []
    for i in range(len(df)):
        nearest_indices = np.argsort(distances[i])[1:k+1]
        for j in nearest_indices:
            edges.append((int(df.iloc[i][id_col]), int(df.iloc[j][id_col])))

    return edges, None

def generate_population_weighted_graph(df, population_data, max_distance, id_col='id') -> Tuple[List, List]:
    df = df[[id_col, 'geometry']].sort_values(id_col).reset_index(drop=True)
    df = df.merge(population_data[[id_col, 'population_size']], on=id_col, how='left')
    df['population_size'] = df['population_size'].fillna(df['population_size'].mean())
    centroids = df.geometry.centroid
    coords = np.array([[p.x, p.y] for p in centroids])
    distances = euclidean_distances(coords)
    edges = []
    weights = []

    for i in range(len(df)):
        for j in range(len(df)):
            if distances[i, j] <= max_distance and i != j:
                edges.append((int(df.iloc[i][id_col]), int(df.iloc[j][id_col])))
                pop_i = df.iloc[i]['population_size']
                pop_j = df.iloc[j]['population_size']
                dist = distances[i, j]
                weight = (pop_i * pop_j) / (dist + 1)
                weights.append(weight)

    
    return edges, weights

def generate_gravity_model_graph(df, population_data, max_distance, alpha=2.0, id_col='id', 
                                 density_control='top_k', k=10, weight_threshold=None, 
                                 distance_decay_factor=1.0) -> Tuple[List, List]:
    """
    Generate gravity model graph with density control options.
    
    Parameters:
    - density_control: 'top_k', 'threshold', 'adaptive', or 'distance_bands'
    - k: number of top connections per node (for top_k method)
    - weight_threshold: minimum weight to keep edge (for threshold method)
    - distance_decay_factor: additional distance penalty factor
    """
    df = df[[id_col, 'geometry']].sort_values(id_col).reset_index(drop=True)
    df = df.merge(population_data[[id_col, 'population_size']], on=id_col, how='left')
    df['population_size'] = df['population_size'].fillna(df['population_size'].mean())
    
    centroids = df.geometry.centroid
    coords = np.array([[point.x, point.y] for point in centroids])
    distances = euclidean_distances(coords)

    # Calculate all potential weights first
    all_weights = []
    all_edges = []
    
    for i in range(len(df)):
        node_weights = []
        node_edges = []
        
        for j in range(len(df)):
            if distances[i, j] <= max_distance and i != j:
                pop_i = df.iloc[i]['population_size']
                pop_j = df.iloc[j]['population_size']
                dist = distances[i, j]
                
                # Enhanced gravity model with additional distance decay
                weight = (pop_i * pop_j) / ((dist * distance_decay_factor)**alpha + 1e-6)
                
                node_weights.append((weight, j))
                node_edges.append((int(df.iloc[i][id_col]), int(df.iloc[j][id_col])))
        
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
            close_band = [x for x in node_weights_with_dist if x[2] <= max_distance * 0.3]
            medium_band = [x for x in node_weights_with_dist if max_distance * 0.3 < x[2] <= max_distance * 0.7]
            far_band = [x for x in node_weights_with_dist if x[2] > max_distance * 0.7]
            
            selected_items = close_band[:6] + medium_band[:3] + far_band[:1]  # 6+3+1=10 connections max
            selected = [(w, j) for w, j, d in selected_items]
        
        else:
            selected = node_weights  # No density control
        
        # Add selected edges and weights
        for idx, (weight, j) in enumerate(selected):
            all_edges.append(node_edges[node_weights.index((weight, j))])
            all_weights.append(weight)

    # Add self-loops
        
    return all_edges, all_weights