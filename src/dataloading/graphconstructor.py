import os
from typing import Optional, Literal
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
                 graph_dir: str,
                 shapes: gpd.GeoDataFrame, 
                 population_data: Optional[pd.DataFrame] = None,
                 id_col: str = 'id'):  # <-- Added id_col for dynamic id field
        self.shapes          = shapes.copy()
        self.population_data = population_data
        self.edge_index      = None
        self.edge_weight     = None
        self.graph_dir       = graph_dir
        self.id_col          = id_col
        self.dict_graphs     = {}
        os.makedirs(self.graph_dir, exist_ok=True)
        
    def generate_graph(self, 
                method: Literal['boolean_neighbors', 'identity_graph', 'mesh_graph', 
                                'distance_threshold', 'k_nearest', 'population_weighted', 
                                'gravity_model'] = 'boolean_neighbors',
                name_addition: str = None,
                self_connection: Literal['1','0'] = None,
                scaling_method: Optional[Literal['log', 'minmax']] = None,
                **kwargs):
        
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
        self.shapes = self.shapes.dropna(subset=[self.id_col])
        self.shapes[self.id_col] = self.shapes[self.id_col].astype(int)

        node_ids = self.shapes[self.id_col].dropna().astype(int).values

        if method == 'boolean_neighbors':
            edges = generate_boolean_neighbors(self.shapes, id_col=self.id_col)
            weights = None

        elif method == 'identity_graph':
            edges = generate_identity_graph(self.shapes, id_col=self.id_col)
            weights = None

        elif method == 'mesh_graph':
            edges = generate_mesh_graph(self.shapes, id_col=self.id_col)
            weights = None

        elif method == 'distance_threshold':
            max_distance = kwargs.get('max_distance', 100_000)
            edges = generate_distance_threshold_graph(self.shapes, max_distance, id_col=self.id_col)
            weights = None

        elif method == 'k_nearest':
            k = kwargs.get('k', 5)
            edges = generate_k_nearest_graph(self.shapes, k, id_col=self.id_col)
            weights = None

        elif method == 'population_weighted':
            max_distance = kwargs.get('max_distance', 100_000)
            if self.population_data is None:
                raise ValueError("population_data required for population_weighted graph")
            edges, weights = generate_population_weighted_graph(
                self.shapes, self.population_data, max_distance, id_col=self.id_col)
            
        elif method == 'gravity_model':
            max_distance = kwargs.get('max_distance', 300_000)
            alpha = kwargs.get('alpha', 2.0)
            density_control = kwargs.get('density_control', 'top_k')
            k = kwargs.get('k', 10)
            weight_threshold = kwargs.get('weight_threshold', None)
            distance_decay_factor = kwargs.get('distance_decay_factor', 1.0)
            if self.population_data is None:
                raise ValueError("population_data required for gravity_model graph")
            edges, weights = generate_gravity_model_graph(
                self.shapes, self.population_data, max_distance, alpha, id_col=self.id_col,
                density_control=density_control, k=k, weight_threshold=weight_threshold,
                distance_decay_factor=distance_decay_factor)
        else:
            raise ValueError(f"Unknown method '{method}'")

        # Self loops
        if method not in ['identity_graph', 'mesh_graph']:
            self_loops = [(nid, nid) for nid in node_ids]
            edges += self_loops

            if weights is not None:

                if self_connection == '1':
                    weights += [max(weights) for _ in node_ids]

                if self_connection == '0':
                    weights += [0 for _ in node_ids]
                
                edge_weight = torch.tensor(weights, dtype=torch.float)

        if weights is None:
            edge_weight = weights

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()


        # Scale weights if requested and edge_weight is not None
        if edge_weight is not None and scaling_method is not None:
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
        
    def preview_graph(self, 
                      graphname: str,
                      node_idx: int    = 391,
                      num_hops: int    = 1):

        """
        previews a subset of the graph, specific to the node_idx,
        with the level of neighborhoods as put in.
        """

        graph = self.dict_graphs[graphname]
        global_edge_index = graph['edge_index']
        global_edge_weight= graph['edge_weight']

        mask             = global_edge_index[0] == node_idx
        local_nodes      = global_edge_index[1][mask]

        local_shapes     = self.shapes[self.shapes['node'].isin(local_nodes.numpy())]
        node_shape       = self.shapes[self.shapes['node'] == node_idx]

        norm = mcolors.Normalize(vmin=0, vmax=1)

        fig, ax_main = plt.subplots(figsize = (10,8))
        self.shapes.plot(color = 'lightgrey',
                                    edgecolor='white',
                                    linewidth= 0.1,
                                    ax = ax_main)

        if global_edge_weight is not None:
            fig, axes_sub = plt.subplots(1, 2, figsize = (15,4))
            local_edge_weight = global_edge_weight[mask]

            local_connectivity = pd.DataFrame({
                "node": local_nodes.tolist(),
                "weight": local_edge_weight.tolist()
            })
            pd.merge(local_shapes,local_connectivity, on = 'node').plot(column = 'weight',cmap ='Reds', legend = True ,edgecolor='black', linewidth= 0.1, ax = ax_main, norm = norm)
            pd.merge(local_shapes,local_connectivity, on = 'node').plot(column = 'weight',cmap ='Reds', legend = True ,edgecolor='black', linewidth= 0.1, ax = axes_sub[0], norm = norm)
            
            axes_sub[0].tick_params(
                left=False, right=False, bottom=False, top=False,  # no ticks
                labelleft=False, labelbottom=False                 # no labels
            )

            sns.histplot(global_edge_weight, ax = axes_sub[1])

            axes_sub[1].set_title('Global edge weights')
            
            

        else:
            local_shapes.plot(color = 'red', edgecolor='white', linewidth= 0.1, alpha = 0.6, ax = ax_main)

        node_shape.plot(color = 'blue', edgecolor='black', linewidth= 0.1, ax = ax_main)

        ax_main.tick_params(
            left=False, right=False, bottom=False, top=False,  # no ticks
            labelleft=False, labelbottom=False                 # no labels
        )

        title = f'Subgraph of {graphname}, node {node_idx}' if num_hops==1 else  f'Subgraph of {graphname}, node {node_idx}\nneighborhood level {num_hops}'

        ax_main.set_title(title)

        fig.show()

    def preview_shape_object(self):

        global_shapes = self.shapes
        fig, ax_main = plt.subplots(figsize = (10,8))

        global_shapes.plot(color = 'lightgrey',
                                    edgecolor='white',
                                    linewidth= 0.1,
                                    ax = ax_main)
        
        points = global_shapes.geometry.centroid

        import seaborn as sns
        import numpy as np

        n_points = len(points)
        palette = sns.color_palette("Blues", n_points)
        np.random.shuffle(palette)  # shuffle to get random shades        
        

        # Plot points with random blue shades
        for i, point in enumerate(points):
            ax_main.plot(point.x, point.y, 'o', color=palette[i], markersize=5.5, markeredgecolor = 'black', markeredgewidth = 0.6)

        ax_main.set_title('Shape object')

        ax_main.tick_params(
            left=False, right=False, bottom=False, top=False,  # no ticks
            labelleft=False, labelbottom=False                 # no labels
        )        

        fig.show()

    def rename_graph(self, old_graphname:str, 
                     new_graphname: str):

        self.dict_graphs[new_graphname] = self.dict_graphs[old_graphname]

        del self.dict_graphs[old_graphname]

        print(f'{old_graphname} has been replaced by {new_graphname}')

        return self 

    def save_graph(self, graphname):

        """Save edge index (and if applicable edge weight) from dictionary"""

        graph = self.dict_graphs[graphname]
        edge_index = graph['edge_index']
        edge_weight= graph['edge_weight']

        torch.save(edge_index, os.path.join(self.graph_dir, f'{graphname}_edge_index.pt'))
        print(f'edge index {graphname} saved to {self.graph_dir}')

        if edge_weight is not None:
            torch.save(edge_weight, os.path.join(self.graph_dir, f'{graphname}_edge_weight.pt'))
            print(f'edge weight {graphname} saved to {self.graph_dir}')

def generate_boolean_neighbors(df, id_col='node'):
    df = df[[id_col,'geometry']]
    df = df.sort_values(id_col).reset_index(drop=True)
    neighbors = gpd.sjoin(df, df, how='inner', predicate='touches').reset_index(drop=False)
    neighbors = neighbors[neighbors[f'{id_col}_left'] != neighbors[f'{id_col}_right']]
    edges = list(zip(neighbors[f'{id_col}_left'], neighbors[f'{id_col}_right']))
    edges += [(t, s) for s, t in edges]
    edges = list(set(edges))
    return edges

def generate_identity_graph(df, id_col='id'):
    df = df[[id_col]].sort_values(id_col).reset_index(drop=True)
    node_ids = df[id_col].dropna().astype(int).values
    edges = [(int(nid), int(nid)) for nid in node_ids]
    return edges

def generate_mesh_graph(df, id_col='id'):
    df = df[[id_col]].sort_values(id_col).reset_index(drop=True)
    node_ids = df[id_col].dropna().astype(int).values
    edges = [(int(s), int(t)) for s in node_ids for t in node_ids]
    return edges
    
def generate_distance_threshold_graph(df, max_distance, id_col='id'):
    df = df[[id_col, 'geometry']].sort_values(id_col).reset_index(drop=True)
    centroids = df.geometry.centroid
    coords = np.array([[p.x, p.y] for p in centroids])
    distances = euclidean_distances(coords)
    edges = []
    for i in range(len(df)):
        for j in range(len(df)):
            if distances[i, j] <= max_distance:
                edges.append((int(df.iloc[i][id_col]), int(df.iloc[j][id_col])))

    return edges

def generate_k_nearest_graph(df, k, id_col='id'):
    df = df[[id_col, 'geometry']].sort_values(id_col).reset_index(drop=True)
    centroids = df.geometry.centroid
    coords = np.array([[p.x, p.y] for p in centroids])
    distances = euclidean_distances(coords)
    edges = []
    for i in range(len(df)):
        nearest_indices = np.argsort(distances[i])[1:k+1]
        for j in nearest_indices:
            edges.append((int(df.iloc[i][id_col]), int(df.iloc[j][id_col])))

    return edges

def generate_population_weighted_graph(df, population_data, max_distance, id_col='id'):
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
                                 distance_decay_factor=1.0):
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