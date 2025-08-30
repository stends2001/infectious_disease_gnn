import os
from typing import Optional, Literal
import geopandas as gpd
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from typing import Optional, Literal

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

    >>> graphconstruction = GraphConstructor(graph_dir = 'src/dataloading/graphs', population_data=epidata.population_by_node, shapes = epidata.shapedata, id_col='node')

    >>> graphconstruction.generate(
    >>>    method='boolean_neighbors', 
    >>>    name_addition='',
    >>> )

    >>> graphconstruction.generate(
    >>>    method='identity_graph', 
    >>>    name_addition='',
    >>> )

    # Example 1: Top-k connections (recommended starting point)
    >>> graphconstruction.generate(
    >>>    method='gravity_model', 
    >>>    name_addition='sparse_topk',
    >>>    scaling_method='minmax',
    >>>    max_distance=200_000,  # Reduce max distance
    >>>    alpha=2.0,
    >>>    density_control='top_k',
    >>>    k=8,  # Each node connects to max 8 others
    >>>    distance_decay_factor=1.5  # Increase distance penalty
    >>> )

    # Example 2: Distance bands approach
    >>> graphconstruction.generate(
    >>>     method='gravity_model', 
    >>>     name_addition='distance_bands',
    >>>     scaling_method='minmax',
    >>>     max_distance=250_000,
    >>>     alpha=1.5,
    >>>     density_control='distance_bands'
    >>> )

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
        os.makedirs(self.graph_dir, exist_ok=True)
        
    def generate(self, 
                method: Literal['boolean_neighbors', 'identity_graph', 'mesh_graph', 
                                'distance_threshold', 'k_nearest', 'population_weighted', 
                                'gravity_model'] = 'boolean_neighbors',
                name_addition: str = None,
                scaling_method: Optional[Literal['log', 'minmax']] = None,
                **kwargs):
        # Ensure IDs are integers and no missing
        self.shapes = self.shapes.dropna(subset=[self.id_col])
        self.shapes[self.id_col] = self.shapes[self.id_col].astype(int)

        if method == 'boolean_neighbors':
            self.edge_index = generate_boolean_neighbors(self.shapes, id_col=self.id_col)
            self.edge_weight = None
        elif method == 'identity_graph':
            self.edge_index = generate_identity_graph(self.shapes, id_col=self.id_col)
            self.edge_weight = None
        elif method == 'mesh_graph':
            self.edge_index = generate_mesh_graph(self.shapes, id_col=self.id_col)
            self.edge_weight = None
        elif method == 'distance_threshold':
            max_distance = kwargs.get('max_distance', 100_000)
            self.edge_index = generate_distance_threshold_graph(self.shapes, max_distance, id_col=self.id_col)
            self.edge_weight = None
        elif method == 'k_nearest':
            k = kwargs.get('k', 5)
            self.edge_index = generate_k_nearest_graph(self.shapes, k, id_col=self.id_col)
            self.edge_weight = None
        elif method == 'population_weighted':
            max_distance = kwargs.get('max_distance', 100_000)
            if self.population_data is None:
                raise ValueError("population_data required for population_weighted graph")
            self.edge_index, self.edge_weight = generate_population_weighted_graph(
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
            self.edge_index, self.edge_weight = generate_gravity_model_graph(
                self.shapes, self.population_data, max_distance, alpha, id_col=self.id_col,
                density_control=density_control, k=k, weight_threshold=weight_threshold,
                distance_decay_factor=distance_decay_factor)
        else:
            raise ValueError(f"Unknown method '{method}'")

        # Scale weights if requested and edge_weight is not None
        if self.edge_weight is not None and scaling_method is not None:
            if scaling_method == 'minmax':
                min_w = self.edge_weight.min()
                max_w = self.edge_weight.max()
                if max_w > min_w:
                    self.edge_weight = (self.edge_weight - min_w) / (max_w - min_w)
                else:
                    self.edge_weight = torch.zeros_like(self.edge_weight)
            elif scaling_method == 'log':
                self.edge_weight = torch.log1p(self.edge_weight)
                max_w = self.edge_weight.max()
                if max_w > 0:
                    self.edge_weight = self.edge_weight / max_w
                else:
                    self.edge_weight = torch.zeros_like(self.edge_weight)
            else:
                raise ValueError(f"Unknown scaling_method '{scaling_method}'")

        # Save results
        torch.save(self.edge_index, os.path.join(self.graph_dir, f'{method}_{name_addition}_edge_index.pt'))
        print(f'edge index {method} saved to {self.graph_dir}')

        if self.edge_weight is not None:
            torch.save(self.edge_weight, os.path.join(self.graph_dir, f'{method}_{name_addition}_edge_weight.pt'))
            print(f'edge weight {method} saved to {self.graph_dir}')

        return self.edge_index, self.edge_weight
# Updated graph generation functions to include dynamic id_col param:

def generate_boolean_neighbors(df, id_col='node'):
    df = df[[id_col,'geometry']]
    df = df.sort_values(id_col).reset_index(drop=True)
    neighbors = gpd.sjoin(df, df, how='inner', predicate='touches').reset_index(drop=False)
    neighbors = neighbors[neighbors[f'{id_col}_left'] != neighbors[f'{id_col}_right']]
    edges = list(zip(neighbors[f'{id_col}_left'], neighbors[f'{id_col}_right']))
    edges += [(t, s) for s, t in edges]
    edges = list(set(edges))

    node_ids = df[id_col].dropna().astype(int).values
    self_loops = [(nid, nid) for nid in node_ids]
    edges += self_loops

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def generate_identity_graph(df, id_col='id'):
    df = df[[id_col]].sort_values(id_col).reset_index(drop=True)
    node_ids = df[id_col].dropna().astype(int).values
    edges = [(int(nid), int(nid)) for nid in node_ids]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def generate_mesh_graph(df, id_col='id'):
    df = df[[id_col]].sort_values(id_col).reset_index(drop=True)
    node_ids = df[id_col].dropna().astype(int).values
    edges = [(int(s), int(t)) for s in node_ids for t in node_ids]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index
    
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

    node_ids = df[id_col].dropna().astype(int).values
    self_loops = [(nid, nid) for nid in node_ids]
    edges += self_loops

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

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

    node_ids = df[id_col].dropna().astype(int).values
    self_loops = [(nid, nid) for nid in node_ids]
    edges += self_loops

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

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

    node_ids = df[id_col].dropna().astype(int).values
    self_loops = [(nid, nid) for nid in node_ids]
    edges += self_loops
    weights += [0.0 for _ in node_ids]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight

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
    node_ids = df[id_col].dropna().astype(int).values
    self_loops = [(nid, nid) for nid in node_ids]
    all_edges += self_loops
    all_weights += [0.0 for _ in node_ids]  # or use small positive value like 1.0

    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(all_weights, dtype=torch.float)
    
    return edge_index, edge_weight