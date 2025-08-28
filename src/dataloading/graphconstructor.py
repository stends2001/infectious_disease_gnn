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

    >>> graphconstruction = GraphConstructor(graph_dir = 'src/dataloading/graphs', population_data=epidata.population_by_node, shapes = epidata.shapedata, id_col='node')
    >>> graphconstruction.generate(method = 'gravity_model', name_addition= 'kreis_minmax',scaling_method='minmax')
    
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
                 scaling_method: Optional[Literal['log', 'minmax']] = None,  # <-- New param
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
            if self.population_data is None:
                raise ValueError("population_data required for gravity_model graph")
            self.edge_index, self.edge_weight = generate_gravity_model_graph(
                self.shapes, self.population_data, max_distance, alpha, id_col=self.id_col)
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

def generate_gravity_model_graph(df, population_data, max_distance, alpha=2.0, id_col='id'):
    df = df[[id_col, 'geometry']].sort_values(id_col).reset_index(drop=True)
    df = df.merge(population_data[[id_col, 'population_size']], on=id_col, how='left')
    df['population_size'] = df['population_size'].fillna(df['population_size'].mean())
    centroids = df.geometry.centroid
    coords = np.array([[point.x, point.y] for point in centroids])
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
                weight = (pop_i * pop_j) / (dist**alpha + 1e-6)
                weights.append(weight)

    node_ids = df[id_col].dropna().astype(int).values
    self_loops = [(nid, nid) for nid in node_ids]
    edges += self_loops
    weights += [0.0 for _ in node_ids]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight
