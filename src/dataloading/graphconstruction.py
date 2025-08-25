from typing import Literal
import geopandas as gpd
import torch

def generate_edge_index(shapes: gpd.GeoDataFrame,
                        method: Literal['boolean_neighbors', 'identity_graph', 'mesh_graph'] = 'boolean_neighbors',
                        ):

    df = shapes.copy()

    if method == 'boolean_neighbors':
        edge_index = generate_boolean_neighbors(df)
    elif method == 'identity_graph':
        edge_index = generate_identity_graph(df)
    elif method == 'mesh_graph':
        edge_index = generate_mesh_graph(df)
    else:
        raise ValueError(f"Unknown method '{method}'")

    torch.save(edge_index, f'src/dataloading/graphs/{method}_edge_index.pt')   


def generate_boolean_neighbors(df):
    df = df[['id','geometry']]
    # Assume df is your GeoDataFrame with 'id' and 'geometry'
    df = df.sort_values('id').reset_index(drop=True)

    # Spatial join: find all pairs where geometries touch
    neighbors = gpd.sjoin(df, df, how='inner', predicate='touches').reset_index(drop = False)

    # Filter out self-joins (where the same polygon matches itself)
    neighbors = neighbors[neighbors.id_left != neighbors.id_right]

    # Extract source and target indices as edges
    edges = list(zip(neighbors.id_left, neighbors.id_right))

    # For undirected graph, add reverse edges
    edges += [(t, s) for s, t in edges]

    # Remove duplicates
    edges = list(set(edges))

    # Convert to PyTorch edge_index tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index


def generate_identity_graph(df: gpd.GeoDataFrame):
    """
    Generate an identity graph where each node has a self-loop edge only.
    """
    df = df[['id']].sort_values('id').reset_index(drop=True)
    node_ids = df['id'].values
    edges = [(nid, nid) for nid in node_ids]  # self-loops

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def generate_mesh_graph(df: gpd.GeoDataFrame):
    """
    Generate a fully connected (mesh) graph where every node is connected to every other node (including self-loops).
    """
    df = df[['id']].sort_values('id').reset_index(drop=True)
    node_ids = df['id'].values

    edges = []
    for s in node_ids:
        for t in node_ids:
            edges.append((s, t))  # all pairs including self loops

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index