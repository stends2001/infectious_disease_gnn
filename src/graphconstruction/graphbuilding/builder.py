from typing import List, Tuple
import numpy as np 
from sklearn.metrics.pairwise import euclidean_distances
import geopandas as gpd 
import pandas as pd
from numpy.typing import NDArray
import numpy as np

from ...utils import registry_method, get_registered_methods, MethodNotInRegistry, CRS_DEGREES, CRS_GERMANY_METRES
from ...utils.types import GraphType

class GraphBuilder:
    """ 
    """
    def __init__(self,
                 id_col:            str,
                 token_col:         str,                 
                 shape_data:        gpd.GeoDataFrame,
                 population_data:   pd.DataFrame):
        
        self.id_col     = id_col 
        self.token_col  = token_col  

        self.shp_data   = shape_data
        self.pop_data   = population_data

        self.methods = get_registered_methods(self.__class__)

    def build(self, method: GraphType, *args, **kwargs) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Main function to be called for building a graph structure. args and kwargs depend on the given method.

        Returns
        -------
        edge_index: List[Tuple[int,int]]
    
        edge_weight: List[float]
        """
        if method not in self.methods:
            raise MethodNotInRegistry(method, list(self.methods))

        return getattr(self, method)(*args, **kwargs)
    
    @registry_method
    def identity(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Build an identity graph; each node is connected to itself only.
        No parameters required.
        """
        node_ids    = list(self.shp_data[self.token_col].unique())
        edges       = [(int(nid), int(nid)) for nid in node_ids]
        weights     = [float(1) for i in range(len(edges))]
        return edges, weights

    @registry_method
    def geographical_contiguity(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Build a geographical neighbors graph; each node is connected to its geographical neighbors only.
        No parameters required.
        """        
        neighbors   = gpd.sjoin(self.shp_data, self.shp_data, how='inner', predicate='touches').reset_index(drop=False)
        neighbors   = neighbors[neighbors[f'{self.token_col}_left'] != neighbors[f'{self.token_col}_right']]
        edges       = list(zip(neighbors[f'{self.token_col}_left'], neighbors[f'{self.token_col}_right']))
        edges      += [(t, s) for s, t in edges]
        edges       = list(set(edges))
        weights     = [float(1) for i in range(len(edges))]
        return edges, weights

    @registry_method
    def gravity_model(self,
                    alpha:             float = 2.0,
                    epsilon:           float = 1e-6,
                    decay:             float = 1.0,
                    max_distance:      float = 100_000) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Build a gravity-model based graph: connection strength depends on distance and population size.

        The Gravity formula is as follows:
        edge_weight_{i,j} = pop_i * pop_j / ((distance * decay) ** alpha + epsilon) if distance < max_distance else 0

        Parameters
        ----------
        alpha: float = 2
            distance exponent
        epsilon: float = 1e-6
            numerical stability factor (prevents division by 0)
        decay:  float = 1.0
            higher means stronger decay with distance
        max_distance: float = 100_000
            maximum distance between two nodes within which they may still be connected (in m! not in km)
        """
        gdfc             = self.shp_data[[self.token_col, "geometry"]].sort_values(self.token_col).reset_index(drop=True)
        population_data  = self.pop_data.sort_values(self.token_col).reset_index(drop=True)

        dfc_projected               = gdfc.to_crs(CRS_GERMANY_METRES)
        dfc_projected['geometry']   = dfc_projected.geometry.centroid
        coords                      = np.column_stack([dfc_projected.geometry.x, dfc_projected.geometry.y])

        pop                         = population_data["population_size"].to_numpy(dtype=np.float64)
        node_ids                    = population_data[self.token_col].values

        distance_matrix = euclidean_distances(coords)

        # Gravity weights — fully vectorized
        pop_product: NDArray[np.float64]    = np.outer(pop, pop)
        denom                               = (distance_matrix * decay) ** alpha + epsilon
        weight_matrix                       = pop_product / denom

        # Remove self-loops and edges beyond max_distance
        np.fill_diagonal(weight_matrix, 0)
        weight_matrix[distance_matrix > max_distance] = 0

        src, dst = np.nonzero(weight_matrix)
        weights: NDArray[np.float64]  = weight_matrix[src, dst]

        edges_list                      = list(zip(node_ids[src].astype(int).tolist(), node_ids[dst].astype(int).tolist()))
        weights_list: List[float]       = weights.tolist()

        return edges_list, weights_list
 
    @registry_method
    def random(self, seed: int = 42, k: int = 5) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Build a random graph where each node has approximately k neighbors.

        Parameters
        ----------
        seed: int = 42
            random seed
        k: int = 5
            target average degree per node
        """
        rng      = np.random.default_rng(seed)
        node_ids = list(self.shp_data[self.token_col].unique())
        n        = len(node_ids)

        # k neighbors out of n-1 possible -> probability per edge
        p = k / (n - 1)

        edges   = []
        weights = []

        for i in node_ids:
            for j in node_ids:
                if i >= j:
                    continue
                if rng.random() < p:
                    edges.append((int(i), int(j)))
                    edges.append((int(j), int(i)))
                    weights.extend([1.0, 1.0])

        return edges, weights
    
    @registry_method    
    def fully_connected(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Build a fully connected graph where each node is connected to all other nodes with weight 1.
        NOTE: such a graph is computationally expensive!
        """
        node_ids = list(self.shp_data[self.token_col].unique())

        edges = [
            (int(i), int(j))
            for i in node_ids
            for j in node_ids
            if i != j
        ]
        weights = [1.0] * len(edges)

        return edges, weights    

    def __repr__(self) -> str:
        representation = f"<{self.__class__.__name__}>"
        return representation