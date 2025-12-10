import os

from typing import Optional, Tuple, List, Union, Literal, Dict
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

from .containers import RawGraphStructure

from .commuterdataloader import CommuterDataLoader

class RawGraphConstructor:

    """
    creates and returns single RawGraphStructures;
    a List of edge_index and a List of edge_weights
    """

    def __init__(self,
                 shapedata: gpd.GeoDataFrame,
                 tokens,
                 id_col:    str = 'node',
                 ):
        
        self.shapedata    = shapedata
        self.id_col       = id_col
        self.tokens       = tokens

        if 'geometry' not in self.shapedata.columns:
            raise ValueError(f'GeoDataFrame must have a "geometry" column')    

        self.GENERATION_FUNCS ={
                            'geographic_neighbors'  : self._geographic_neighbors,
                            'identity'              : self._identity,
                            'mesh'                  : self._mesh,
                            'distance_threshold'    : self._distance_threshold,
                            'k_nearest'             : self._k_nearest,
                            'population_weighted'   : self._population_weighted,
                            'gravity_model'         : self._gravity_model,
                            'commuter'              : self._commuter
                            }           

    def generate_graph(self, method: str, **kwargs) -> RawGraphStructure:
        """
        collects the required generation-function and feeds in the kwargs

        Parameters
        ----------
        method: str
            which graph to construct
        """

        if method not in self.GENERATION_FUNCS:
            available = ', '.join(self.GENERATION_FUNCS.keys())
            raise ValueError(
                f"Unknown graph generation method: '{method}'. "
                f"Available methods: {available}"
            )
        
        edge_index, edge_weights = self.GENERATION_FUNCS[method](**kwargs)
        
        return RawGraphStructure(edge_index, edge_weights)      
        
    def _identity(self) -> Tuple[List[Tuple[int,int]], List[float]]:
        """
        creates an identity - graph:
        each node is connected only to itself with uniform edge_weights
        """
        dfc         = self.shapedata.copy()
        dfc         = dfc[[self.id_col, "geometry"]].sort_values(self.id_col).reset_index(drop=True)
        node_ids    = list(dfc[self.id_col].unique())
        edges       = [(int(nid), int(nid)) for nid in node_ids]
        weights     = [float(1) for i in range(len(edges))]
        return edges, weights

    def _geographic_neighbors(self) -> Tuple[List[Tuple[int,int]], List[float]]:
        """ 
        creates a geographic neighbors - graph:
        each node is connected to directly surrounding nodes with uniform edge_weights
        """
        dfc         = self.shapedata.copy()
        dfc         = dfc[[self.id_col, "geometry"]].sort_values(self.id_col).reset_index(drop=True)
        neighbors   = gpd.sjoin(dfc, dfc, how='inner', predicate='touches').reset_index(drop=False)
        neighbors   = neighbors[neighbors[f'{self.id_col}_left'] != neighbors[f'{self.id_col}_right']]
        edges       = list(zip(neighbors[f'{self.id_col}_left'], neighbors[f'{self.id_col}_right']))
        edges      += [(t, s) for s, t in edges]
        edges       = list(set(edges))
        weights     = [float(1) for i in range(len(edges))]
        return edges, weights

    def _mesh(self) -> Tuple[List[Tuple[int,int]], List[float]]:
        """
        creates a mesh - graph:
        each node is connected to every other node with uniform edge_weights
        """        
        dfc         = self.shapedata.copy()
        dfc         = dfc[[self.id_col, "geometry"]].sort_values(self.id_col).reset_index(drop=True)
        node_ids    = list(dfc[self.id_col].unique())
        edges       = [(int(s), int(t)) for s in node_ids for t in node_ids]
        weights     = [float(1) for i in range(len(edges))]        
        return edges, weights      

    def _gravity_model(self,
                       population_data:   pd.DataFrame, 
                       alpha:             float = 2.0,
                       epsilon:           float = 1e-6,
                       decay:             float = 1.0,
                       max_distance:      float = np.inf,
                       top_k:             Optional[int] = None) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        population data should only have column population_size and node
        """
        # sort data to ensure alignment
        dfc                         = self.shapedata.copy()
        dfc                         = dfc[[self.id_col, "geometry"]].sort_values(self.id_col).reset_index(drop=True)        
        population_data             = population_data.sort_values(self.id_col).reset_index(drop=True)

        dfc_centroids               = dfc.copy() 
        dfc_centroids['geometry']   = dfc.geometry.centroid
        coords                      = np.column_stack([dfc_centroids.geometry.x, dfc_centroids.geometry.y])

        # Compute Euclidean distances between all pairs
        distance_matrix = euclidean_distances(coords)

        edges   = []
        weights = []

        num_nodes = population_data[self.id_col].nunique()
        popvalues = population_data['population_size'].values
        node_ids  = population_data[self.id_col].values

        for i in range(num_nodes):
            node_i_id = int(node_ids[i])
            pop_i = popvalues[i]

            connections = []

            for j in range(num_nodes):
                if i == j:
                    continue

                d_ij = distance_matrix[i, j]
                if d_ij > max_distance:
                    continue

                pop_j = popvalues[j]
                node_j_id = int(node_ids[j])

                weight = (pop_i * pop_j) / ((d_ij * decay) ** alpha + epsilon)
                connections.append((node_i_id, node_j_id, weight))

            # Apply top-k filtering if specified
            if top_k is not None:
                connections = sorted(connections, key=lambda x: x[2], reverse=True)[:top_k]

            for source, target, weight in connections:
                edges.append((source, target))
                weights.append(weight)

        return edges, weights

    def _commuter(self, commuter_data: pd.DataFrame, commuting_threshold: int = 1_000, top_k: Optional[int] = None, **kwargs)  -> Tuple[List[Tuple[int,int]], List[float]]:
        """ 
        creates a commuter - graph

        Parameters
        ----------
        years: str = '2024'
            the year for which to retrieve the data
            when mode == 'dynamic', please ensure to input an iterable for years
        commuting_threshold: int = 1_000
            the threshold for number of commuters for nodes to be connected       
        top_k: int = None
            the maximum number of connections for each node
        """
  
        data = commuter_data.copy()

        data.loc[:, 'nuts3_work']      = data.loc[:, 'nuts3_work'].map(self.tokens)
        data.loc[:, 'nuts3_residence'] = data.loc[:, 'nuts3_residence'].map(self.tokens)

        data   = data[data['commuters'] > commuting_threshold]

        if top_k is not None:
            # Keep only top_k strongest links per source (nuts3_work)
            data = (
                data
                .sort_values(by='commuters', ascending=False)
                .groupby('nuts3_work', group_keys=False)
                .head(top_k)
            )

        edges = list(zip(
            data['nuts3_work'].astype(int),
            data['nuts3_residence'].astype(int)
        ))
        weights = data['commuters'].astype(float).tolist()


        return edges, weights
                         
    def _distance_threshold(self, max_distance: float) -> Tuple[List[Tuple[int,int]], None]:
        """
        creates a distance_threshold - graph:
        each node is connected to all other nodes within max_distance with uniform edge_weights
        """        
        dfc         = self.gdf[[self.id_col, 'geometry']].sort_values(self.id_col).reset_index(drop=True)
        centroids   = dfc.geometry.centroid
        coords      = np.column_stack([centroids.x, centroids.y])
        distances   = euclidean_distances(coords)
        edges       = []
        for i in range(len(dfc)):
            for j in range(len(dfc)):
                if distances[i, j] <= max_distance:
                    edges.append((int(dfc.iloc[i][self.id_col]), int(dfc.iloc[j][self.id_col])))
        return edges, None

    def _k_nearest(self, k: int) -> Tuple[List[Tuple[int,int]], None]:
        """
        creates a k-nearest-neighbors - graph:
        each node is connected to its k-nearest-neighbors with uniform edge_weights
        """        
        dfc         = self.gdf[[self.id_col, 'geometry']].sort_values(self.id_col).reset_index(drop=True)
        centroids   = dfc.geometry.centroid
        coords      = np.column_stack([centroids.x, centroids.y])
        distances   = euclidean_distances(coords)
        edges       = []
        for i in range(len(dfc)):
            nearest_indices = np.argsort(distances[i])[1:k+1]
            for j in nearest_indices:
                edges.append((int(dfc.iloc[i][self.id_col]), int(dfc.iloc[j][self.id_col])))
        return edges, None

    def _population_weighted(self, max_distance: float) -> Tuple[List[Tuple[int,int]], List[float]]:
        """
        creates a population_weighted - graph:
        each node is connected only to all nodes within max_distance, with weights proportional to population_sizes of these nodes.
        """        
        dfc                     = self.gdf[[self.id_col, 'geometry']].sort_values(self.id_col).reset_index(drop=True)
        dfc                     = dfc.merge(self.popdata[[self.id_col, 'population_size']], on=self.id_col, how='left')
        dfc['population_size']  = dfc['population_size'].fillna(dfc['population_size'].mean())
        centroids               = dfc.geometry.centroid
        coords                  = np.column_stack([centroids.x, centroids.y])
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
