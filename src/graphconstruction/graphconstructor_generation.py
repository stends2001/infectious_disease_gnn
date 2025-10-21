import os

from typing import Optional, Tuple, List, Union, Literal, Dict

import pandas as pd
import geopandas as gpd
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

from ..utils import get_data_env

from .pendlerdatenprocessor import PendlerDatenProcessor

class GraphGeneration:

    """ 
    creates graphs (edge indices and edge weights) based on geopandas dataframes
    Called through GraphConstructor

    Parameters:
    ----------
    gdf: gpd.GeoDataFrame
        shapefile including all nodes
    tokens: dict[int, int]
        dictionary comprising the kz and a token-integer
    popdata: pd.DataFrame
        population data per token
    id_col: column by which name the nodes are found

    See also:
    --------
    GraphConstructor
    """

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

        self.GENERATION_FUNCS ={
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
        """
        collects the required generation-function and feeds in the kwargs
        """

        if method not in self.GENERATION_FUNCS:
            available = ', '.join(self.GENERATION_FUNCS.keys())
            raise ValueError(
                f"Unknown graph generation method: '{method}'. "
                f"Available methods: {available}"
            )
        
        edge_indices, edge_weights =  self.GENERATION_FUNCS[method](**kwargs) 
        return (edge_indices, edge_weights)

    def _commuter(self, commuter_type: Literal['static','dynamic'], commuting_threshold: int = 1_000, top_k: Optional[int] = None) -> Tuple[List[Tuple[int,int]], List[float]]:
        """ 
        creates a commuter - graph:

        Parameters:
        ----------
        commuter_type: Literal['static','dynamic']
            whether to have dynamic graphs (one per year) or a static one. 
            TODO: currently no dynamic commuter graphs implemented
        commuting_threshold: int = 1_000
            the threshold for number of commuters for nodes to be connected
        top_k
        """
        if commuter_type == 'static':
            commuter_data_object = PendlerDatenProcessor(raw_folder_path = os.path.join(get_data_env(),'raw/germany/mobility/commuter_data/auspendler/'), processed_folder_path= os.path.join(get_data_env(),'processed/germany/mobility/commuter_data/')).import_raw_data('2024')
            commuter_data        = commuter_data_object.data['2024']
            commuter_data.loc[:, 'nuts3_work'] = commuter_data.loc[:, 'nuts3_work'].map(self.tokens)
            commuter_data.loc[:, 'nuts3_residence'] = commuter_data.loc[:, 'nuts3_residence'].map(self.tokens)

        else:
            raise ValueError('dynamic commuter graphs not yet implemented')

        commuter_data   = commuter_data[commuter_data['commuters'] > commuting_threshold]

        if top_k is not None:
            # Keep only top_k strongest links per source (nuts3_work)
            commuter_data = (
                commuter_data
                .sort_values(by='commuters', ascending=False)
                .groupby('nuts3_work', group_keys=False)
                .head(top_k)
            )

        node_edges = list(zip(
            commuter_data['nuts3_work'].astype(int),
            commuter_data['nuts3_residence'].astype(int)
        ))
        node_weights = commuter_data['commuters'].astype(float).tolist()

        return (node_edges, node_weights)
                
    def _boolean_neighbors(self) -> Tuple[List[Tuple[int,int]], None]:
        """ 
        creates a boolean neighbors - graph:
        each node is connected to directly surrounding nodes with uniform edge_weights
        """
        dfc         = self.gdf[[self.id_col,'geometry']].copy()
        dfc         = dfc.sort_values(self.id_col).reset_index(drop=True)
        neighbors   = gpd.sjoin(dfc, dfc, how='inner', predicate='touches').reset_index(drop=False)
        neighbors   = neighbors[neighbors[f'{self.id_col}_left'] != neighbors[f'{self.id_col}_right']]
        edges       = list(zip(neighbors[f'{self.id_col}_left'], neighbors[f'{self.id_col}_right']))
        edges      += [(t, s) for s, t in edges]
        edges       = list(set(edges))
        return edges, None

    def _identity(self) -> Tuple[List[Tuple[int,int]], None]:
        """
        creates an identity - graph:
        each node is connected only to itself with uniform edge_weights
        """
        dfc         = self.gdf[[self.id_col]].sort_values(self.id_col).reset_index(drop=True)
        node_ids    = dfc[self.id_col].dropna().astype(int).values
        edges       = [(int(nid), int(nid)) for nid in node_ids]
        return edges, None

    def _mesh(self) -> Tuple[List[Tuple[int,int]], None]:
        """
        creates a mesh - graph:
        each node is connected to every other node with uniform edge_weights
        """        
        dfc         = self.gdf[[self.id_col]].sort_values(self.id_col).reset_index(drop=True)
        node_ids    = dfc[self.id_col].dropna().astype(int).values
        edges       = [(int(s), int(t)) for s in node_ids for t in node_ids]
        return edges, None
            
    def _distance_threshold(self, max_distance: float) -> Tuple[List[Tuple[int,int]], None]:
        """
        creates a distance_threshold - graph:
        each node is connected to all other nodes within max_distance with uniform edge_weights
        """        
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
        """
        creates a k-nearest-neighbors - graph:
        each node is connected to its k-nearest-neighbors with uniform edge_weights
        """        
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
        """
        creates a population_weighted - graph:
        each node is connected only to all nodes within max_distance, with weights proportional to population_sizes of these nodes.
        """        
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

    def _gravity_model(self,
        alpha:  float = 2.0,
        epsilon:float = 1e-6,
        decay:  float = 1.0,
        max_distance: float = np.inf,
        top_k:  Optional[int] = None) -> Tuple[List[Tuple[int,int]], List[float]]:
        """ 
        classic gravity model following:
            weight = (pop_i * pop_j) / ((distance * distance_decay_factor)^alpha + epsilon)
        """

        dfc = self.gdf[[self.id_col, 'geometry']].sort_values(self.id_col)

        dfc_centroids = dfc 
        dfc_centroids['geometry'] = dfc.geometry.centroid


        coords = np.array([[point.x, point.y] for point in dfc_centroids.geometry])

        # Compute Euclidean distances between all pairs
        distance_matrix = euclidean_distances(coords)

        edges = []
        weights = []

        num_nodes = len(self.popdata)

        popvalues = self.popdata['population_size'].values
        node_ids  = self.popdata['node'].values

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

    # def _gravity_model(self, 
    #                 max_distance: float, 
    #                 alpha: float = 2.0, 
    #                 density_control: Literal['top_k', 'threshold', 'adaptive', 'distance_bands'] = 'top_k', 
    #                 k: int = 10, 
    #                 weight_threshold: Optional[float] = None, 
    #                 distance_decay_factor: float = 1.0) -> Tuple[List[Tuple[int,int]], List[float]]:
    #     """
    #     Creates a gravity model graph where edge weights are proportional to node populations
    #     and inversely proportional to distance, similar to gravitational force between masses.
        
    #     The gravity model weight formula:
    #         weight = (pop_i * pop_j) / ((distance * distance_decay_factor)^alpha + epsilon)
        
    #     Parameters:
    #     ----------
    #     max_distance : float
    #         Maximum distance threshold in meters (or coordinate units). Nodes farther apart 
    #         than this distance will not be connected. Acts as a hard cutoff for potential edges.
            
    #     alpha : float, default=2.0
    #         Distance decay exponent. Higher values make distance matter more (steeper decay).
    #         - alpha=1.0: linear decay
    #         - alpha=2.0: quadratic decay (similar to physical gravity)
    #         - alpha>2.0: super-quadratic decay (very local connections)
            
    #     density_control : {'top_k', 'threshold', 'adaptive', 'distance_bands'}, default='top_k'
    #         Method for controlling graph density and preventing over-connection:
            
    #         - 'top_k': Keep only the k strongest connections per node (recommended for balanced graphs)
    #         - 'threshold': Keep connections above a fixed weight threshold
    #         - 'adaptive': Keep connections above 10% of each node's maximum weight (node-specific)
    #         - 'distance_bands': Stratified selection prioritizing closer connections
    #         (6 close + 3 medium + 1 far = 10 total max per node)
            
    #     k : int, default=10
    #         Number of top connections to keep per node when density_control='top_k'.
    #         Larger k creates denser graphs with more computational cost.
            
    #     weight_threshold : float or None, default=None
    #         Absolute weight threshold for density_control='threshold'. If None, 
    #         automatically set to the 75th percentile of computed weights.
            
    #     distance_decay_factor : float, default=1.0
    #         Multiplicative factor applied to distance before computing decay. 
    #         Values > 1.0 make distances "feel" larger (stronger locality).
    #         Values < 1.0 make distances "feel" smaller (weaker locality).
    #         Example: distance_decay_factor=2.0 makes 100km behave like 200km.
        
    #     Returns:
    #     -------
    #     edges : List[Tuple[int, int]]
    #         List of directed edges as (source_node_id, target_node_id) tuples
            
    #     weights : List[float]
    #         Corresponding edge weights based on gravity model calculation
            
    #     Notes:
    #     -----
    #     - Self-loops (i==j) are automatically excluded
    #     - Missing population data is imputed with mean population
    #     - Small epsilon (1e-6) added to denominator to prevent division by zero
    #     - All edges within max_distance are computed first, then filtered by density_control
        
    #     Examples:
    #     --------
    #     >>> # Standard gravity model with top 10 connections per node
    #     >>> edges, weights = self._gravity_model(max_distance=100000, alpha=2.0, 
    #     ...                                       density_control='top_k', k=10)
        
    #     >>> # Very local model with adaptive thresholding
    #     >>> edges, weights = self._gravity_model(max_distance=50000, alpha=3.0,
    #     ...                                       density_control='adaptive')
        
    #     >>> # Distance-stratified connections emphasizing closer neighbors  
    #     >>> edges, weights = self._gravity_model(max_distance=150000, 
    #     ...                                       density_control='distance_bands')
    #     """
    #     dfc                     = self.gdf[[self.id_col, 'geometry']].sort_values(self.id_col).reset_index(drop=True)
    #     dfc                     = dfc.merge(self.popdata[[self.id_col, 'population_size']], on=self.id_col, how='left')
    #     dfc['population_size']  = dfc['population_size'].fillna(dfc['population_size'].mean())
        
    #     centroids               = dfc.geometry.centroid
    #     coords                  = np.array([[point.x, point.y] for point in centroids])
    #     distances               = euclidean_distances(coords)

    #     # Calculate all potential weights first
    #     all_weights = []
    #     all_edges = []
        
    #     for i in range(len(dfc)):
    #         node_weights = []
    #         node_edges = []
            
    #         for j in range(len(dfc)):
    #             if distances[i, j] <= max_distance and i != j:
    #                 pop_i = dfc.iloc[i]['population_size']
    #                 pop_j = dfc.iloc[j]['population_size']
    #                 dist = distances[i, j]
                    
    #                 # Enhanced gravity model with additional distance decay
    #                 weight = (pop_i * pop_j) / ((dist * distance_decay_factor)**alpha + 1e-6)
                    
    #                 node_weights.append((weight, j))
    #                 node_edges.append((int(dfc.iloc[i][self.id_col]), int(dfc.iloc[j][self.id_col])))
            
    #         # Apply density control
    #         if density_control == 'top_k':
    #             # Keep only top k connections per node
    #             node_weights.sort(reverse=True)
    #             selected = node_weights[:k]
                
    #         elif density_control == 'threshold':
    #             # Keep connections above weight threshold
    #             if weight_threshold is None:
    #                 weight_threshold = np.percentile([w[0] for w in node_weights], 75)
    #             selected = [(w, j) for w, j in node_weights if w >= weight_threshold]
                
    #         elif density_control == 'adaptive':
    #             # Adaptive threshold based on node's maximum weight
    #             if node_weights:
    #                 max_weight = max(w[0] for w in node_weights)
    #                 adaptive_threshold = max_weight * 0.1  # Keep top 10% of weights
    #                 selected = [(w, j) for w, j in node_weights if w >= adaptive_threshold]
    #             else:
    #                 selected = []
                    
    #         elif density_control == 'distance_bands':
    #             # Prioritize closer connections, limit distant ones
    #             node_weights_with_dist = [(w, j, distances[i, j]) for w, j in node_weights]
    #             node_weights_with_dist.sort(key=lambda x: x[2])  # Sort by distance
                
    #             # Take more from closer distance bands
    #             close_band  = [x for x in node_weights_with_dist if x[2] <= max_distance * 0.3]
    #             medium_band = [x for x in node_weights_with_dist if max_distance * 0.3 < x[2] <= max_distance * 0.7]
    #             far_band    = [x for x in node_weights_with_dist if x[2] > max_distance * 0.7]
                
    #             selected_items = close_band[:6] + medium_band[:3] + far_band[:1]  # 6+3+1=10 connections max
    #             selected = [(w, j) for w, j, d in selected_items]
            
    #         else:
    #             selected = node_weights  # No density control
            
    #         # Add selected edges and weights
    #         for idx, (weight, j) in enumerate(selected):
    #             all_edges.append(node_edges[node_weights.index((weight, j))])
    #             all_weights.append(weight)
            
    #     return all_edges, all_weights