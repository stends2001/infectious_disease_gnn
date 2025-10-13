import os

from typing import Optional, Tuple, List, Union, Literal, Dict

import pandas as pd
import geopandas as gpd
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

from ..utils import get_data_env

from .pendlerdatenprocessor import PendlerDatenProcessor

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