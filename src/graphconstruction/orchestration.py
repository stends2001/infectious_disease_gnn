import os
import torch
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, List, Union, Literal, Tuple

import geopandas as gpd
from matplotlib.figure import Figure

from .edgeweight_normalizer import EdgeWeightNormalizer
from .graphconstructor import GraphConstructor
from .selfloopadder import SelfLoopAdder
from .graphviewer import GraphViewer

from .graphregistry import GraphEntry, GraphRegistry
from .graphstructures import GraphStructure, DynamicGraphStructure
from .graphstats import StaticGraphStats, DynamicGraphStats
from .graphconfig import StaticGraphConfig, DynamicGraphConfig

from ..dataloading import DataOrchestrator


class BaseGraphOrchestrator:
    """
    Parent class for StaticGraphOrchestrator and DynamicGraphOrchestrator. 

    Both inherit from this class, but interact differently with 
    the construction classes and the registry.
    
    
    """
    def __init__(self,                  
                 data_orchestrator: DataOrchestrator,
                 id_col:          str = 'node',
                 graph_dir:       str = "data/graphs/"):
        
        # extract metadata
        self.context_data    = data_orchestrator.data_context
        self.epipopdata      = data_orchestrator.data_harmonized.data

        self.id_col          = id_col 
        self.tokens          = self.context_data.tokenization_map['id_idx']
        self.shapes          = self._validate_shapedata(self.context_data.shapedata)
        
        self.nuts_level      = self.context_data.nuts_level
        self.graph_dir       = os.path.join(graph_dir, f'{self.nuts_level}')
        self.num_nodes       = self.context_data.num_nodes
        self.node_ids        = np.array(self.shapes[self.id_col].values) 

        # registry of graphs
        os.makedirs(self.graph_dir, exist_ok=True)
        self.graph_registry = GraphRegistry(self.graph_dir)
        self.previewer      = GraphViewer(self.graph_registry, self.shapes)
         
    def generate_graphstructure(self, 
                                method,
                                self_connection,
                                scaling_method,
                                graphname,
                                **kwargs
                                ):
        raise NotImplementedError("Each graph orchestrator class must implement its own generate_graphstructure method.")     
    
    def preview_graphstructure(self,
                               graphname: Optional[str] = None,
                               node_idx: Optional[int]  = None,
                               subplots: bool           = True,
                               title:    Optional[str]  = None) -> Figure:
        if not graphname:
            graphname = 'empty'

        return self.previewer.view(graphname, node_idx, subplots, title)

    def _generate_graphstats(self):
        """
        ...
        """
        raise NotImplementedError("Each graph orchestrator class must implement its own _generate_graphstats method.")      
      
    def _generate_graphname(self, method: str, name_addition: str, self_connection: str, scaling_method: str):
        """returns graphname str"""
        graphname = f'{method}_{name_addition}'         if name_addition    else f'{method}'
        graphname = f'{graphname}_self{self_connection}'if self_connection  else f'{graphname}'
        graphname = f'{graphname}_{scaling_method}'     if scaling_method   else f'{graphname}'
        return graphname

    def _validate_shapedata(self, shapedata: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        shapedata               = shapedata.dropna(subset=[self.id_col])
        shapedata[self.id_col]  = shapedata[self.id_col].astype(int)
        return shapedata      

class StaticGraphOrchestrator(BaseGraphOrchestrator):

    def __init__(self,                  
                 data_orchestrator: DataOrchestrator,
                 id_col:            str = 'node',
                 graph_dir:         str = "data/graphs/"):
        
        super().__init__(data_orchestrator, id_col, graph_dir)
        
        self.population_data    = self.epipopdata.groupby(id_col)['population_size'].mean().reset_index()
        self.graph_methods      = ['boolean_neighbors', 
                                   'identity', 
                                   'mesh', 
                                   'distance_threshold',
                                   'k_nearest', 
                                   'population_weighted', 
                                   'gravity_model', 
                                   'commuter']        

    def generate_graphstructure(self, method, self_connection: Literal['mean','0','max'] = 'mean', scaling_method: Optional[str] = None, graphname: Optional[str] = None, **kwargs) -> None:
        
        if method not in self.graph_methods:
            raise ValueError(f'invalid method {method} for StaticGraphOrchestrator. Supported methods include: {self.graph_methods}')
        
        if graphname is None:
            graphname = self._generate_graphname(method, self_connection, scaling_method, graphname)    

        # generate graph:   
        graph_generator = GraphConstructor(gdf = self.shapes.copy(), tokens = self.tokens, popdata = self.population_data, id_col = self.id_col)
        edges, weights  = graph_generator.generate_graph(method=method, **kwargs)         

        # if weights is none => each edge gets equal weight
        if weights is None:                                                                                                             # if weights is undefined, give 1 everywhere
            weights = [float(1) for _ in edges]      

        # self-loops => only when method is neither identity nor mesh
        if method not in ['identity', 'mesh']:
            edges, weights = SelfLoopAdder(edge_indices=edges, edge_weights=weights, node_ids=self.node_ids).add_loops(self_connection)                                                                # remove zero valued loops        

        edge_weight     = torch.tensor(weights, dtype=torch.float)
        edge_index      = torch.tensor(edges,   dtype=torch.long).t().contiguous()    

        # edge-weight normalization. Zero values (pre-normalization) are removed
        if scaling_method:
            norm = EdgeWeightNormalizer(edge_indices=edge_index, edge_weights=edge_weight, num_nodes=self.num_nodes)
            edge_index, edge_weight = norm.normalize(scaling_method)

        graphstructure = GraphStructure(edge_index, edge_weight)
        graphstats     = self._generate_graphstats(graphstructure)
        graphconfig    = StaticGraphConfig(method, self_connection, scaling_method, kwargs)

        self.graph_registry.add_entry(graphname, GraphEntry(graphstructure, graphstats, graphconfig, 'static'))

    def _generate_graphstats(self, graph_structure: GraphStructure)  -> StaticGraphStats:
        """ 
        Returns a summary of the static graph
        """
        global_edge_index   = graph_structure.edge_index
        global_edge_weight  = graph_structure.edge_weight

        # statistics
        num_edges           = global_edge_index.shape[1]
        num_nodes           = int(global_edge_index.max().item()) + 1
        edge_density        = num_edges / (num_nodes * (num_nodes))
        edge_weight_np      = global_edge_weight.cpu().numpy()
        
        # Round edge weight statistics at creation time
        edge_weight_mean    = round(float(edge_weight_np.mean()), 4)
        edge_weight_min     = round(float(edge_weight_np.min()), 4)
        edge_weight_max     = round(float(edge_weight_np.max()), 4)

        # isolated nodes:
        edges_out, edges_in = global_edge_index[0], global_edge_index[1] 
        out_degree          = torch.bincount(edges_out, minlength=num_nodes)
        in_degree           = torch.bincount(edges_in, minlength=num_nodes)
        isolated_mask       = (out_degree == 0) & (in_degree == 0)
        num_isolated_nodes  = isolated_mask.sum().item()

        # Out-degree stats
        out_degree_mean     = round(float(out_degree.float().mean().item()), 2)
        out_degree_max      = out_degree.max().item()
        out_degree_min      = out_degree[out_degree > 0].min().item() if (out_degree > 0).any() else 0

        # In-degree stats
        in_degree_mean      = round(float(in_degree.float().mean().item()), 2)
        in_degree_max       = in_degree.max().item()
        in_degree_min       = in_degree[in_degree > 0].min().item() if (in_degree > 0).any() else 0

        return StaticGraphStats(
            num_edges           = num_edges,
            num_nodes           = num_nodes,
            edge_density        = round(edge_density, 4),
            edge_weight_mean    = edge_weight_mean,
            edge_weight_min     = edge_weight_min,
            edge_weight_max     = edge_weight_max,
            num_isolated_nodes  = num_isolated_nodes,
            out_degree_mean     = out_degree_mean,
            out_degree_max      = out_degree_max,
            out_degree_min      = out_degree_min,
            in_degree_mean      = in_degree_mean,
            in_degree_max       = in_degree_max,
            in_degree_min       = in_degree_min
        )
                
    def __repr__(self) -> str:
        return f'<StaticGraphConstructor(level {self.nuts_level}. Registry: {self.graph_registry})>'

class DynamicGraphOrchestrator(BaseGraphOrchestrator):


    def __init__(self,                  
                 data_orchestrator: DataOrchestrator,
                 id_col:            str = 'node',
                 graph_dir:         str = "data/graphs/"):
        
        super().__init__(data_orchestrator, id_col, graph_dir)    

        self.population_data= self.epipopdata[['timestamp',id_col,'population_size']]
        self.graph_methods  = ['population_weighted', 'gravity_model', 'commuter']

    def generate_graphstructure(self, method, self_connection, scaling_method, graphname: Optional[str] = None, **kwargs):
        
        if method not in self.graph_methods:
            raise ValueError(f'invalid method {method} for StaticGraphOrchestrator. Supported methods include: {self.graph_methods}')
        
        if graphname is None:
            graphname = self._generate_graphname(method, self_connection, scaling_method, graphname)

    def __repr__(self) -> str:
        return f'<DynamicGraphOrchestrator(level {self.nuts_level}. Registry: {self.graph_registry})>'       