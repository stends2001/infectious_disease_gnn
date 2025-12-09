import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, List, Union, Literal, Tuple
import geopandas as gpd
from matplotlib.figure import Figure
from pandas import Timestamp

from .containers import RawGraphStructure, DynamicRawGraphStructure
from .edgeweight_normalizer import EdgeWeightNormalizer
from .rawgraphconstructor import RawGraphConstructor
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
         
    def generate_graphstructure(self, method, self_connection, scaling_method, graphname, **kwargs):
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
        raise NotImplementedError("Each graph orchestrator class must implement its own _generate_graphstats method.")      
      
    def _generate_graphname(self, method: str, self_connection: str, scaling_method: str, name_addition: str = ""):
        """returns graphname str"""
        graphname = f'{method}_{name_addition}' if name_addition else f'{method}'
        graphname = f'{graphname}_self{self_connection}' if self_connection else f'{graphname}'
        graphname = f'{graphname}_{scaling_method}' if scaling_method else f'{graphname}'
        return graphname

    def _validate_shapedata(self, shapedata: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        shapedata = shapedata.dropna(subset=[self.id_col])
        shapedata[self.id_col] = shapedata[self.id_col].astype(int)
        return shapedata      


class StaticGraphOrchestrator(BaseGraphOrchestrator):
    """
    Orchestrates creation of static graph structures.
    """
    def __init__(self,                  
                 data_orchestrator: DataOrchestrator,
                 id_col:            str = 'node',
                 graph_dir:         str = "data/graphs/"):
        
        super().__init__(data_orchestrator, id_col, graph_dir)
        
        self.population_data = self.epipopdata.groupby(id_col)['population_size'].mean().reset_index()
        self.graph_methods = ['geographic_neighbors', 'identity', 'mesh', 
                             'distance_threshold', 'k_nearest', 'population_weighted', 
                             'gravity_model', 'commuter']        

    def generate_graphstructure(self, 
                                method: str, 
                                self_connection: Literal['mean','0','max'] = 'mean', 
                                scaling_method: Optional[str] = None, 
                                graphname: Optional[str] = None, 
                                **kwargs) -> None:
        
        if method not in self.graph_methods:
            raise ValueError(f'invalid method {method} for StaticGraphOrchestrator. Supported methods: {self.graph_methods}')
        
        if graphname is None:
            graphname = self._generate_graphname(method, self_connection, scaling_method)    

        # Generate raw graph structure
        graph_generator = RawGraphConstructor(shapedata=self.shapes.copy(), 
                                             tokens=self.tokens, 
                                             id_col=self.id_col)
        
        # Methods that don't need population data
        if method in ['geographic_neighbors', 'identity', 'mesh', 'distance_threshold', 'k_nearest']:
            raw_structure = graph_generator.generate_graph(method=method, **kwargs)
        else:
            raw_structure = graph_generator.generate_graph(method=method, 
                                                          population_data=self.population_data, 
                                                          **kwargs)         
  
        # Add self-loops (only when method is neither identity nor mesh)
        if method not in ['identity', 'mesh']:
            raw_structure = SelfLoopAdder(rawgraphstructure=raw_structure, 
                                         node_ids=self.node_ids).add_loops(self_connection)

        # Convert to tensors
        edge_weight = torch.tensor(raw_structure.edge_weight, dtype=torch.float)
        edge_index = torch.tensor(raw_structure.edge_index, dtype=torch.long).t().contiguous()    

        # Edge-weight normalization (removes zero values)
        if scaling_method:
            norm = EdgeWeightNormalizer(edge_indices=edge_index, 
                                       edge_weights=edge_weight, 
                                       num_nodes=self.num_nodes)
            edge_index, edge_weight = norm.normalize(scaling_method)

        # Create final structures
        graphstructure = GraphStructure(edge_index, edge_weight)
        graphstats = self._generate_graphstats(graphstructure)
        graphconfig = StaticGraphConfig(method, self_connection, scaling_method, kwargs)

        self.graph_registry.add_entry(graphname, 
                                     GraphEntry(graphstructure, graphstats, graphconfig, 'static'))

    def _generate_graphstats(self, graph_structure: GraphStructure) -> StaticGraphStats:
        """Returns a summary of the static graph"""
        global_edge_index = graph_structure.edge_index
        global_edge_weight = graph_structure.edge_weight

        # Basic statistics
        num_edges = global_edge_index.shape[1]
        num_nodes = int(global_edge_index.max().item()) + 1
        edge_density = num_edges / (num_nodes * num_nodes)
        edge_weight_np = global_edge_weight.cpu().numpy()
        
        edge_weight_mean = round(float(edge_weight_np.mean()), 4)
        edge_weight_min = round(float(edge_weight_np.min()), 4)
        edge_weight_max = round(float(edge_weight_np.max()), 4)

        # Degree statistics
        edges_out, edges_in = global_edge_index[0], global_edge_index[1] 
        out_degree = torch.bincount(edges_out, minlength=num_nodes)
        in_degree = torch.bincount(edges_in, minlength=num_nodes)
        isolated_mask = (out_degree == 0) & (in_degree == 0)
        num_isolated_nodes = isolated_mask.sum().item()

        out_degree_mean = round(float(out_degree.float().mean().item()), 2)
        out_degree_max = out_degree.max().item()
        out_degree_min = out_degree[out_degree > 0].min().item() if (out_degree > 0).any() else 0

        in_degree_mean = round(float(in_degree.float().mean().item()), 2)
        in_degree_max = in_degree.max().item()
        in_degree_min = in_degree[in_degree > 0].min().item() if (in_degree > 0).any() else 0

        return StaticGraphStats(
            num_edges=num_edges,
            num_nodes=num_nodes,
            edge_density=round(edge_density, 4),
            edge_weight_mean=edge_weight_mean,
            edge_weight_min=edge_weight_min,
            edge_weight_max=edge_weight_max,
            num_isolated_nodes=num_isolated_nodes,
            out_degree_mean=out_degree_mean,
            out_degree_max=out_degree_max,
            out_degree_min=out_degree_min,
            in_degree_mean=in_degree_mean,
            in_degree_max=in_degree_max,
            in_degree_min=in_degree_min
        )
                
    def __repr__(self) -> str:
        return f'<StaticGraphOrchestrator(level {self.nuts_level}. Registry: {self.graph_registry})>'


class DynamicGraphOrchestrator(BaseGraphOrchestrator):
    """
    Orchestrates creation of dynamic (time-varying) graph structures.
    """
    def __init__(self,                  
                 data_orchestrator: DataOrchestrator,
                 id_col:            str = 'node',
                 graph_dir:         str = "data/graphs/"):
        
        super().__init__(data_orchestrator, id_col, graph_dir)    
        
        # Prepare yearly population data
        population_data = self.epipopdata[['timestamp', id_col, 'population_size']].copy()
        population_data['timestamp'] = population_data['timestamp'].dt.year
        self.population_data = population_data[['timestamp', id_col, 'population_size']].drop_duplicates().reset_index(drop=True)
        
        self.graph_methods = ['population_weighted', 'gravity_model', 'commuter']

    def generate_graphstructure(self, 
                                method: str, 
                                time_window: List[Union[Timestamp, str]], 
                                frequency: str = 'yearly', 
                                self_connection: Literal['mean','0','max'] = 'mean', 
                                scaling_method: Optional[str] = None, 
                                graphname: Optional[str] = None, 
                                **kwargs) -> None:
        
        if method not in self.graph_methods:
            raise ValueError(f'invalid method {method}. Supported methods: {self.graph_methods}')
        
        if graphname is None:
            graphname = self._generate_graphname(method, self_connection, scaling_method)

        # Prepare time axis and population data
        popdata = self._resample_population_data(frequency)
        freq = self._set_frequency(frequency)
        time_axis = self._return_list_timestamps(time_window, frequency, freq)

        # Generate raw graph structures for each timestamp
        graph_generator = RawGraphConstructor(shapedata=self.shapes.copy(), 
                                             tokens=self.tokens, 
                                             id_col=self.id_col)
        
        raw_structures = []
        for tt in time_axis:
            population_data_tt = popdata[popdata['timestamp'] == tt].reset_index()[[self.id_col, 'population_size']]
            raw_structure = graph_generator.generate_graph(method=method, 
                                                          population_data=population_data_tt, 
                                                          **kwargs)
            
            # Add self-loops if needed
            if method not in ['identity', 'mesh']:
                raw_structure = SelfLoopAdder(rawgraphstructure=raw_structure, 
                                             node_ids=self.node_ids).add_loops(self_connection)
            
            raw_structures.append(raw_structure)

        # Convert time_axis to string format for storage
        time_strings = [str(t) for t in time_axis]
        dynamic_raw = DynamicRawGraphStructure(time_strings, raw_structures)

        # Process each snapshot into tensors
        edge_indices_list = []
        edge_weights_list = []
        
        for raw_struct in dynamic_raw.rawstructures:
            edge_weight = torch.tensor(raw_struct.edge_weight, dtype=torch.float)
            edge_index = torch.tensor(raw_struct.edge_index, dtype=torch.long).t().contiguous()
            
            # Apply normalization if specified
            if scaling_method:
                norm = EdgeWeightNormalizer(edge_indices=edge_index, 
                                           edge_weights=edge_weight, 
                                           num_nodes=self.num_nodes)
                edge_index, edge_weight = norm.normalize(scaling_method)
            
            edge_indices_list.append(edge_index)
            edge_weights_list.append(edge_weight)

        # Convert timestamps to Unix timestamps for tensor storage
        timestamps_pd = pd.to_datetime(time_strings)
        timestamps_unix = torch.tensor([int(ts.timestamp()) for ts in timestamps_pd], dtype=torch.int64)

        # Create final structures
        graphstructure = DynamicGraphStructure(timestamps_unix, edge_indices_list, edge_weights_list)
        graphstats = self._generate_graphstats(graphstructure, time_strings)
        graphconfig = DynamicGraphConfig(method, time_strings, frequency, 
                                        self_connection, scaling_method, kwargs)

        self.graph_registry.add_entry(graphname, 
                                     GraphEntry(graphstructure, graphstats, graphconfig, 'dynamic'))

    def _generate_graphstats(self, graph_structure: DynamicGraphStructure, 
                            time_strings: List[str]) -> DynamicGraphStats:
        """Generate statistics for dynamic graph"""
        num_edges_list = []
        num_nodes_list = []
        edge_density_list = []
        edge_weight_mean_list = []
        edge_weight_min_list = []
        edge_weight_max_list = []
        
        for t_idx in range(len(graph_structure)):
            snapshot = graph_structure.get_snapshot(t_idx)
            edge_index = snapshot.edge_index
            edge_weight = snapshot.edge_weight
            
            num_edges = edge_index.shape[1]
            num_nodes = int(edge_index.max().item()) + 1
            edge_density = num_edges / (num_nodes * num_nodes)
            
            edge_weight_np = edge_weight.cpu().numpy()
            
            num_edges_list.append(num_edges)
            num_nodes_list.append(num_nodes)
            edge_density_list.append(round(edge_density, 4))
            edge_weight_mean_list.append(round(float(edge_weight_np.mean()), 4))
            edge_weight_min_list.append(round(float(edge_weight_np.min()), 4))
            edge_weight_max_list.append(round(float(edge_weight_np.max()), 4))
        
        return DynamicGraphStats(
            timestamps=time_strings,
            num_edges=num_edges_list,
            num_nodes=num_nodes_list,
            edge_density=edge_density_list,
            edge_weight_mean=edge_weight_mean_list,
            edge_weight_min=edge_weight_min_list,
            edge_weight_max=edge_weight_max_list
        )

    def _resample_population_data(self, frequency: str):
        """Resample population data to match frequency"""
        if frequency != 'yearly':
            raise ValueError('currently only yearly population data supported')
        return self.population_data

    def _return_list_timestamps(self, time_window: List[Union[Timestamp, str]], 
                               frequency: str, freq: str) -> List[Union[int, Timestamp]]:
        """Generate list of timestamps for the time window"""
        mindate = time_window[0]
        maxdate = time_window[1]

        if isinstance(mindate, str):
            mindate = pd.to_datetime(mindate)
            maxdate = pd.to_datetime(maxdate)

        time_axis = list(pd.date_range(start=mindate, end=maxdate, freq=freq))

        if frequency == 'yearly':
            time_axis = [t.year for t in time_axis]

        return time_axis

    def _set_frequency(self, frequency: str) -> str:
        """Convert frequency string to pandas offset alias"""
        if frequency == 'yearly':
            return 'YS'
        else:
            raise ValueError(f'currently only yearly frequency supported')

    def __repr__(self) -> str:
        return f'<DynamicGraphOrchestrator(level {self.nuts_level}. Registry: {self.graph_registry})>'