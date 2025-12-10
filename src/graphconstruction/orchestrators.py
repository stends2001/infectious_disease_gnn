import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, List, Union, Literal, Tuple
import geopandas as gpd
from matplotlib.figure import Figure
from pandas import Timestamp
from tqdm import tqdm


from ..utils.textformatting import warning_emoji, error_emoji

from .containers import RawGraphStructure, DynamicGraphStructure, GraphStructure

from .edgeweight_normalizer import EdgeWeightNormalizer
from .rawgraphconstructor import RawGraphConstructor
from .selfloopadder import SelfLoopAdder
from .staticgraphviewer import StaticGraphViewer
from .graphregistry import GraphEntry, GraphRegistry
# from .graphstructures import GraphStructure, DynamicGraphStructure
from .graphstats import StaticGraphStats, DynamicGraphStats
from .graphconfig import StaticGraphConfig, DynamicGraphConfig
from ..dataloading import DataOrchestrator

from .commuterdataloader import CommuterDataLoader

class BaseGraphOrchestrator:
    """
    Parent class for StaticGraphOrchestrator and DynamicGraphOrchestrator
    while there is some shared behaviour among the two, there's large fundamental
    differences in generating a static or a dynamic graph structure

    Methods
    -------
    - _generate_graphname
    - _validate_shapedata
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
         
    def generate_graphstructure(self):
        raise NotImplementedError("Each graph orchestrator class must implement its own generate_graphstructure method.")     
    
    def preview_graphstructure(self):
        raise NotImplementedError("Each graph orchestrator class must implement its own preview_graphstructure method.")    

    def _generate_graphstats(self):
        raise NotImplementedError("Each graph orchestrator class must implement its own _generate_graphstats method.")      
      
    def _generate_graphname(self, method: str, self_connection: str, scaling_method: str):
        """returns graphname str"""
        graphname = f'{method}'
        graphname = f'{graphname}_self{self_connection}' if self_connection else f'{graphname}'
        graphname = f'{graphname}_{scaling_method}' if scaling_method else f'{graphname}'
        return graphname

    def _validate_shapedata(self, shapedata: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        shapedata = shapedata.dropna(subset=[self.id_col])
        shapedata[self.id_col] = shapedata[self.id_col].astype(int)
        return shapedata      

class StaticGraphOrchestrator(BaseGraphOrchestrator):
    """
    Orchestrates creation of static graph structures

    Methods
    -------
    - generate_graphstructure
    - preview_graphstructure
    - _generate_graphstats
    """
    def __init__(self,                  
                 data_orchestrator: DataOrchestrator,
                 id_col:            str = 'node',
                 graph_dir:         str = "data/graphs/"):
        
        super().__init__(data_orchestrator, id_col, graph_dir)
               
        self.previewer       = StaticGraphViewer(self.graph_registry, self.shapes) 
        self.population_data = self.epipopdata.groupby(id_col)['population_size'].mean().reset_index()
        self.graph_methods   = ['geographic_neighbors', 'identity',  'mesh', 
                                'distance_threshold',   'k_nearest', 'population_weighted', 
                                'gravity_model',        'commuter']        

    def generate_graphstructure(self, 
                                method:             str, 
                                graphname:          Optional[str] = None,                                
                                self_connection:    Literal['mean','0','max','eps'] = '0', 
                                normalization:      Optional[Literal['minmax','log','zscore','symmetric','rowwise']] = None, 
                                **kwargs) -> None:
        """
        Generates a single (static) graphstructure which is added to the graph registry

        Parameters
        ----------
        method: str
            static graphstructure- method. any of:
            - identity
            - mesh
            - geographic_neighbors
            - distance_threshold
            - k_nearest
            - population_weighted
            - gravity_model
            - commuter
             
        graphname: Optional[str]
            the name under which the graph entry will be saved. Recommended to supply. 
            If None, an artificial name will be generated according to `_generate_graphname`

        self_connection: Literal['mean','0','max'] = '0'
            how to create self-loops. By default '0' => None

        normalization: Optional[Literal['minmax','log','zscore','symmetric','rowwise']] = None
            how to normalize edge-weights. By default non-normalized

        Downstream
        ----------
        when called, this method calls upon:
        - RawGraphConstructor
            creates RawGraphStructure depending on method called and **kwargs. 
            this is a collection of a list of edge-index and a list of edge-weights
        - SelfLoopAdder
            adds self-loops depending on method called in, according to parameter self_connection
        - EdgeWeightNormalizer
            normalizes edge-weights according to parameter normalization
        
        to complete a GraphEntry which is to be added to graphregistry, further called are:
        - self._generate_graphstats() 
            returns a dictionary of statistics for the graph
        - StaticGraphConfig
            returns a dictionary of the configuration for the graph
            
        then the graphstructure is added to the registry:
            self.graph_registry.add_entry()
        """
        if method not in self.graph_methods:
            raise ValueError(f'invalid method {method} for StaticGraphOrchestrator. Supported methods: {self.graph_methods}')
        
        if graphname is None:
            graphname = self._generate_graphname(method, self_connection, normalization)    

        # Generate raw graph structure
        graph_generator = RawGraphConstructor(shapedata=self.shapes.copy(), 
                                              tokens=self.tokens, 
                                              id_col=self.id_col)
        
        # Methods that need population data
        if method in ['population_weighted','gravity_model']:
            raw_structure = graph_generator.generate_graph(method=method, 
                                                          population_data=self.population_data, 
                                                          **kwargs)           
        elif method == 'commuter':

            if 'year' not in kwargs:
                raise ValueError(f'when using commuter data please supply a year')
            
            else:
                commuter_data = CommuterDataLoader(years=kwargs['year']).return_data()    
                raw_structure = graph_generator.generate_graph(method=method, 
                                                          commuter_data=commuter_data, 
                                                          **kwargs)      
        else:
            raw_structure = graph_generator.generate_graph(method=method, **kwargs)

        # Add self-loops (only when method is neither identity nor mesh)
        if method not in ['identity', 'mesh']:
            raw_structure = SelfLoopAdder(rawgraphstructure=raw_structure, 
                                         node_ids=self.node_ids).add_loops(self_connection)

        # Convert to tensors
        edge_weight                 = torch.tensor(raw_structure.edge_weight_ls, dtype=torch.float)
        edge_index                  = torch.tensor(raw_structure.edge_index_ls,   dtype=torch.long).t().contiguous() 

        graphstructre_prenormalized = GraphStructure(edge_index, edge_weight)

        # Edge-weight normalization (removes zero values)
        edgeweight_normalizer       = EdgeWeightNormalizer(graphstructre_prenormalized)
        graphstructure_normalized   = edgeweight_normalizer.normalize(normalization)

        # Create final structures
        graphstats = self._generate_graphstats(graphstructure_normalized)
        graphconfig = StaticGraphConfig(method, self_connection, normalization, kwargs)

        self.graph_registry.add_entry(graphname, 
                                     GraphEntry(graphstructure_normalized, graphstats, graphconfig, 'static'))

    def preview_graphstructure(self,
                               graphname: Optional[str] = None,
                               node_idx: Optional[int]  = None,
                               subplots: bool           = True,
                               title:    Optional[str]  = None) -> Figure:
        """
        preview a static graphstructure in self.graph_registry
        
        Parameters
        ----------
        graphname: Optional[str] = None
            name under which graph is saved in self.graph_registry
            when none, the emtpy graph structure is viewed
        node_idx: Optional[int] = None
            a node to zoom into (neighborhood will be shown)
            when none, the global graph structure is viewed
        subplots: bool = True
            whether or not to show some more information in side-panels
            when False, only the main map is shown
        title: Optional[str] = None
            extra title to be shown

        Returns
        -------
        Figure

        Downstream
        ----------
        heavy lifting is done by StaticGraphViewer
        """
        if not graphname:
            graphname = 'empty'
        return self.previewer.view(graphname, node_idx, subplots, title)

    def _generate_graphstats(self, graph_structure: GraphStructure) -> StaticGraphStats:
        """Returns a summary of the static graph"""
        global_edge_index = graph_structure.edge_index
        global_edge_weight = graph_structure.edge_weight

        # Basic statistics
        num_edges = global_edge_index.shape[1]
        num_nodes = torch.unique(global_edge_index).numel()
        edge_density = num_edges / (num_nodes * num_nodes)
        edge_weight_np = global_edge_weight.cpu().numpy()
        
        edge_weight_mean = round(float(edge_weight_np.mean()), 4)
        edge_weight_min = round(float(edge_weight_np.min()), 4)
        edge_weight_max = round(float(edge_weight_np.max()), 4)

        # Degree statistics
        edges_out, edges_in = global_edge_index[0], global_edge_index[1] 
        out_degree = torch.bincount(edges_out, minlength=num_nodes)
        in_degree  = torch.bincount(edges_in, minlength=num_nodes)
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
    Orchestrates creation of dynamic (time-varying) graph structures

    Methods
    -------
    - generate_graphstructure
    - preview_graphstructure
    - _generate_graphstats
    - _resample_population_data
    - _return_list_timestamps
    - _set_frequency
    """
    def __init__(self,                  
                 data_orchestrator: DataOrchestrator,
                 id_col:            str = 'node',
                 graph_dir:         str = "data/graphs/"):
        
        super().__init__(data_orchestrator, id_col, graph_dir)    
        
        # Prepare yearly population data
        
        population_data              = self.epipopdata[['timestamp', id_col, 'population_size']].copy()
        population_data['timestamp'] = population_data['timestamp'].dt.year
        self.population_data         = population_data[['timestamp', id_col, 'population_size']].drop_duplicates().reset_index(drop=True)
        self.static_previewer        = StaticGraphViewer(self.graph_registry, self.shapes) 
        self.graph_methods = ['population_weighted', 'gravity_model', 'commuter']

    def generate_graphstructure(self, 
                                method:             str,                              
                                time_window:        List[Union[Timestamp, str]], 
                                frequency:          str = 'yearly',                                 
                                graphname:          Optional[str] = None,                                    
                                self_connection:    Literal['mean','0','max','eps'] = '0', 
                                normalization:      Optional[str] = None, 
                                **kwargs) -> None:
        """
        Generates a dynamic graphstructure which is added to the graph registry

        Parameters
        ----------
        method: str
            dynamic graphstructure- method. any of:
            - population_weighted
            - gravity_model
            - commuter

        time_window: List[Union[Timestamp, str]]
            ...

        frequency: str = 'yearly'
            frequency of graphs to be created

        graphname: Optional[str]
            the name under which the graph entry will be saved. Recommended to supply. 
            If None, an artificial name will be generated according to `_generate_graphname`

        self_connection: Literal['mean','0','max'] = '0'
            how to create self-loops. By default '0' => None

        normalization: Optional[Literal['minmax','log','zscore','symmetric','rowwise']] = None
            how to normalize edge-weights. By default non-normalized

        Downstream
        ----------
        when called, this method calls upon:
        - RawGraphConstructor
            creates RawGraphStructure depending on method called and **kwargs. 
            this is a collection of a list of edge-index and a list of edge-weights
        - SelfLoopAdder
            adds self-loops depending on method called in, according to parameter self_connection
        - EdgeWeightNormalizer
            normalizes edge-weights according to parameter normalization
        
        to complete a GraphEntry which is to be added to graphregistry, further called are:
        - self._generate_graphstats() 
            returns a dictionary of statistics for the graph
        - StaticGraphConfig
            returns a dictionary of the configuration for the graph
            
        then the graphstructure is added to the registry:
            self.graph_registry.add_entry()
        """        
        if method not in self.graph_methods:
            raise ValueError(f'invalid method {method}. Supported methods: {self.graph_methods}')
        
        if graphname is None:
            graphname = self._generate_graphname(method, self_connection, normalization)

        # Prepare population data if needed
        if method in ['population_weighted', 'gravity_model']:
            all_external_data     = self._resample_population_data(frequency)
            all_external_data     = all_external_data[[self.id_col, 'population_size','timestamp']]
        # Prepare time axis
        freq        = self._set_frequency(frequency)
        time_axis   = self._return_list_timestamps(time_window, frequency, freq)

        if method == 'commuter':
            all_external_data = CommuterDataLoader(years=time_axis).return_data()
            all_external_data['year'] = all_external_data['year'].astype(int)
            all_external_data=all_external_data.rename(columns = {'year':'timestamp'})
            
            all_external_data = all_external_data[['nuts3_work', 'nuts3_residence', 'commuters', 'timestamp']]                       

        # Generate raw graph structures for each timestamp
        graph_generator = RawGraphConstructor(shapedata=self.shapes.copy(), 
                                             tokens=self.tokens, 
                                             id_col=self.id_col)
        
        structures = []
        for tt in tqdm(time_axis,desc=f"generating graph => {graphname}", total = len(time_axis)):

            external_data_tt = all_external_data[all_external_data['timestamp'] == tt].reset_index(drop = True).drop(columns = 'timestamp')

            if len(external_data_tt) == 0:
                raise ValueError(f'{error_emoji} empty dataframe found for {tt}')

            if method == 'commuter':
                raw_structure = graph_generator.generate_graph(method=method, 
                                                            commuter_data=external_data_tt, 
                                                            **kwargs)                
            else:
                raw_structure = graph_generator.generate_graph(method=method, 
                                                            population_data=external_data_tt, 
                                                            **kwargs)
            
            raw_structure = SelfLoopAdder(rawgraphstructure=raw_structure, 
                                            node_ids=self.node_ids).add_loops(self_connection)

            # Convert to tensors
            edge_weight                 = torch.tensor(raw_structure.edge_weight_ls, dtype=torch.float)
            edge_index                  = torch.tensor(raw_structure.edge_index_ls,   dtype=torch.long).t().contiguous() 

            graphstructre_prenormalized = GraphStructure(edge_index, edge_weight)

            # Edge-weight normalization (removes zero values)
            edgeweight_normalizer       = EdgeWeightNormalizer(graphstructre_prenormalized)
            graphstructure_normalized   = edgeweight_normalizer.normalize(normalization)

            structures.append(graphstructure_normalized)

        # timestamps_pd   = pd.to_datetime(time_axis)
        # timestamps_unix = torch.tensor([int(ts.timestamp()) for ts in timestamps_pd], dtype=torch.int64)
        time_axis_str = [str(tt) for tt in time_axis]

        # Create final structures
        dynamicgraphstructure   = DynamicGraphStructure(time_axis_str, structures)
        graphstats              = self._generate_graphstats(dynamicgraphstructure, time_axis_str)
        graphconfig             = DynamicGraphConfig(method, time_axis, frequency, 
                                        self_connection, normalization, kwargs)

        self.graph_registry.add_entry(graphname, 
                                     GraphEntry(dynamicgraphstructure, graphstats, graphconfig, 'dynamic'))

    def preview_graphstructure(self,
                               graphname: Optional[str] = None,
                               time_idx:  Optional[int] = None,
                               node_idx:  Optional[int]  = None,
                               subplots:  bool           = True,
                               title:     Optional[str]  = None) -> Figure:
        """
        preview a static graphstructure in self.graph_registry
        
        Parameters
        ----------
        graphname: Optional[str] = None
            name under which graph is saved in self.graph_registry
            when none, the emtpy graph structure is viewed
        time_idx: Optional[int] = None
            timestamp for which to show the graph
            when none, the graphname must also be none
        node_idx: Optional[int] = None
            a node to zoom into (neighborhood will be shown)
            when none, the global graph structure is viewed
        subplots: bool = True
            whether or not to show some more information in side-panels
            when False, only the main map is shown
        title: Optional[str] = None
            extra title to be shown

        Returns
        -------
        Figure

        Downstream
        ----------
        heavy lifting is done by StaticGraphViewer
        """
        if not graphname:
            graphname = 'empty'
        return self.static_previewer.view(graphname, node_idx, subplots, title, time_idx = time_idx)

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