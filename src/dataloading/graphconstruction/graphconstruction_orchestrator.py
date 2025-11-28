import os
import torch
import numpy as np
import seaborn as sns
from dataclasses import dataclass, asdict
from typing import Optional, List, Union, Literal, Dict

from .edgeweight_normalizer import EdgeWeightNormalizer
from .graphconstructor import GraphConstructor
from .selfloop_adder import SelfLoopAdder
from .graphviewer import GraphViewer

from ..dataorchestration.dataorchestrator import DataOrchestrator

from ...utils.textformatting import checkmark, warning_emoji

@dataclass 
class GraphStructure:
    """
    Single graphstructure with

    Parameters:
    ----------
    edge_index: torch.Tensor
        index of edges
        shape [num_edges, 2]
    edge_weight: torch.Tensor
        weights of edges
        shape [num_edges, 1]
    """
    edge_index:     torch.Tensor 
    edge_weight:    torch.Tensor    
    
    def __repr__(self) -> str:
        num_nodes      = len(self.edge_index[0].unique())
        num_edges      = len(self.edge_index)
        representation = f'<GraphEntry(num_nodes = {num_nodes}, num_edges = {num_edges})>'
        return representation

class GraphRegistry:
    """
    Registry of GraphStructure objects

    Attributes:
    ----------
    registry: Dict[str, GraphStructure]
        graphnames : GraphStructure object
    
    Methods:
    -------
    add_entry       ->  None
    rename_entry    ->  None
    get_entry       ->  GraphStructure
    """
    def __init__(self):
        self.registry: Dict[str, GraphStructure] = {}

    def add_entry(self, graphname: str, structure: GraphStructure) -> None:
        """Add structure to .registry under graphname"""
        if self.check_entry(graphname):
            print(f'{warning_emoji} {graphname} already exists, please rename the already existing entry')
        else:
            self.registry[graphname] = structure
        
    def rename_entry(self, current_graphname: str, new_graphname: str) -> None:
        """rename entry from current_graphname to new_graphname; current_graphname is removed"""        
        if self.check_entry(new_graphname):
            print(f'{warning_emoji} {new_graphname} already exists, please rename the already existing entry')
        else:
            self.registry[new_graphname] = self.registry[current_graphname]
            del self.registry[current_graphname]            
            print(f'{current_graphname} has been replaced by {new_graphname}')

    def get_entry(self, graphname: str) -> 'GraphStructure':
        """return GraphStructure from graphname"""
        if not self.check_entry(graphname):
            print(f'{warning_emoji} {graphname} not found')
            registered_entries = ', '.join(self.return_entrynames())
            raise ValueError(f"the following graphs are registered:\n{registered_entries}")
        
        else:
            return self.registry[graphname]

    def check_entry(self, graphname: str) -> bool:
        """return boolean reflecting whether or not graphname is registered"""
        if graphname in self.registry.keys():
            return True
        else:
            return False
        
    def return_entrynames(self) -> List[str]:
        """returns a list of entrynames"""
        return list(self.registry.keys())
        
    def __repr__(self) -> str:
        registered_entries = ', '.join(self.return_entrynames())
        return f'<GraphRegistry({registered_entries})'

@dataclass
class GraphConfig:
    """
    Config with which graph structure was created
    """
    method:         str
    name_addition:  Optional[str]
    self_connection:str
    scaling_method: Optional[str]
    kwargs:         dict

palette_blues = sns.color_palette("Blues", n_colors=100)
palette_reds  = sns.color_palette("Reds", n_colors=100)


class GraphOrchestrator:
    """
    Orchestrates the entire process of graph creation

    Parameters:
    ----------
    data_orchestrator: DataOrchestrator
        object with which data orchestration was created
    id_col: str = 'node'
        column used accross dataframes to refer to nodes
    graph_dir: str = 'data/graphs'
        directory in which to store, and from which to retrieve, graphs

    Methods:
    -------
    generate_graph  
    preview_graph


    Examples:
    --------
    >>> orch                = data_orchestrator
    >>> graphconstruction   = GraphOrchestrator(data_orchestrator=data_orchestrator)
    >>> graphconstruction.generate_graph('commuter', scaling_method='rowwise')
    >>> graphconstruction.rename_graph('commuter_selfmean_rowwise', 'commuter')
    >>> graphconstruction.generate_graph('boolean_neighbors')
    >>> graphconstruction.rename_graph('boolean_neighbors_selfmean', 'boolean_neighbors')

    >>> graphconstruction.preview_graph('commuter', node_idx = 223, subplots = True, title= "Preview commuter graph from Munich")
    >>> graphconstruction.preview_graph('boolean_neighbors', subplots = True, title= "Preview boolean neighbors graph")

    Limitations: #TODO
    -----------
    - population_size is determined as average per node over all years
    - currently deals with static graphs only, dynamic graphs should be dealt with
    - no selfloops visualization
    
    See Also:
    ------------
    GraphRegistry -> registry of graph structures (.graph_registry)
    GraphViewer   -> graph previewer object (.previewer)


    """
    def __init__(self,                  
                 data_orchestrator: DataOrchestrator,
                 id_col:          str = 'node',
                 graph_dir:       str = "data/graphs/"):
        
        # extract metadata
        shapes               = data_orchestrator.data_context.shapedata
        self.tokens          = data_orchestrator.data_context.tokenization_map['id_idx']
        self.epipopdata      = data_orchestrator.data_context.epipopdata
        
        # mean population data TODO => currently mean is taken instead of yearly
        self.population_data = self.epipopdata.groupby(id_col)['population_size'].mean().reset_index()
        self.shapes          = shapes.copy()
        self.id_col          = id_col 
        self.nuts_level      = data_orchestrator.data_context.nuts_level
        self.graph_dir       = os.path.join(graph_dir, f'{self.nuts_level}')

        # registry of graphs
        self.graph_registry = GraphRegistry()
        self.previewer      = GraphViewer(self.graph_registry, self.shapes)
        os.makedirs(self.graph_dir, exist_ok=True)

        self.graph_methods          = ['boolean_neighbors', 
                                       'identity', 
                                       'mesh', 
                                       'distance_threshold',
                                       'k_nearest', 
                                       'population_weighted', 
                                       'gravity_model', 
                                       'commuter']
        
        self.num_nodes              = data_orchestrator.data_context.num_nodes
        
    def generate_graph(self, 
                       method: str                                          = 'boolean_neighbors',
                       name_addition:   Optional[str]                       = None,
                       self_connection:  Literal['max','0','mean']          = 'mean',
                       scaling_method:  Optional[Literal['minmax','log','zscore','symmetric','rowwise']] = None,
                       **kwargs) -> None:
        """
        Generates a graph structure based on the method. Depending on the method, additional kwargs may be required.
        A graph structure and config are created and saved into the dictionary `self.graph_registry` under the key
        corresponding to `graph_name`, which is equal to:

            method + name_addition + self{self_connection} + scaling_method

        where '_' is used as separator

        Parameters:
        ----------
        method: str 
            limited to: ['boolean_neighbors', 'identity', 'mesh', 'distance_threshold','k_nearest',  'population_weighted', 'gravity_model', 'commuter']
        name_addition: Optional[str]
            adjustment of graphname
        self_connection: Literal['max','0','mean']
            how to deal with self_connection
        scaling_method: Optional[Literal['minmax','log','zscore','symmetric','rowwise']] = None
            how to deal with edge_weights
        **kwargs:
            kwargs are method-specific.

        See also:
        --------
        The heavy lifting is done through the following classes. Each of these contains further information.
            - GraphGeneration
            - GraphEdgeWeightNormalizer
            - GraphAddSelfLoops
        """

        if method not in self.graph_methods:
            raise ValueError(f'{method} not a valid graph method. Please choose a method from this list:\n{self.graph_methods}')

        graphconfig = GraphConfig(
            method          = method,
            name_addition   = name_addition,
            self_connection = self_connection,
            scaling_method  = scaling_method,
            kwargs          = kwargs
            )

        
        graphname = f'{method}_{name_addition}'         if name_addition    else f'{method}'
        graphname = f'{graphname}_self{self_connection}'if self_connection  else f'{graphname}'
        graphname = f'{graphname}_{scaling_method}'     if scaling_method   else f'{graphname}'

        # Ensure IDs are integers and no missing
        shapes_cp               = self.shapes.dropna(subset=[self.id_col])
        shapes_cp[self.id_col]  = shapes_cp[self.id_col].astype(int)
        node_ids                = np.array(shapes_cp[self.id_col].dropna().astype(int).values)

        ##########################
        ##### Create Graphs ######
        ##########################     
        graph_generator = GraphConstructor(
            gdf     = shapes_cp,
            tokens  = self.tokens,
            popdata = self.population_data,
            id_col  = self.id_col
        )

        # Generate the graph with whatever method and kwargs
        edges, weights = graph_generator.generate_graph(method=method, **kwargs)

        # if weights is undefined, give 1 everywhere
        if weights is None:
            weights = [float(1) for _ in edges]

        ##########################
        ##### Add self-loops #####
        ##########################        
        if method not in ['identity', 'mesh']:
            edges, weights = SelfLoopAdder(edge_indices=edges, edge_weights=weights, node_ids=node_ids).add_loops(self_connection)

        # remove zero valued loops        
        edges   = [edge for edge, weight in zip(edges, weights) if weight != 0]
        weights = [weight for weight in weights if weight != 0]   

        # transform into torch objects        
        edge_weight = torch.tensor(weights, dtype=torch.float)
        edge_index  = torch.tensor(edges, dtype=torch.long).t().contiguous()

        ##########################
        # Normalize edge-weights #
        ##########################
        if scaling_method:
           edge_weight = EdgeWeightNormalizer(edge_indices=edge_index, edge_weights=edge_weight, num_nodes = self.num_nodes).normalize(scaling_method)

        graphstructure = GraphStructure(edge_index, edge_weight)

        # save config
        graphdict = {'structure': graphstructure,
                     'config'   : asdict(graphconfig),
                     'summary'  : self._get_graph_summary(edge_index,edge_weight)}   

        self.graph_registry.add_entry(graphname, graphstructure)
        print(f'{checkmark} graph generated: {graphname}')

    def preview_graph(self, graphname: str, node_idx: Optional[int] = None, subplots: bool = True, title: Optional[str] = None):        
        """
        Preview graph found in registry

        Parameters:
        ----------
        graphname: str
            the name under which the graph structure is saved in the registry (.graph_registry shows registered graphs)
            for viewing an empty graphstructure (unconnected nodes) use graphname = 'empty'
        node_idx: Optional[int] = None
            the node of which to view the neighborhood (when int)
            by default, view global graph (node_idx = None)
        subplots: bool = True
            whether or not to show more (distributions) than just a global map
        title: Optional[str] = None
            title for the main (global) map
            
        See Also:
        --------
        GraphViewer -> does the actual heavy lifting. This method simply relays parameters.
        """
        return self.previewer.view(graphname, node_idx, subplots, title)

    def rename_graph(self, old_graphname: str, new_graphname: str) -> None:
        """ 
        Rename a graph in the registry (the key by which the graph is saved)
        the old graph is copied into the `new graphname` and the `old_graphname` is removed.
        """
        self.graph_registry.rename_entry(old_graphname, new_graphname)
        
    def save_graph(self, graphname: Union[str,List[str]] = 'all') -> None:
        """
        Save edge index and weight from registry. 
        If graphname == 'all', all graphs are saved.
        """

        if graphname == ['all']:
            graphname = 'all'

        if graphname == 'all':
            graph_entries_to_save = self.graph_registry.return_entrynames()

        elif isinstance(graphname, str):
            graph_entries_to_save = [graphname]

        else:
            raise ValueError(f'Please provide a list or string for the graphname.')

        for graphname in graph_entries_to_save:

            graph_entry = self.graph_registry.get_entry(graphname)
            if graph_entry is not None:
                
                edge_index  = graph_entry.edge_index
                edge_weight = graph_entry.edge_weight

                torch.save(edge_index, os.path.join(self.graph_dir, f'{graphname}_edge_index.pt'))
                print(f'{checkmark} graph saved: edge index {graphname} saved to {self.graph_dir}')

                if edge_weight is not None:
                    torch.save(edge_weight, os.path.join(self.graph_dir, f'{graphname}_edge_weight.pt'))
                    print(f'{checkmark} graph saved: edge weight {graphname} saved to {self.graph_dir}')

    def _get_graph_summary(self, global_edge_index: torch.Tensor, global_edge_weight: torch.Tensor) -> Dict[str, float]:

        """ 
        validate the graph created
        """

        # statistics
        num_edges           = global_edge_index.shape[1]
        num_nodes           = int(global_edge_index.max().item())+1
        edge_density        = num_edges / (num_nodes * (num_nodes - 1))
        edge_weight_np      = global_edge_weight.cpu().numpy()
        edge_weight_mean    = edge_weight_np.mean()
        edge_weight_min     = edge_weight_np.min()
        edge_weight_max     = edge_weight_np.max()

        # isolated nodes:
        edges_out, edges_in = global_edge_index[0], global_edge_index[1] 
        out_degree          = torch.bincount(edges_out, minlength=num_nodes)
        in_degree           = torch.bincount(edges_in, minlength=num_nodes)
        isolated_mask       = (out_degree == 0) & (in_degree == 0)
        num_isolated        = isolated_mask.sum().item()

        summary = {
            'num_nodes'     : num_nodes,
            'num_edges'     : num_edges,
            'edge_density'  : edge_density,
            'num_isolated'  : num_isolated,
            'edge_weight_mean'  : edge_weight_mean,
            'edge_weight_min'   : edge_weight_min,
            'edge_weight_max'   : edge_weight_max
        }   

        return summary
    
    def __repr__(self) -> str:
        return f'<GraphConstructor(level {self.nuts_level}. Registry: {self.graph_registry})>'