import os
import torch
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, List, Union, Literal, Tuple


from .edgeweight_normalizer import EdgeWeightNormalizer
from .graphconstructor import GraphConstructor
from .selfloopadder import SelfLoopAdder
from .graphviewer import GraphViewer

from .graphregistry import GraphEntry, GraphRegistry
from .graphstructures import GraphStructure, DynamicGraphStructure
from .graphstats import StaticGraphStats, DynamicGraphStats

from ..dataloading import DataOrchestrator


@dataclass
class GraphConfig:
    """
    Config with which graph structure was created

    Parameters
    ----------
    method: str
        name of the method with which graphstructure was generated
    
    self_connection: str
        connections of nodes to themselves (options are 'mean', '0' and 'max')

    kwargs: Optional[dict]= None

    Downstream
    ----------
    GraphRegistry contains numerous GraphEntry - objects
    Each of those is associated with each of the following objects:
    - GraphStructure
    - GraphStatistics
    - GraphConfig    
    """
    method:         str
    self_connection:str
    scaling_method: Optional[str] = None
    mode:           Literal['static','dymamic'] = None
    kwargs:         Optional[dict]= None

class GraphOrchestrator:
    """
    Orchestrates the entire process of graph creation

    Parameters
    ----------
    data_orchestrator: DataOrchestrator
        object with which data orchestration was created

    id_col: str = 'node'
        column used accross dataframes to refer to nodes

    graph_dir: str = 'data/graphs'
        directory in which to store, and from which to retrieve, graphs

    Methods
    -------
    generate_graph  
    preview_graph


    Examples
    --------
    #### Graph Generation
    >>> data_orchestrator   = ...
    >>> graphconstruction   = GraphOrchestrator(data_orchestrator=data_orchestrator)

    >>> # identity graph
    >>> graphconstruction.generate_graph(method = 'identity')
    >>> graphconstruction.rename_graph('identity_selfmean', 'identity_graph')

    >>> # mesh graph
    >>> graphconstruction.generate_graph(method = 'mesh')
    >>> graphconstruction.rename_graph('mesh_selfmean', 'mesh_graph')

    >>> # Neighbors
    >>> #   boolean-self
    >>> graphconstruction.generate_graph(method='boolean_neighbors')
    >>> graphconstruction.rename_graph('boolean_neighbors_selfmean',            'geographical_neighbors1')
    >>> #   boolean-nonself
    >>> graphconstruction.generate_graph(method='boolean_neighbors', self_connection='0')
    >>> graphconstruction.rename_graph('boolean_neighbors_self0',               'geographical_neighbors2')
    >>> #   numerical-self
    >>> graphconstruction.generate_graph(method='boolean_neighbors', scaling_method='rowwise')
    >>> graphconstruction.rename_graph('boolean_neighbors_selfmean_rowwise',    'geographical_neighbors3')
    >>> #   numerical-nonself
    >>> graphconstruction.generate_graph(method='boolean_neighbors', scaling_method='rowwise', self_connection='0')
    >>> graphconstruction.rename_graph('boolean_neighbors_self0_rowwise',       'geographical_neighbors4')

    >>> # Gravity Models
    >>> #   sparse - long distance gravity model
    >>> graphconstruction.generate_graph(method             = 'gravity_model', 
                                    self_connection    = '0',
                                    max_distance       = 1000_000,
                                    top_k              = 3,
                                    alpha              = 1,
                                    decay              = 1,
                                    scaling_method     = 'rowwise'
                                    )
    >>> graphconstruction.rename_graph('gravity_model_self0_rowwise','gravity1')
    >>> #   sparse - long distance gravity model
    >>> graphconstruction.generate_graph(method             = 'gravity_model', 
                                    self_connection    = '0',
                                    max_distance       = 1000_000,
                                    top_k              = 10,
                                    alpha              = 1,
                                    decay              = 1,
                                    scaling_method     = 'rowwise'
                                    )
    >>> graphconstruction.rename_graph('gravity_model_self0_rowwise','gravity2')
    >>> #   sparse - short distance gravity model
    >>> graphconstruction.generate_graph(method             = 'gravity_model', 
                                    self_connection    = '0',
                                    max_distance       = 1000_000,
                                    top_k              = 3,
                                    alpha              = 2,
                                    decay              = 1,
                                    scaling_method     = 'rowwise'
                                    )
    >>> graphconstruction.rename_graph('gravity_model_self0_rowwise','gravity3')
    >>> #   dense - short distance gravity model
    >>> graphconstruction.generate_graph(method             = 'gravity_model', 
                                    self_connection    = '0',
                                    max_distance       = 1000_000,
                                    top_k              = 10,
                                    alpha              = 2,
                                    decay              = 1,
                                    scaling_method     = 'rowwise'
                                    )
    >>> graphconstruction.rename_graph('gravity_model_self0_rowwise','gravity4')
    >>> #   medium-dense - medium-distance gravity model
    >>> graphconstruction.generate_graph(method             = 'gravity_model', 
                                    self_connection    = '0',
                                    max_distance       = 1000_000,
                                    top_k              = 7,
                                    alpha              = 1.5,
                                    decay              = 1,
                                    scaling_method     = 'rowwise'
                                    )
    >>> graphconstruction.rename_graph('gravity_model_self0_rowwise','gravity5')
    >>> # Commuter
    >>> #   static - 2024-1
    >>> #   low threshold
    >>> graphconstruction.generate_graph(
        method              = 'commuter', 
        self_connection     = '0',
        commuting_threshold = 500,
        scaling_method      = 'rowwise',
        name_addition       = '1'
        )
    >>> graphconstruction.rename_graph('commuter_1_self0_rowwise', 'static_commuter24_1')
    >>> #   static - 2024-2
    >>> #   medium threshold
    >>> graphconstruction.generate_graph(
        method              = 'commuter', 
        self_connection     = '0',
        commuting_threshold = 1000,
        scaling_method      = 'rowwise',
        name_addition       = '2'
        )
    >>> graphconstruction.rename_graph('commuter_2_self0_rowwise', 'static_commuter24_2')
    >>> #   static - 2024-3
    >>> #   high threshold
    >>> graphconstruction.generate_graph(
        method              = 'commuter', 
        self_connection     = '0',
        commuting_threshold = 2500,
        scaling_method      = 'rowwise',
        name_addition       = '3'
        )
    graphconstruction.rename_graph('commuter_3_self0_rowwise', 'static_commuter24_3')
    #   static - 2024-4
    #   top_k=4
    >>> graphconstruction.generate_graph(
         method              = 'commuter', 
         self_connection     = '0',
         commuting_threshold = 1000,
         scaling_method      = 'rowwise',
         name_addition       = '4',
         top_k               = 4
         )
    >>> graphconstruction.rename_graph('commuter_4_self0_rowwise', 'static_commuter24_4')

    #### Previewing
    >>> figure_empty                    = graphconstruction.preview_graph('empty',                                                          title= "Preview Germany NUTS3")
    >>> figure_identity_graph           = graphconstruction.preview_graph('identity_graph',             node_idx = 26,  subplots = True,    title= "Preview identity_graph for Hannover")

    #### Saving
    >>> graphconstruction.save_graphentry('identity_graph')

    Limitations #TODO
    -----------
    - population_size is determined as average per node over all years
    - currently deals with static graphs only, dynamic graphs should be dealt with

    See Also
    --------
    Child classes in this module:
    - EdgeWeightNormalizer
    - GraphConstructor
    - GraphViewer
    - SelfLoopAdder
    """
    def __init__(self,                  
                 data_orchestrator: DataOrchestrator,
                 id_col:          str = 'node',
                 graph_dir:       str = "data/graphs/"):
        
        # extract metadata
        shapes               = data_orchestrator.data_context.shapedata
        self.tokens          = data_orchestrator.data_context.tokenization_map['id_idx']
        self.epipopdata      = data_orchestrator.data_harmonized.data
        
        # mean population data TODO => currently mean is taken instead of yearly
        self.population_data = self.epipopdata.groupby(id_col)['population_size'].mean().reset_index()
        self.shapes          = shapes.copy()
        self.id_col          = id_col 
        self.nuts_level      = data_orchestrator.data_context.nuts_level
        self.graph_dir       = os.path.join(graph_dir, f'{self.nuts_level}')

        # registry of graphs
        os.makedirs(self.graph_dir, exist_ok=True)
        self.graph_registry = GraphRegistry(self.graph_dir )
        self.previewer      = GraphViewer(self.graph_registry, self.shapes)

        self.graph_methods          = ['boolean_neighbors', 
                                       'identity', 
                                       'mesh', 
                                       'distance_threshold',
                                       'k_nearest', 
                                       'population_weighted', 
                                       'gravity_model', 
                                       'commuter']
        self.dynamic_methods        = ['population_weighted',  'gravity_model', 'commuter']
        
        self.num_nodes              = data_orchestrator.data_context.num_nodes

    def generate_graph(self, 
                       method:          str                                                             = 'identity',
                       name_addition:   Optional[str]                                                   =  None,
                       self_connection: Literal['max','0','mean']                                       = 'mean',
                       scaling_method:  Optional[Literal['minmax','log','zscore','symmetric','rowwise']]= None,
                       mode:            Literal['static','dynamic']                                     = 'static',
                       **kwargs) -> None:
        """
        Generates a graph structure based on the method. Depending on the method, additional kwargs may be required.
        A graph structure and config are created and saved into the dictionary `self.graph_registry` under the key
        corresponding to `graph_name`, which is equal to:

            method + name_addition + self{self_connection} + scaling_method

        where '_' is used as separator

        Parameters
        ----------
        method: str 
            limited to: ['boolean_neighbors', 'identity', 'mesh', 'distance_threshold','k_nearest',  'population_weighted', 'gravity_model', 'commuter']
        name_addition: Optional[str]
            adjustment of graphname
        self_connection: Literal['max','0','mean']
            how to deal with self_connection
        scaling_method: Optional[Literal['minmax','log','zscore','symmetric','rowwise']] = None
            how to deal with edge_weights
        mode: Literal['static','dynamic'] = 'static'
            whether to create a static or dynamic graph structure.
            only certain methods are compatible with dynamic graph structure
        **kwargs:
            kwargs are method-specific.

        See Also
        --------
        The heavy lifting is done through the following classes. Each of these contains further information.
            - GraphGeneration
            - GraphEdgeWeightNormalizer
            - GraphAddSelfLoops
        """

        if method not in self.graph_methods:
            raise ValueError(f'{method} not a valid graph method. Please choose a method from this list:\n{self.graph_methods}')
        
        if mode == 'dynamic':

            if method not in self.dynamic_methods:
                print(f'for method {method}, no dynamic graph possible. mode is set to "static"')

                mode = 'static'

        graphconfig = GraphConfig(method, self_connection, scaling_method, mode, kwargs)
        graphname   = self._generate_graphname(method, name_addition, self_connection, scaling_method)

        # Ensure IDs are integers and no missing
        shapes_cp               = self.shapes.dropna(subset=[self.id_col])
        shapes_cp[self.id_col]  = shapes_cp[self.id_col].astype(int)
        node_ids                = np.array(shapes_cp[self.id_col].dropna().astype(int).values)

        # generate graph:   
        graph_generator = GraphConstructor(gdf = shapes_cp, tokens = self.tokens, popdata = self.population_data, id_col = self.id_col)
        edges, weights  = graph_generator.generate_graph(method=method, mode = mode, **kwargs)
        
        if weights is None:                                                                                                             # if weights is undefined, give 1 everywhere
            weights = [float(1) for _ in edges]

        # self-loops => only when method is neither identity nor mesh
        if method not in ['identity', 'mesh']:
            edges, weights = SelfLoopAdder(edge_indices=edges, edge_weights=weights, node_ids=node_ids).add_loops(self_connection)
        edges, weights  = self._remove_zero_weights(edges, weights)                                                                     # remove zero valued loops        

        edge_weight     = torch.tensor(weights, dtype=torch.float)
        edge_index      = torch.tensor(edges,   dtype=torch.long).t().contiguous()

        # edge-weight normalization
        if scaling_method:
           edge_weight = EdgeWeightNormalizer(edge_indices=edge_index, edge_weights=edge_weight, num_nodes=self.num_nodes).normalize(scaling_method)

        edge_index, edge_weight = self._remove_zero_weights(edge_index, edge_weight)                                                    # remove zero valued loops  (now post-normalization)

        graphstructure = GraphStructure(edge_index, edge_weight)

        if mode == 'static':
            graphentry     = GraphEntry(graphstructure, self._generate_static_graph_stats(graphstructure), asdict(graphconfig))

        self.graph_registry.add_entry(graphname, graphentry)

    def preview_graph(self, graphname: str = 'empty', node_idx: Optional[int] = None, subplots: bool = True, title: Optional[str] = None):        
        """
        Preview graph found in registry

        Parameters
        ----------
        graphname: str = 'empty'
            the name under which the graph structure is saved in the registry (.graph_registry shows registered graphs)
            for viewing an empty graphstructure (unconnected nodes) use graphname = 'empty'
        node_idx: Optional[int] = None
            the node of which to view the neighborhood (when int)
            by default, view global graph (node_idx = None)
        subplots: bool = True
            whether or not to show more (distributions) than just a global map
        title: Optional[str] = None
            title for the main (global) map
            
        See Also
        --------
        GraphViewer -> does the actual heavy lifting. This method simply relays parameters.
        """
        return self.previewer.view(graphname, node_idx, subplots, title)

# GraphRegistry - maintenance
    def rename_graph(self, old_graphname: str, new_graphname: str) -> None:
        """ 
        Rename a graph in the registry (the key by which the graph is saved)
        the old graph is copied into the `new graphname` and the `old_graphname` is removed.
        """
        self.graph_registry.rename_entry(old_graphname, new_graphname)
        
    def get_graph_stats(self, graphname: str, type: Optional[Literal['small','large']] = 'large') -> str:
        """Returns a string representation of the graph statistics, either small or extensive"""
        self.graph_registry.get_graph_stats(graphname, type)

    def remove_entry(self, graphname: str) -> None:
        """CAREFUL with this one. Removes a GraphEntry"""
        self.graph_registry.remove_entry(graphname)      

    def save_graphentry(self, graphname: Union[str,List[str]]) -> None:
        """
        Save edge index and weight from registry. 
        If graphname == 'all', all graphs are saved.
        """

        if graphname == 'all' or graphname == ['all']:
            graph_entries_to_save = self.graph_registry.return_entrynames()

        elif isinstance(graphname, str):
            graph_entries_to_save = [graphname]

        else:
            raise ValueError(f'Please provide a list or string for the graphname(s).')


        for graphname in graph_entries_to_save:

            self.graph_registry.save_graphentry(graphname)

            

# HelperFunctions        
    def _generate_graphname(self, method: str, name_addition: str, self_connection: str, scaling_method: str):
        """returns graphname str"""
        graphname = f'{method}_{name_addition}'         if name_addition    else f'{method}'
        graphname = f'{graphname}_self{self_connection}'if self_connection  else f'{graphname}'
        graphname = f'{graphname}_{scaling_method}'     if scaling_method   else f'{graphname}'
        return graphname

    def _remove_zero_weights(self, edge_index: Union[List[Tuple[int, int]], torch.Tensor], edge_weight: Union[List[float], torch.Tensor], threshold: float = 1e-9
    ) -> Tuple[Union[List[Tuple[int, int]], torch.Tensor], Union[List[float], torch.Tensor]]:
        """
        Remove edges with zero or near-zero weights.
        Type (List or torch.Tensor) depends on the input type
        
        Parameters
        ----------
        edge_index : Union[List[Tuple[int, int]], torch.Tensor]
            Edge indices as list of tuples or tensor [2, num_edges]
        edge_weight : Union[List[float], torch.Tensor]
            Edge weights as list or tensor [num_edges]
        threshold : float
            Values below this are considered zero (default: 1e-9)
            
        Returns
        -------
        Tuple containing filtered edge_index and edge_weight
        """
        if isinstance(edge_index, torch.Tensor):
            if not isinstance(edge_weight, torch.Tensor):
                raise TypeError(
                    f'edge_index is torch.Tensor but edge_weight is {type(edge_weight).__name__}'
                )
            
            mask = edge_weight.abs() > threshold
            return edge_index[:, mask], edge_weight[mask]
        
        elif isinstance(edge_index, list):
            if not isinstance(edge_weight, list):
                raise TypeError(
                    f'edge_index is list but edge_weight is {type(edge_weight).__name__}'
                )
            
            # Filter and unzip in one go
            filtered = [(e, w) for e, w in zip(edge_index, edge_weight) if abs(w) > threshold]
            
            if not filtered:  # Handle empty case
                return [], []
            
            filtered_edges, filtered_weights = zip(*filtered)
            return list(filtered_edges), list(filtered_weights)
        
        else:
            raise TypeError(
                f'edge_index must be list or torch.Tensor, got {type(edge_index).__name__}'
            )       

    def _generate_dynamic_graph_stats(self, graph_structures: DynamicGraphStructure) -> DynamicGraphStats:
        """ 
        Returns a summary of the dynamic graph
        """ 
        graph_snapshot = graph_structures[0]

        num_graphs = len(graph_structures)       
        num_nodes  = int(graph_snapshot.edge_index.max().item()) + 1

        return DynamicGraphStats(
            num_graphs = num_graphs,
            num_nodes  = num_nodes
        )

    def _generate_static_graph_stats(self, graph_structure: GraphStructure) -> StaticGraphStats:
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
        return f'<GraphConstructor(level {self.nuts_level}. Registry: {self.graph_registry})>'