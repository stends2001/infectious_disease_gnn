from typing import Optional, List, Dict, Any, Literal

from .builder import GraphBuilder
from .contextdataprocessor import GraphContextDataProcessor
from .postprocessor import GraphPostProcessor
from .graphpreviewer import GraphViewer

from ..graphregistry import GraphRegistry
from ..graphobjects import GraphObject, GraphStructure, TopKConfig, GraphConfig

from ...utils.pathmanager import PathManager
from ...utils.types import GraphType, GraphNormType, Country, Level

import logging
logger = logging.getLogger(__name__)

class InvalidPreviewInput(Exception):

    def __init__(self, msg: str):
        super().__init__(msg)

class GraphManager:
    """
    Main class used to construct graphs. The heavy lifting is outsourced to 'delegation classes'.
    Note that each instance of GraphManager is associated with a single combination of country and level.
    
    Parameters
    ----------
    country: Country
        country represented by the graph structure(s) to be created
        Country = Literal['germany', 'hungary']        
    level: Level
        level of the country represented by the graph structure(s) to be created
        Level = Literal['nuts1', 'nuts2', 'nuts3']
    id_col: str = 'key
        the column name in which the code of each spatial unit is stored. These are the ones that will be 
        mapped to node-idx (tokens) alphabetically, the mapping of which will be stored in the overarching
        directory (self.dir_graphs_partition) under tokenization_map.json
    token_col: str = 'node'
        the column name in which the tokens of the `id_col` will be stored

    Attributes
    ----------
    - `graph_registry`
    - `num_nodes`
    - `tokenization_map`

    See Also
    --------
    #### Delegation classes
    - GraphContextDataProcessor
    - GraphBuilder
    - GraphPostProcessor
    - GraphViewer

    #### Others
    - PathManager
    - GraphRegistry
    """
    def __init__(self,                  
                 country:           Country,
                 level:             Level,
                 
                 id_col:            str = 'key',
                 token_col:         str = 'node'):

        self.id_col             = id_col 
        self.token_col          = token_col   
        self.level              = level
        self.country            = country

        # ======== MANAGE PATHS ======== #
        self._pathmanager        = PathManager()
        self.dir_country_data    = self._pathmanager.data / 'data' / country        # 
        self.dir_graphs_root     = self.dir_country_data  / 'graphs'                # path points to all graphs of all different partitions (nuts1/2/3)
        self.dir_graphs_partition= self.dir_graphs_root   / level                   # path points to all graphs of 1 partition (nuts1 OR nuts2 OR nuts3). These have a shared tokenization map

        if not self.dir_graphs_root.exists():
            self.dir_graphs_root.mkdir()
            logger.info('directory %s created', self.dir_graphs_root)

        if not self.dir_graphs_partition.exists():
            self.dir_graphs_partition.mkdir()        
            logger.info('directory %s created', self.dir_graphs_partition)                

        # ========= DELEGATION CLASSES ============ #
        self.graph_registry = GraphRegistry(self.dir_graphs_partition)

        self.preprocessor   = GraphContextDataProcessor(level, id_col, token_col, self.dir_country_data)
        self.shape_data, self.population_data, self.tokenization_map = self.preprocessor.process()
        self.num_nodes                                               = len(self.tokenization_map.keys())
        
        self.builder        = GraphBuilder(self.id_col, self.token_col, self.shape_data, self.population_data)        
        self.postprocessor  = GraphPostProcessor() 
        self.previewer      = GraphViewer(self.preprocessor.shp_raw, self.shape_data, self.country, self.level)
    
    # ======= HIDDEN METHODS ======== #
    def construct_graph(self, 
                        graph_name: str,                         
                        graph_type: GraphType, 
                        normalization_method: GraphNormType, 
                        top_k: Optional[Dict[str,Any]] = None, 
                        *args, **kwargs) -> None:
        """
        Main function of the class. Creates and processes a graph strucutre, 
        which is saved in self.graph_registry

        Parameters
        ----------
        graph_name: str
            the name under which graph has been saved
        graph_type: GraphType
            the type of graph. The following are supported: 
            GraphType = Literal['identity', 'geographical_contiguity', 'gravity_model', 'random', 'fully_connected']
        normalization_method: GraphNormType
            method used to normalize edge_weights. the following are supported:
            GraphNormType = Literal['minmax', 'symmetric', 'rowwise']
        top_k: Optional[Dict[str,Any]]  
            top-k arguments. Optional, may therefore be None, or a dictionary with keys 'k': int and 'mode': Literal['local','global']
        args: List[Any]
            any other arguments (`seed` for graph_type == 'random')
        kwargs: Dict[str, Any]
        """
        topk_cfg = None if top_k is None else TopKConfig(**top_k)

        # step 1: produce graphconfig:
        graphcfg = self._build_graphconfig(
            graph_name,            
            graph_type,
            self.num_nodes,
            normalization_method,
            topk_cfg,
            *args, **kwargs
        )
        logger.info('Graph construction %s update: graphconfig made', graph_name)    

        # step 2: get edge_index and edge_weight as lists
        edge_index, edge_weight = self.builder.build(graph_type, *args, **kwargs)
        logger.info('Graph construction %s update: raw graph created', graph_name)    

        # step 3: get raw graphstructure instance
        graph_structure         = GraphStructure.from_list(edge_index, edge_weight, self.num_nodes)
        logger.info('Graph construction %s update: GraphStructure class initiated', graph_name)    

        # step 4: optionally, top k graph => select top k indices
        if top_k is not None:
            graph_structure     = self.postprocessor.filter_top_k(graph_structure, **top_k)   
            logger.info('Graph construction %s update: top_k selection done', graph_name)         

        # step 5: normalize weights
        graph_structure         = self.postprocessor.normalize(graph_structure, normalization_method)
        logger.info('Graph construction %s update: edges normalized', graph_name)            

        # step 6: create graphobject
        graph_object = GraphObject(
            graph_structure, 
            self.tokenization_map,
            graphcfg
        )
        logger.info('Graph construction %s update: GraphObject class initiated', graph_name)                

        # step 7: store graph in graph_registry
        self.graph_registry.add_entry(graph_name, graph_object)

    # ======= METHODS ======== #
    def _build_graphconfig(self, 
                           graph_name: str,
                           graph_type: GraphType,
                           num_nodes: int,
                           normalization_method: GraphNormType,
                           top_k: Optional[TopKConfig],
                           *args: List[Any],
                           **kwargs: Dict[str, Any]) -> GraphConfig:
        """builds and returns an instance of GraphConfig based on input from `construct_graph()`"""
        
        return GraphConfig(
            graph_name, 
            graph_type,
            num_nodes,             
            normalization_method,
            top_k, 
            args,
            kwargs
        )
    
    def preview(self, 
                graph_name: str, 
                variable:   Literal['edge_weights','network','degree','strength','strength_vs_degree'],
                locality:   Literal['local','global'],
                plot_type:  Literal['histogram','map'],
                neighborhood: Optional[int],
                connections_type: Optional[Literal['in','out']]):
        """ 
        Previews registered GraphStructure from `graph_registry`.

        Parameters
        ----------
        graph_name: str
            the name under which GraphStructure is stored in graph_registry.
        variable: Literal['edge_weights','connections','network']
            the variable to preview
        locality: Literal['local','global']
            the context of the variable; local means in a neighborhood, global means over the graph.
        plot_type: Literal['histogram','map']
            whether to plot a histogram or a map
        neighborhood: Optional[int] = None
            which neighborhood to plot (only applicable when locality == 'local')
        connections_type: Optional[Literal['in','out']] = None
            which type of connections to plot (i.e. ingoing or outgoing).
        """

        self._validate_view_input(variable, locality, plot_type, neighborhood, connections_type)

        graph_object = self.graph_registry.get_entry(graph_name)

        if graph_object is None:
            return

        return self.previewer.view(graph_object.graph, variable, locality, plot_type, neighborhood, connections_type)
      
    def _validate_view_input(
            self,
            variable: Literal['edge_weights','network','degree','strength','strength_vs_degree'],
            locality: Literal['local','global'],
            plot_type: Literal['histogram','map'],
            neighborhood: Optional[int] = None,
            connections_type: Optional[Literal['in','out']] = None
        ):              
            """validates the input given into `preview()`"""               
            match (variable, locality, plot_type, neighborhood, connections_type):

                # 'network' must use 'map' plot
                case ('network', _, 'histogram', _, _):
                    raise InvalidPreviewInput(
                        'Invalid combination of view arguments: '
                        'When variable == "network" plot_type must be "map"'
                    )
                
                # 'edge_weights' cannot be global map
                case ('edge_weights', 'global', 'map', _, _):
                    raise InvalidPreviewInput(
                        'Invalid combination of view arguments: '
                        'When variable == "edge_weights" and plot_type == "map", locality must be "local".'
                    )
                
                # local neighborhood must be provided
                case (_, 'local', _, None, _):
                    raise InvalidPreviewInput(
                        'Invalid combination of view arguments: '
                        'When locality == "local", neighborhood must be supplied.'
                    )
            
                # local neighborhood_type must be provided
                case (_, 'local', _, _, None):
                    raise InvalidPreviewInput(
                        'Invalid combination of view arguments: '
                        'When locality == "local", connections_type must be supplied.'
                    )
                
                # strength in a local setting doesn't make any sense
                case ('strength' | 'strength_vs_degree', 'local', _, _, _):
                    raise InvalidPreviewInput(
                        'Invalid combination of view arguments: '
                        'strength is meaningless with locality == "local"'
                    )      

                case ('degree', 'local', _, _, _):
                    raise InvalidPreviewInput(
                            "'degree' map is only meaningful with locality='global'. "
                            "Use ('edge_weights', 'map') with locality='local' to inspect a specific neighborhood."
                        )      
                case ('edge_weights', 'global', _, _, 'in' | 'out'):          
                        raise InvalidPreviewInput(
                            'Invalid combination of arguments. ' \
                            'Edge-weights-distribution globally of a connection type is meaningless. Globally, in-connections = out-connections.')
                # anything else is valid
                case _:
                    pass

    def __repr__(self) -> str:
        representation = f"<{self.__class__.__name__}({self.country}, {self.level})>"
        return representation