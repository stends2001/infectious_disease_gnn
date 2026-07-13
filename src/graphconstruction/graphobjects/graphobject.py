from dataclasses import dataclass 
import torch 
from typing import Dict, Self, Union, ClassVar
import os 
from pathlib import Path 

from .graphconfig import GraphConfig
from .graphstructure import GraphStructure
from ..exceptions import InvalidGraphObject
from ...utils.helpers import PathNotFound, load_mapping_dict, save_mapping_dict

import logging
logger = logging.getLogger(__name__)

@dataclass 
class GraphObject:
    """
    Class that stores a graph structure

    Parameters
    -----------
    graph: GraphStructure
        the torch representation of graph
    tokenization_map: Dict[str,int]
        mapping of node-identifier (i.e. NUTS code) -> node idx
    config: GraphConfig   
        config-dataclass for the graph 

    One may also set a graph structure with lists  instead of torch.Tensors, using
    the classmethod `from_list()`. These are then directly converted into Tensors.

    Examples
    ---------
    >>> edges   = [(0,1), (1,0), (3,1)]
    >>> graph   = GraphStructure.from_list(edges, [1, 1, 1], 4)    
    >>> graphobj= GraphObject(graph, {'A': 0, 'B': 1, 'C': 2, 'D':3}, {})    

    """
    graph:              GraphStructure
    tokenization_map:   Dict[str, int]   
    config:             GraphConfig

    # these are class variables (has to be typed explicitly as ClassVar when in dataclasses)
    edge_index_filename:            ClassVar[str] = 'edge_index.pt'
    edge_weight_filename:           ClassVar[str] = 'edge_weight.pt'
    graphconfig_filename:           ClassVar[str] = 'config.json'
    tokenization_map_filename:      ClassVar[str] = 'tokenization_map.json'

    def __post_init__(self):
        self._validate()

    def save(self, path: Union[str, Path]):
        """ 
        Saves four things:
        - graphconfig       => path / config.json
        - edge-index        => path / edge_index.pt
        - edge-weight       => path / edge_weight.pt
        - tokenization_map  => path.parent / tokenization_map.json (only saves if doesn't exist yet!)
        """

        if isinstance(path, str):
            path = Path(path)

        graphconfig_dict = self.config.asdict()

        if not path.exists():
            path.mkdir()

        save_mapping_dict(graphconfig_dict, path / self.graphconfig_filename)

        torch.save(self.graph.edge_index, path / self.edge_index_filename)
        torch.save(self.graph.edge_weight, path /self.edge_weight_filename)             
        
        if self.tokenization_map_filename not in os.listdir(str(path.parent)):
            save_mapping_dict(self.tokenization_map, path.parent / self.tokenization_map_filename)
            logger.info('Tokenization map saved under %s.', path.parent / self.tokenization_map_filename)

    @classmethod
    def load(cls, path: Union[str, Path]) -> Self:
        """
        Loads an instance of itself based on supplied path.
        That is, the graph-folder name containing the files:
        - `config.json`
        - `edge_index.pt`
        - `edge_weight.pt`

        Returns
        -------
        An instance of GraphObject
        """
        if isinstance(path, str):
            path = Path(path)

        # validate path-existence
        if not path.exists():
            raise PathNotFound(path)

        # validate Tokenization map's path
        tokenization_map_path = path.parent / cls.tokenization_map_filename
        if not tokenization_map_path.exists():
            raise PathNotFound(tokenization_map_path)            

        # validate specific files' presence
        for ff in [cls.graphconfig_filename, cls.edge_index_filename, cls.edge_weight_filename]:
            filepath  = path / ff
            if not filepath.exists():
                raise PathNotFound(filepath)                

        graphconfig_dict= load_mapping_dict(path / cls.graphconfig_filename)
        graphconfig     = GraphConfig.fromdict(graphconfig_dict)        

        edge_index      = torch.load(path / cls.edge_index_filename, weights_only=True)
        edge_weight     = torch.load(path / cls.edge_weight_filename, weights_only=True)
        logger.info('Graph %s loaded from file.', path.stem)

        graphstructure  = GraphStructure(edge_index, edge_weight, graphconfig.num_nodes)

        tokenization_map: Dict[str, int]= load_mapping_dict(path.parent / cls.tokenization_map_filename)

        return cls(graphstructure, tokenization_map, graphconfig)

    def _validate(self):
        num_expected_nodes = len(self.tokenization_map.values())
        
        if num_expected_nodes != self.graph.num_nodes:
            raise InvalidGraphObject(f'GraphStructure has {self.graph.num_nodes} nodes but tokenization map has {num_expected_nodes}.')                 
        
        if len(self.tokenization_map.values()) != len(set(self.tokenization_map.values())):
            raise InvalidGraphObject(f'Found doubles inside the values of tokenzation map!')    

    @property 
    def reverse_tokenization_map(self) -> Dict[int, str]:
        return  {v: k for k, v in self.tokenization_map.items()}
    
    def __repr__(self) -> str:
        return (
            f"GraphObject(graph={self.graph})"
        )
