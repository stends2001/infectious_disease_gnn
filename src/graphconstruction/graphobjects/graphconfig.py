from dataclasses import dataclass 
from typing import Optional , Any, Dict, Literal, Union
from ..utils import GraphType, GraphNormType

class InvalidTopKConfig(Exception):
    
    def __init__(self):
        msg = f'k needs to be at least 1'
        super().__init__(msg)

@dataclass(frozen=True)
class TopKConfig:
    """Simple dataclass for top_k configuration. Downstream use in GraphConfig"""

    k: int
    mode: Literal["global", "local"]

    def __post_init__(self):
        if self.k < 1:
            raise InvalidTopKConfig()
        
    def asdict(self) -> Dict[str, Union[int, str]]:
        """returns dictionary-ready representation for in graphconfig(.yaml)"""
        return {'k' : self.k, 'mode': self.mode}


@dataclass 
class GraphConfig:
    """
    Config class containing info on how the graph was generated

    Parameters
    ----------
    graph_name: str
        the name under which graph has been saved
    graph_type: GraphType
        the type of graph. The following are supported: 
        GraphType = Literal['identity', 'geographical_contiguity', 'gravity_model', 'random', 'fully_connected']
    num_nodes: int    
        number of nodes in the relevant graph structure (also counting isolated nodes that don't show up in edge_index!)
    normalization_method: GraphNormType
        method used to normalize edge_weights. the following are supported:
        GraphNormType = Literal['minmax', 'symmetric', 'rowwise']
    top_k: Optional[TopKConfig]  
        top-k arguments. Optional, may therefore be None, or an instance of TopKConfig
    args: List[Any]
        any other arguments (`seed` for graph_type == 'random')
    kwargs: Dict[str, Any]

    Methods
    -------
    - `asdict()`
    - `fromdict()` (classmethod)

    Downstream Use
    --------------
    GraphObject
        contains GraphStructure and an instance of GraphConfig
    """
    
    graph_name: str     
    graph_type: GraphType
    num_nodes: int
    
    normalization_method: GraphNormType
    top_k: Optional[TopKConfig]  

    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def asdict(self) -> dict[str, Any]:
        """returns a config (yaml) file representation of itself. Antithesis of `cls.fromdict()`."""
        return {
            "graph_name": self.graph_name,
            "graph_type": self.graph_type,
            "num_nodes": self.num_nodes,
            "normalization_method": self.normalization_method,
            "top_k": None if self.top_k is None else self.top_k.asdict(),
            "args": list(self.args),
            "kwargs": self.kwargs,
        }

    @classmethod
    def fromdict(cls, d: dict[str, Any]) -> "GraphConfig":
        """reads a config (yaml) file's content of an instance of itself. Antithesis of `self.asdict()`."""        
        top_k = d.get("top_k")

        if top_k is not None:
            top_k = TopKConfig(**top_k)

        return cls(
            graph_name=d["graph_name"],
            graph_type=d["graph_type"],
            num_nodes=d["num_nodes"],
            normalization_method=d["normalization_method"],
            top_k=top_k,
            args=tuple(d.get("args", [])),
            kwargs=d.get("kwargs", {}),
        )    