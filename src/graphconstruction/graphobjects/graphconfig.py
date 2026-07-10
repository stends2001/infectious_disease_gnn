from dataclasses import dataclass 
from typing import Optional, List , Any, Dict, Literal
from ...utils.types import GraphType, GraphNormType

class InvalidTopKConfig(Exception):
    msg = f'k needs to be at least 1'

@dataclass(frozen=True)
class TopKConfig:
    k: int
    mode: Literal["global", "local"]

    def __post_init__(self):
        if self.k < 1:
            raise InvalidTopKConfig()


@dataclass 
class GraphConfig:
    """
    Config class containing info on how the graph was generated

    Parameters
    ----------
    graph_name:str

    graph_type: str

    num_nodes: int    

    normalization_method: str

    top_k: Optional[str]

    args: List[Any]

    kwargs: Dict[str, Any]
    """
    
    graph_name: str     
    graph_type: GraphType
    num_nodes: int
    
    normalization_method: GraphNormType
    top_k: Optional[TopKConfig]  

    args: tuple[Any, ...]
    kwargs: dict[str, Any]