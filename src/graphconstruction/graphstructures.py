import torch
from dataclasses import dataclass
from typing import List, Tuple

@dataclass 
class GraphStructure:
    """
    """
    edge_index:     torch.Tensor 
    edge_weight:    torch.Tensor    
    
    def __repr__(self) -> str:
        num_nodes      = len(self.edge_index[0].unique())
        num_edges      = len(self.edge_index)
        representation = f'<GraphStructure(num_nodes = {num_nodes}, num_edges = {num_edges})>'
        return representation   

class DynamicGraphStructure:
    """
    Iterable container of GraphStructure instances aligned with timestamps.

    Parameters
    ----------
    graphstructures : List[GraphStructure]
        List of GraphStructure objects. Indexing corresponds to timestamps.
    timestamps : List[str]
        Timestamp labels, same length as graphstructures.
    """   
    def __init__(self, graphstructures: List[GraphStructure], timestamps: List[str]):
        self._validate_graphtimes(graphstructures, timestamps)
        self.graphstructures = graphstructures  
        self.timestamps      = timestamps

    def __iter__(self):
        return iter(self.graphstructures)
    
    def __len__(self):
        return len(self.graphstructures)    
    
    def __getitem__(self, idx: int) -> Tuple[GraphStructure, str]:
        return self.graphstructures[idx], self.timestamps[idx]    
    
    def __repr__(self):
        if not self.graphstructures:
            return f"<DynamicGraphStructure(empty)>"
        
        return f"<DynamicGraphStructure(n={len(self.graphstructures)} graphstructures)>"

    @staticmethod
    def _validate_graphtimes(graphstructures: List[GraphStructure], timestamps: List[str]):
        if len(graphstructures) != len(timestamps):
            raise ValueError(
                f"number of graphstructures ({len(graphstructures)}) "
                f"must equal number of timestamps ({len(timestamps)})"
            )

