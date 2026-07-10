from dataclasses import dataclass 
import torch 
from typing import List, Tuple, Self

from ..exceptions import InvalidGraphStructure

@dataclass 
class GraphStructure:
    """ 
    Graph Structure class

    Parameters
    -----------
    edge_index:     torch.Tensor [2, num_edges]
        list-wise-representation of edge index
    edge_weight:    torch.Tensor [num_edges]
        list-wise-representation of edge weights
    num_nodes:  int
        number of nodes represented in the graph

    Attributes
    -----------
    - `num_edges`
    - `adjacency_matrix`
    - `edge_index_list`
    - `edge_weight_list`

    Downstream use
    ---------------
    - GraphObject

    Examples
    ---------
    >>> edges   = [(0,1), (1,0), (3,1)]
    >>> graph   = GraphStructure.from_list(edges, [1, 1, 1], 4)
    """
    edge_index:     torch.Tensor
    edge_weight:    torch.Tensor 
    num_nodes:      int

    def __post_init__(self):
        self._validate()

    @property
    def num_edges(self):
        return self.edge_index.shape[1]
    
    @property
    def adjacency_matrix(self) -> torch.Tensor:
        return self._get_adjacency_matrix()
        
    @property
    def edge_index_list(self) -> List[Tuple[int, int]]:
        return [tuple(edge) for edge in self.edge_index.tolist()]

    @property
    def edge_weight_list(self) -> List[float]:
        return self.edge_weight.tolist()

    def _validate(self):
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise InvalidGraphStructure(f'Expected edge_index shape [2, num_edges] but got {self.edge_index.shape}')

        if self.edge_weight.ndim != 1:
            raise InvalidGraphStructure(f'Expected edge_weight shape [num_edges] but got {self.edge_weight.shape}')            

        if self.num_edges != self.edge_weight.shape[0]:
            raise InvalidGraphStructure(f"edge_index and edge_weight length mismatch ({self.num_edges}, {self.edge_weight.shape[0]})")

        if self.edge_index.min().item() < 0:
            raise InvalidGraphStructure(f"Node IDs must be non-negative. Got {self.edge_index.min()}")

        if self.edge_index.max().item() >= self.num_nodes:
            raise InvalidGraphStructure(f"Maximum node index is {self.num_nodes}. Starting counting from 0, expected largest to be {self.num_nodes - 1} but got {self.edge_index.max().item()}")

    def _get_adjacency_matrix(self) -> torch.Tensor:
        """returns tensor of adjacency matrix"""
        adj = torch.zeros(
            (self.num_nodes, self.num_nodes),
            dtype=self.edge_weight.dtype,
            device=self.edge_index.device
        )

        adj[self.edge_index[0], self.edge_index[1]] = self.edge_weight

        return adj    
    
    @classmethod
    def from_list(cls, 
                  edge_index:       List[Tuple[int, int]], 
                  edge_weight:      List[float],
                  num_nodes:        int) -> Self:
        """
        Returns an instance using Lists as parameters rather than tensors.
        
        Parameters
        ----------
        edge_index: List[Tuple[int, int]]
            shape must be [2, num_edges]
            accepts [num_edges, 2] in addition
        edge_weight: List[float]
            shape must be [num_edges] 
        """
        edge_index_tensor = torch.tensor(edge_index,  dtype = torch.long)
        edge_weight_tensor= torch.tensor(edge_weight, dtype = torch.float)

        if edge_index_tensor.ndim == 2 and edge_index_tensor.shape[1] == 2:
            edge_index_tensor = edge_index_tensor.t()

        return cls(edge_index_tensor, edge_weight_tensor, num_nodes)    
    
    def __repr__(self) -> str:
        representation = f'<{self.__class__.__name__}(num_nodes = {self.num_nodes}, num_edges = {self.num_edges})>'
        return representation 


