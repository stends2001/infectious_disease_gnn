import torch 
from dataclasses import dataclass

class GraphStructureError(Exception):
    def __init__(self, explanation: str):
        statement = "Error found in graphstructure" + "\n" + explanation
        super().__init__(statement)

@dataclass 
class GraphStructure:
    edge_index:     torch.Tensor 
    edge_weight:    torch.Tensor    

    def __post_init__(self):
        self.num_nodes      = len(self.edge_index[0].unique())
        self.num_edges      = len(self.edge_index)

        self._validate_graphshape()

    def _validate_graphshape(self):
        # check number of edges
        if self.edge_index.shape[1] != len(self.edge_weight):
            raise GraphStructureError(
                f"len of edge index is (len = {self.edge_index.shape[1]}) and edge_weight (len = {len(self.edge_weight)})"
            )      
    
    def __repr__(self) -> str:
        representation = f'<GraphStructure(num_nodes = {self.num_nodes}, num_edges = {len(self.edge_weight)})>'
        return representation 