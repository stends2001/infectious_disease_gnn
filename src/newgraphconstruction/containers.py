from dataclasses import dataclass
from typing import List, Tuple
import torch

@dataclass
class RawGraphStructure:
    """
    Raw graph structure containing

    Parameters
    ----------
    edge_index_ls:  List[Tuple[int, int]]
        each edge is a tuple of node-ids
    edge_weight_ls: List[float]
        a list of edge weights; a weight per edge

    Downstream
    ----------
    """
    edge_index_ls:  List[Tuple[int, int]]
    edge_weight_ls: List[float]

    def __post_init__(self):
        if len(self.edge_index_ls) != len(self.edge_weight_ls):
            raise ValueError(
                f"Incompatiable shapes of edge_index (len = {len(self.edge_index_ls)}) and edge_weight (len = {self.edge_weight_ls}) in RawGraphStructure"
            )

    def __repr__(self) -> str:
        num_nodes = len({v for edge in self.edge_index_ls for v in edge})
        num_edges = len(self.edge_index_ls)
        return f"<RawGraphStructure(num_nodes={num_nodes}, num_edges={num_edges})>"


@dataclass
class DynamicRawGraphStructure:
    """
    Sequence of RawGraphStructure objects with associated timestamps

    Parameters
    ----------
    timestamps: List[str]
        list of timestamps, each of which associated with a RawGraphStructure
    rawstructures: List[RawGraphStructure]
        list of RawGraphStructures, associated with timestamps

    Methods
    -------
    - length
    - iterate
    - getitem

    Downstream
    ----------
    """
    timestamps:    List[str]
    rawstructures: List[RawGraphStructure]

    def __post_init__(self):
        T = len(self.timestamps)
        if len(self.rawstructures) != T:
            raise ValueError(
                f"Number of rawstructures ({len(self.rawstructures)}) must match timestamps ({T})"
            )

    def __len__(self):
        return len(self.timestamps)

    def __iter__(self):
        return iter(self.rawstructures)

    def __getitem__(self, idx) -> Tuple[RawGraphStructure, str]:
        return self.rawstructures[idx], self.timestamps[idx]

    def __repr__(self):
        return f"<DynamicRawGraphStructure(T={len(self)})>"


@dataclass 
class GraphStructure:
    """
    """
    edge_index:     torch.Tensor 
    edge_weight:    torch.Tensor    

    def __post_init__(self):
        if len(self.edge_index) != len(self.edge_weight):
            raise ValueError(
                f"Incompatiable shapes of edge_index (len = {len(self.edge_index[0])}) and edge_weight (len = {self.edge_weight}) in RawGraphStructure"
            )
        
        self.num_nodes      = len(self.edge_index[0].unique())
        self.num_edges      = len(self.edge_index)
    
    def __repr__(self) -> str:
        representation = f'<GraphStructure(num_nodes = {self.num_nodes}, num_edges = {self.num_edges})>'
        return representation 