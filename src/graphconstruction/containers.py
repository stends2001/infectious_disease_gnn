from dataclasses import dataclass
from typing import List, Tuple, Union
import torch
import pandas as pd

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
                f"Incompatiable shapes of edge_index (len = {len(self.edge_index_ls)}) and edge_weight (len = {len(self.edge_weight_ls)}) in RawGraphStructure"
            )

    def __repr__(self) -> str:
        num_nodes = len({v for edge in self.edge_index_ls for v in edge})
        num_edges = len(self.edge_index_ls)
        return f"<RawGraphStructure(num_nodes={num_nodes}, num_edges={num_edges})>"

@dataclass 
class GraphStructure:
    """
    edge_index:     torch.Tensor 
    edge_weight:    torch.Tensor   
    """
    edge_index:     torch.Tensor 
    edge_weight:    torch.Tensor    

    def __post_init__(self):
        if self.edge_index.shape[1] != len(self.edge_weight):
            raise ValueError(
                f"Incompatiable shapes of edge_index (len = {self.edge_index.shape[1]}) and edge_weight (len = {len(self.edge_weight)}) in GraphStructure"
            )
        
        self.num_nodes      = len(self.edge_index[0].unique())
        self.num_edges      = len(self.edge_index)
    
    def __repr__(self) -> str:
        representation = f'<GraphStructure(num_nodes = {self.num_nodes}, num_edges = {len(self.edge_weight)})>'
        return representation 
    
@dataclass
class DynamicGraphStructure:
    """
    Dynamic graph structure storing snapshots over time.
    
    Parameters
    ----------
    timestamps : torch.Tensor
        Tensor of shape [T] containing timestamp values (Unix timestamps)
    graphstructures: List[GraphStructure]
    """
    timestamps:      List[pd.Timestamp | str]       
    graphstructures: List[GraphStructure]
    
    def __post_init__(self):
        T = len(self.timestamps)
        if len(self.graphstructures) != T:
            raise ValueError(f"Number of graphstructures ({len(self.graphstructures)}) must match timestamps ({T})")
        
    def get_snapshot(self, t: Union[pd.Timestamp, str]):
        """Return the GraphStructure at timestamp t."""

        times = [
            ts for ts in self.timestamps
        ]

        try:
            idx = times.index(t)
        except ValueError:
            raise KeyError(f"Timestamp {t} not found in DynamicGraphStructure")

        return self.graphstructures[idx]
    
    def __len__(self):
        """Return number of time snapshots"""
        return len(self.timestamps)
    
    def __getitem__(self, idx: int):
        return self.timestamps[idx], self.graphstructures[idx]
    
    def __iter__(self):
        return iter(zip(self.timestamps, self.graphstructures))
    
    def __repr__(self):
        dates = self.timestamps
        return f"<DynamicGraphStructure(T={len(self)}, {dates[0]} to {dates[-1]})>"
