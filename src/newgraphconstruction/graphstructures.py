import torch
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd

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

@dataclass
class DynamicGraphStructure:
    """
    Dynamic graph structure storing snapshots over time.
    
    Parameters
    ----------
    timestamps : torch.Tensor
        Tensor of shape [T] containing timestamp values (Unix timestamps)
    edge_indices : List[torch.Tensor]
        List of T tensors, each of shape [2, E_t] containing edge indices
    edge_weights : List[torch.Tensor]
        List of T tensors, each of shape [E_t] containing edge weights
    """
    timestamps: torch.Tensor        # Unix timestamps as int64
    edge_indices: List[torch.Tensor]
    edge_weights: List[torch.Tensor]
    
    def __post_init__(self):
        T = len(self.timestamps)
        if len(self.edge_indices) != T:
            raise ValueError(f"Number of edge_indices ({len(self.edge_indices)}) must match timestamps ({T})")
        if len(self.edge_weights) != T:
            raise ValueError(f"Number of edge_weights ({len(self.edge_weights)}) must match timestamps ({T})")
    
    def get_snapshot(self, t_idx: int):
        """Get static graph structure at time index t_idx"""
        from .graphstructures import GraphStructure
        return GraphStructure(
            edge_index=self.edge_indices[t_idx],
            edge_weight=self.edge_weights[t_idx]
        )
    
    def get_timestamps_as_datetime(self) -> List[pd.Timestamp]:
        """Convert Unix timestamps back to pandas Timestamps"""
        return [pd.Timestamp(ts, unit='s') for ts in self.timestamps.tolist()]
    
    def __len__(self):
        """Return number of time snapshots"""
        return len(self.timestamps)
    
    def __repr__(self):
        dates = self.get_timestamps_as_datetime()
        return f"<DynamicGraphStructure(T={len(self)}, {dates[0]} to {dates[-1]})>"