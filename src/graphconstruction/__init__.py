from .orchestrators import StaticGraphOrchestrator, DynamicGraphOrchestrator

import torch

def has_self_loops(edge_index: torch.Tensor) -> bool:
    """
    Checks whether the graph has self-loops.
    
    Parameters
    ----------
        edge_index: torch.Tensor of shape [2, num_edges]
    
    Returns
    -------
        True if any edge is a self-loop, False otherwise.
    """
    # Compare source and target nodes
    return torch.any(edge_index[0] == edge_index[1]).item()