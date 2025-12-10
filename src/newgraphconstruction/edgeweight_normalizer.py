import torch
from typing import Tuple
from .containers import GraphStructure

class EdgeWeightNormalizer:
    """ 
    Normalizes edge-weights in a number of different ways (See Methods)
    Before the normalization, edges with weight zero are removed.

    Parameters
    ----------
    graphstructure: GraphStructure

    Returns
    -------
    ...

    Methods
    -------
    - normalize
        the main orchestration function that calls the other hidden functions
    - _remove_zeroes
    - _minmax
    - _log
    - _zscore
    - _symmetric
    - _rowwise

    Examples
    --------
    >>> edge_weight = EdgeWeightNormalizer(graphstructure).normalize(scaling_method)
    
    Downstream
    ----------
    """    
    def __init__(self,
                 graphstructure: GraphStructure):
        
        self.graphstructure= graphstructure
        self.num_nodes     = graphstructure.num_nodes

        self.NORMALIZATION_FUNCS = {
                                    'minmax'    : self._minmax,
                                    'log'       : self._log,
                                    'zscore'    : self._zscore,
                                    'symmetric' : self._symmetric,
                                    'rowwise'   : self._rowwise
        }

    def normalize(self, method: str) -> GraphStructure:
        """
        collects the required generation-function and feeds in the kwargs
        """
        # filter zero weights
        edge_index, edge_weight = self._remove_zeros(
            self.graphstructure.edge_index,
            self.graphstructure.edge_weight
        )

        if method not in self.NORMALIZATION_FUNCS:
            raise KeyError(
                f"Unknown normalization method: {method}. Available methods are: "
                f"{', '.join(self.NORMALIZATION_FUNCS.keys())}"
            ) 
        
        # pass tensors explicitly instead of storing to self
        normalize_fn        = self.NORMALIZATION_FUNCS[method]
        normalized_weights  = normalize_fn(edge_index, edge_weight)

        return GraphStructure(edge_index, normalized_weights)        
         
    def _remove_zeros(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, threshold: float = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        removes zero weights
        when ran, should be done so before normalization!
        """
        mask = edge_weight > threshold
        return edge_index[:, mask], edge_weight[mask]

    def _minmax(self, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """ 
        Normalizes edge-weights globally by linearly mapping between epison and 1
        """
        min_w = edge_weight.min()
        max_w = edge_weight.max()
        if max_w > min_w:
            edge_weight = (edge_weight - min_w) / (max_w - min_w)
        else:
            edge_weight = torch.zeros_like(edge_weight)

        return edge_weight 
    
    def _log(self, edge_index: torch.Tensor, edge_weight: torch.Tensor,) -> torch.Tensor:
        """ 
        Normalizes edge weights using log transformation followed by scaling to [0, 1].
        """        
        edge_weight = torch.log1p(edge_weight)
        max_w = edge_weight.max()
        if max_w > 0:
            edge_weight = edge_weight / max_w
        else:
            edge_weight = torch.zeros_like(edge_weight)
        return edge_weight

    def _zscore(self, edge_index: torch.Tensor, edge_weight: torch.Tensor,) -> torch.Tensor:
        """ 
        Normalizes edge weights using z-score standardization: w_norm = (w - μ) / σ

        Note
        ----
        it may seem unconventional to have negative weights included, but a graph's 
        edge-weights just another continuous feature of an arbitrary scale. 
        Negative edge-weights should be handled by any model in the project.
        """
        mean_w = edge_weight.mean()
        std_w  = edge_weight.std()

        if std_w > 0:
            edge_weight = (edge_weight - mean_w) / std_w
        else:
            edge_weight = torch.zeros_like(edge_weight)

        return edge_weight

    def _symmetric(self, edge_index: torch.Tensor, edge_weight: torch.Tensor,) -> torch.Tensor:
        """ 
        Applies symmetric (bi-directional) normalization: D^(-1/2) * W * D^(-1/2)
        """        
        row, col = edge_index[0], edge_index[1]

        deg = torch.zeros(self.num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
        deg.scatter_add_(0, row, edge_weight)
        
        # Compute D^{-1/2} for each node, avoid division by zero
        deg_inv_sqrt                                = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')]  = 0
        
        # For each edge (i,j), normalized weight is w_ij * D_i^{-1/2} * D_j^{-1/2}
        edge_weight = edge_weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]    
        return edge_weight

    def _rowwise(self, edge_index: torch.Tensor, edge_weight: torch.Tensor,) -> torch.Tensor:
        """ 
        Applies row-wise normalization: each node's outgoing edges sum to 1
        This tends to be the most commonly used version to normalizes edge-weights.
        """        
        # Row-normalize: divide each edge weight by sum of weights in the source node's row
        row, _ = edge_index
        deg = torch.zeros(self.num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
        deg.scatter_add_(0, row, edge_weight)
        
        # avoid division by zero
        deg_inv = torch.zeros_like(deg)
        nonzero_mask = deg > 0
        deg_inv[nonzero_mask] = 1.0 / deg[nonzero_mask]
        
        edge_weight = edge_weight * deg_inv[row]    
        return edge_weight