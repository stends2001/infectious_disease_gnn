# EdgeWeightNormalizer
import torch

class EdgeWeightNormalizer:
    """ 
    normalizes edge-weights. 
    Called in the process of GraphConstructor

    Parameters:
    ----------
    edge_indices: torch.Tensor
    edge_weights: torch.Tensor
    num_nodes: int

    Examples:
    --------
    >>> edge_weight = EdgeWeightNormalizer(edge_indices=edge_index, 
    >>>                                    edge_weights=edge_weight, 
    >>>                                    num_nodes = self.num_nodes).normalize(scaling_method)

    See also:
    --------
    GraphConstructor    → creates and returns an edge_index and a edge_weight
    GraphOrchestrator   → orchestrates the creation of a graph through GraphConstructor
    """    
    def __init__(self,
                 edge_indices: torch.Tensor,
                 edge_weights: torch.Tensor,
                 num_nodes   : int):
        
        self.edge_weights = edge_weights 
        self.edge_indices = edge_indices
        self.num_nodes    = num_nodes


        self.NORMALIZATION_FUNCS = {
                                    'minmax'    : self._minmax,
                                    'log'       : self._log,
                                    'zscore'    : self._zscore,
                                    'symmetric' : self._symmetric,
                                    'rowwise'   : self._rowwise
        }

    def normalize(self, method: str) -> torch.Tensor:
        """
        collects the required generation-function and feeds in the kwargs
        """

        if method not in self.NORMALIZATION_FUNCS:
            raise KeyError(f"Unknown normalization method: {method}. Available methods are: {', '.join(self.NORMALIZATION_FUNCS.keys())}")
        
        return self.NORMALIZATION_FUNCS[method]()
        
    def _minmax(self) -> torch.Tensor:
        """ 
        Normalizes edge-weights globally by linearly mapping between 0 and 1.
        """
        min_w = self.edge_weights.min()
        max_w = self.edge_weights.max()
        if max_w > min_w:
            edge_weights = (self.edge_weights - min_w) / (max_w - min_w)
        else:
            edge_weights = torch.zeros_like(self.edge_weights)

        return edge_weights 
    
    def _log(self) -> torch.Tensor:
        """ 
        Normalizes edge weights using log transformation followed by scaling to [0, 1].
        """        
        edge_weights = torch.log1p(self.edge_weights)
        max_w = edge_weights.max()
        if max_w > 0:
            edge_weights = edge_weights / max_w
        else:
            edge_weights = torch.zeros_like(edge_weights)
        return edge_weights

    def _zscore(self) -> torch.Tensor:
        """ 
        Normalizes edge weights using z-score standardization: w_norm = (w - μ) / σ
        """
        mean_w = self.edge_weights.mean()
        std_w  = self.edge_weights.std()
        if std_w > 0:
            edge_weights = (self.edge_weights - mean_w) / std_w
        else:
            edge_weights = torch.zeros_like(self.edge_weights)

        return edge_weights

    def _symmetric(self) -> torch.Tensor:
        """ 
        Applies symmetric (bi-directional) normalization: D^(-1/2) * W * D^(-1/2)
        """        
        # Build adjacency matrix in sparse form
        # D_ii = sum of weights connected to node i
        row, col = self.edge_indices[0], self.edge_indices[1]

        deg = torch.zeros(self.num_nodes, dtype=self.edge_weights.dtype, device=self.edge_weights.device)
        deg.scatter_add_(0, row, self.edge_weights)
        
        # Compute D^{-1/2} for each node, avoid division by zero
        deg_inv_sqrt                                = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')]  = 0
        
        # For each edge (i,j), normalized weight is w_ij * D_i^{-1/2} * D_j^{-1/2}
        edge_weights = self.edge_weights * deg_inv_sqrt[row] * deg_inv_sqrt[col]    
        return edge_weights

    def _rowwise(self) -> torch.Tensor:
        """ 
        Applies row-wise normalization: each node's outgoing edges sum to 1
        """        
        # Row-normalize: divide each edge weight by sum of weights in the source node's row
        row, _ = self.edge_indices
        deg = torch.zeros(self.num_nodes, dtype=self.edge_weights.dtype, device=self.edge_weights.device)
        deg.scatter_add_(0, row, self.edge_weights)
        
        # avoid division by zero
        deg_inv = torch.zeros_like(deg)
        nonzero_mask = deg > 0
        deg_inv[nonzero_mask] = 1.0 / deg[nonzero_mask]
        
        edge_weights = self.edge_weights * deg_inv[row]    
        return edge_weights