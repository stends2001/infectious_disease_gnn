import torch
from typing import Literal, assert_never

from ..graphobjects import GraphStructure, TopKConfig
from ...utils import registry_method, get_registered_methods, MethodNotInRegistry
from ...utils.types import GraphNormType

class GraphPostProcessor:
    """    
    """
    def __init__(self):
        self.methods     = get_registered_methods(self.__class__)

    def filter_top_k(self, 
                     graph_structure: GraphStructure, 
                     mode: Literal['local','global'],
                     k: int
                     ) -> GraphStructure:
        """ 
        """
        # unpacking topk_cfg:


        edge_index   = graph_structure.edge_index
        edge_weight = graph_structure.edge_weight

        match mode:

            case "global":
                keep = edge_weight.topk(k).indices

            case "local":
                keep = []

                for node in range(graph_structure.num_nodes):
                    mask = edge_index[0] == node
                    node_edges = mask.nonzero(as_tuple=True)[0]

                    if len(node_edges) <= k:
                        keep.append(node_edges)
                    else:
                        top = edge_weight[node_edges].topk(k).indices
                        keep.append(node_edges[top])

                keep = torch.cat(keep)

            case _:
                assert_never(mode)

        filtered_edge_index = edge_index[:, keep]
        filtered_edge_weight = edge_weight[keep]

        filtered_graphstructure = GraphStructure(
            filtered_edge_index,
            filtered_edge_weight,
            graph_structure.num_nodes,
        )

        return filtered_graphstructure

    def normalize(self, graph_structure: GraphStructure, method: GraphNormType, *args, **kwargs) -> GraphStructure:
        """ 
        Normalizes edge-weights according to method. Specific methods may require *args or **kwargs
        Returns a GraphStructure with normalized edge-weight and unchanged edge-index.        
        """        
        if method not in self.methods:
            raise MethodNotInRegistry(method, list(self.methods))
    
        normalized_weights: torch.Tensor = getattr(self, method)(graph_structure, *args, **kwargs)

        normalized_graph_structure = GraphStructure(
            graph_structure.edge_index,
            normalized_weights,
            graph_structure.num_nodes,
        )

        return normalized_graph_structure

    @registry_method
    def minmax(self, graph_structure: GraphStructure, epsilon: float = 1e-3) -> torch.Tensor:
        """
        scales edge-weights by minmax (i.e. max => 1 and min => 0 + `epsilon`)
        """
        raw_weights = graph_structure.edge_weight

        min_w = raw_weights.min().item()
        max_w = raw_weights.max().item()    

        if max_w > min_w:
            edge_weights = (epsilon + raw_weights - min_w) / (max_w - min_w)
        else:
            edge_weights = torch.zeros_like(raw_weights)    

        return edge_weights       

    @registry_method
    def symmetric(self, graph_structure: GraphStructure) -> torch.Tensor:
        """
        Applies symmetric normalization to edge weights.
        NOTE This should be the default normalization method

        This computes the normalized edge weights according to:

            w_ij' = w_ij / sqrt(d_i * d_j)

        where:
            - w_ij is the original edge weight between node i -> j
            - d_i is the (weighted) out-degree of node i:
                d_i = sum_j w_ij

        In matrix form, this corresponds to:

            D^{-1/2} A D^{-1/2}

        where:
            - A is the weighted adjacency matrix
            - D is the diagonal degree matrix
        """
        indices     = graph_structure.edge_index
        raw_weights = graph_structure.edge_weight
        row, col    = indices[0], indices[1]

        deg = torch.zeros(graph_structure.num_nodes, dtype=raw_weights.dtype, device=raw_weights.device)
        deg.scatter_add_(0, row, raw_weights)

        deg_inv_sqrt                               = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_weight = raw_weights * deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return edge_weight

    @registry_method
    def rowwise(self, graph_structure: GraphStructure) -> torch.Tensor:
        """
        Applies row-wise normalization: D^(-1) * W
        Each node's outgoing edge weights sum to 1.
        """
        indices     = graph_structure.edge_index
        raw_weights = graph_structure.edge_weight
        row, col    = indices[0], indices[1]

        deg = deg = torch.zeros(graph_structure.num_nodes).to(raw_weights)
        deg.scatter_add_(0, row, raw_weights)

        deg_inv                          = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0

        edge_weight = raw_weights * deg_inv[row]
        return edge_weight
    
    def __repr__(self) -> str:
        representation = f'<{self.__class__.__name__}>'
        return representation