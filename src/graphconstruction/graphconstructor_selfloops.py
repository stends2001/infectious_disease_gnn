from typing import Optional, Tuple, List, Union, Literal, Dict
from numpy.typing import NDArray
from statistics import mean

class GraphAddSelfLoops:
    def __init__(self,
                 edge_indices: List[Tuple[int,int]],
                 edge_weights: List[float],
                 num_nodes   : int,
                 node_ids    : NDArray):
        
        self.edge_weights_ls = edge_weights 
        self.edge_indices_ls = edge_indices
        self.num_nodes       = num_nodes
        self.node_ids        = node_ids


        self.SELFLOOP_FUNCS = {
                                    '0'     : self._add0,
                                    'max'   : self._addmax,
                                    'mean'  : self._addmean,
        }    

    def add_loops(self, method: str) -> Tuple[List[Tuple[int,int]], List[float]]:

        if method not in self.SELFLOOP_FUNCS:
            raise ValueError(f"Unknown selfloop addition method: {method}")

        # add the indices
        self_loops              = [(nid, nid) for nid in self.node_ids]   
        updated_indices         = self.edge_indices_ls + self_loops

        # add the weights
        updated_weights        = self.SELFLOOP_FUNCS[method]() 

        return (updated_indices, updated_weights)
    
                
    def _add0(self):
        return self.edge_weights_ls + [0 for _ in self.node_ids]

    def _addmax(self):
        max_weight = max(self.edge_weights_ls) if self.edge_weights_ls else 1
        return self.edge_weights_ls + [max_weight for _ in self.node_ids]        

    def _addmean(self):
        mean_weight = mean(self.edge_weights_ls)
        return self.edge_weights_ls + [mean_weight for _ in self.node_ids]        
       