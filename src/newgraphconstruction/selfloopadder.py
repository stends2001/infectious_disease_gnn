# SelfLoopAdder
from typing import Optional, Tuple, List, Union, Literal, Dict
import numpy as np
from statistics import mean

from .containers import RawGraphStructure

class SelfLoopAdder:
    """ 
    adds self-loops to edge_indices and edge_weights
    Called in the process of GraphConstructor

    Parameters
    ----------
    edge_indices: List[Tuple[int,int]]
    edge_weights: List[float]
    node_ids: NDArray
        an array of node ids to loop over

    See also
    --------
    GraphConstructor    → creates and returns an edge_index and a edge_weight
    GraphOrchestrator   → orchestrates the creation of a graph through GraphConstructor
    """    
    def __init__(self,
                 rawgraphstructure: RawGraphStructure,
                 node_ids    : np.ndarray):
        
        self.edge_indices_ls =rawgraphstructure.edge_index
        self.edge_weights_ls =rawgraphstructure.edge_weight
        self.node_ids        = node_ids


        self.SELFLOOP_FUNCS = {
                                    '0'     : self._add0,
                                    'max'   : self._addmax,
                                    'mean'  : self._addmean,
        }    

    def add_loops(self, method: str) -> Tuple[List[Tuple[int,int]], List[float]]:
        """
        collects the required loop-adding-function and feeds in the kwargs
        """
        if method not in self.SELFLOOP_FUNCS:
            raise ValueError(f"Unknown selfloop addition method: {method}. Available methods are: {', '.join(self.SELFLOOP_FUNCS.keys())}")      

        # add the indices
        self_loops              = [(nid, nid) for nid in self.node_ids]   
        updated_indices         = self.edge_indices_ls + self_loops

        # add the weights
        updated_weights        = self.SELFLOOP_FUNCS[method]() 

        return RawGraphStructure(updated_indices, updated_weights)
                  
    def _add0(self):
        return self.edge_weights_ls + [0 for _ in self.node_ids]

    def _addmax(self):
        max_weight = max(self.edge_weights_ls) if self.edge_weights_ls else 1
        return self.edge_weights_ls + [max_weight for _ in self.node_ids]        

    def _addmean(self):
        mean_weight = mean(self.edge_weights_ls)
        return self.edge_weights_ls + [mean_weight for _ in self.node_ids]        