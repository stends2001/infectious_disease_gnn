from typing import TYPE_CHECKING, Optional, Tuple 
import torch 
import numpy as np 
import pandas as pd

if TYPE_CHECKING:
    from .airportrawgraphbuilder import GraphConnectionsDataFrame

from .graphstructure import GraphStructure


class GraphProcessingError(Exception):
    def __init__(self, explanation: str):
        statement = "GraphConnectionsDataFrame couldnt be processed" + "\n" + explanation
        super().__init__(statement)

class AirportRawGraphProcessor():

    def __init__(self, connections_dataframe: 'GraphConnectionsDataFrame', normalization_method: Optional[str] = None):
        self.normalization_methods = ['method1?']

        self.connections_dataframe = connections_dataframe
        self.normalization_method  = normalization_method

    def process(self) -> 'GraphStructure':
        self.cd             = self.connections_dataframe
        self.filtered_cd    = self._filter_zeroes(self.cd)
        self.normalized_cd  = self._normalize_weights(self.normalization_method, self.filtered_cd)        
        self.tensors        = self._tensorize(self.normalized_cd)
        self.graphstructure = GraphStructure(self.tensors[0], self.tensors[1])
        return self.graphstructure

    def _filter_zeroes(self, connections_df: 'GraphConnectionsDataFrame') -> pd.DataFrame:
        df = connections_df.df_copy
        return df[df['weight']>0].reset_index()
    
    def _normalize_weights(self, method, connections_df: pd.DataFrame) -> pd.DataFrame:
        df = connections_df.copy()
        if not method:
            return df

        elif method and method not in self.normalization_methods:
            raise GraphProcessingError(f'normalization method {method} not found. Please supply any of {self.normalization_methods}')
        
        else:
            print('currently not implemented')
        
        return df
        
    def _tensorize(self, connections_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """converts connctions_dataframe into an edge_index-tensor and an edge_weight-tensor"""
        df = connections_df.copy()
        edge_index  = np.stack(arrays =[df['node_layer2'], df['node_layer3']], axis=0) 
        edge_weights= df['weight']      
        return torch.tensor(edge_index, dtype = torch.long),  torch.tensor(edge_weights, dtype = torch.float)      
