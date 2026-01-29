from typing import TYPE_CHECKING, Optional, Tuple 
import torch 
import numpy as np 
import pandas as pd

if TYPE_CHECKING:
    from .airportrawgraphbuilder import GraphConnectionsDataFrame


class GraphProcessingError(Exception):
    def __init__(self, explanation: str):
        statement = "GraphConnectionsDataFrame couldnt be processed" + "\n" + explanation
        super().__init__(statement)


class AirportRawGraphProcessor():

    def __init__(self, connections_dataframe: 'GraphConnectionsDataFrame', normalization_method: Optional[str] = None):
        self.normalization_methods = ['method1?']

        self.connections_dataframe = connections_dataframe
        self.normalization_method  = normalization_method

    def process(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.cd             = self.connections_dataframe
        self.filtered_cd    = self._filter_zeroes()
        self.normalized_cd  = self._normalize_weights(self.normalization_method)        
        return self._tensorize()

    def _filter_zeroes(self) -> pd.DataFrame:
        df = self.cd.df_copy
        return df[df['weight']>0].reset_index()
    
    def _normalize_weights(self, method) -> pd.DataFrame:
        if not method:
            return self.filtered_cd

        elif method and method not in self.normalization_methods:
            raise GraphProcessingError(f'normalization method {method} not found. Please supply any of {self.normalization_methods}')
        
        else:
            print('currently not implemented')
        
    def _tensorize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """converts connctions_dataframe into an edge_index-tensor and an edge_weight-tensor"""
        df = self.normalized_cd
        edge_index  = np.stack(arrays =[df['node_layer2'], df['node_layer3']], axis=0) 
        edge_weights= df['weight']      
        return torch.tensor(edge_index, dtype = torch.long),  torch.tensor(edge_weights, dtype = torch.float)      
