from ._deepmodel import DeepModel

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.nn as nn

import pandas as pd
import numpy as np
from typing import Optional, Tuple

from ..dataloading.gnndataloader import GNNDataLoader
from torch_geometric_temporal.nn.recurrent import A3TGCN2


class A3TGCN2Module(torch.nn.Module):
    def __init__(self, node_features, hidden_size, periods, horizon, self_loops):
        super(A3TGCN2Module, self).__init__()

        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features,  out_channels=hidden_size, periods=periods, batch_size = 1, add_self_loops=self_loops) # node_features=2, periods=12
        
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(hidden_size, horizon)

    def forward(self, x, edge_index, edge_weight, debug=False):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        if debug:
            print('shape x:', x.shape)

        x = x.unsqueeze(0)        

        if debug:
            print('shape unsqueezed x:', x.shape)

        h = self.tgnn(x, edge_index, edge_weight)
        if debug:
            print('shape h:', h.shape)        
        h = F.relu(h) 
        h = self.linear(h)

        h = h.squeeze(0)
        if debug:
            print('shape output:', h.shape)
        
        return h
    
class A3TGCNModel(DeepModel):
    """
    A3TGCN-based GNN model
    """
    def __init__(self, 
                 dataloader: GNNDataLoader, 
                 name: Optional[str] = None):
        super().__init__(dataloader, name=name)
        if not self.name:
            self.name = 'A3TGCN'

        self.model_color = "#3E6BCD"
        self.dataloader = dataloader

        self.config_info['model'] = 'a3tgcnmodel'

    def set_model_hparams(self, 
                          hidden_size: int = 32,
                          self_loops: bool = True) -> 'A3TGCNModel':
        """
        initializes a3tgcn model

        Parameters:
        ----------
        hidden_size: int = 32

        self_loops: bool = True
            Whether or not to add self loops, model-internally
        """
        self.model = A3TGCN2Module(
            node_features= len(self.gnn_dataloader.feature_columns),
            hidden_size  = hidden_size,
            periods      = self.gnn_dataloader.periods,
            horizon      = self.prediction_horizon,
            self_loops   = self_loops
        ).to(self.device)

        model_hparams_config = {'hidden_size': hidden_size,
                                'self_loops' : self_loops}

        self.config_info['model_hparams'] = model_hparams_config

        self._state['model_initialized'] = True

        return self    