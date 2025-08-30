
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, ChebConv, GINConv
from torch_geometric_temporal.nn.recurrent import DCRNN, TGCN, A3TGCN
import pandas as pd
import numpy as np
from typing import Optional, Tuple

from ..dataloading.gnndataloader import GNNDataLoader
from .modelcore import DeepLearningModelCore

class SpatialGCN(nn.Module):
    """
    Spatial Graph Convolutional Network that only uses graph structure
    """
    def __init__(self, 
                 node_features: int, 
                 hidden_size:   int = 64, 
                 num_layers:    int = 2, 
                 dropout: float     = 0.2):
        super(SpatialGCN, self).__init__()
        
        self.hidden_size = hidden_size
        self.dropout     = nn.Dropout(dropout)

        # Build GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(node_features, hidden_size))
        
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_size, hidden_size))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [num_nodes, node_features, time_steps]
        edge_index: [2, num_edges]
        edge_weight: [num_edges] (optional)
        """
        # Use only the most recent time step
        x_t = x[:, :, -1]  # [num_nodes, node_features]

        h = x_t
        for conv in self.gcn_layers:
            h = conv(h, edge_index, edge_weight)
            h = F.relu(h)
            h = self.dropout(h)

        output = self.output_proj(h)  # [num_nodes, 1]
        return output.squeeze(-1)     # [num_nodes]


class SpatialGCNModel(DeepLearningModelCore):
    """
    Purely spatial GCN model that does nto use temporal axis into account.
    Useful to validate the use of graph-structure
    """
    def __init__(self, 
                 dataloader: GNNDataLoader, 
                 name: Optional[str] = None):
        super().__init__(dataloader, name=name)
        if not self.name:
            self.name = 'SpatialGCN'

        self.model_color = '#4ECDC4'
        self.dataloader = dataloader

    def set_model_hparams(self, 
                          hidden_size: int = 64, 
                          num_layers: int = 2, 
                          dropout: float = 0.2):
        self.model_hparams_set = True
        self.model = SpatialGCN(
            node_features=len(self.dataloader.feature_columns),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)

        return self
