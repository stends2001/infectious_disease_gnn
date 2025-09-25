from ._deepmodel import DeepModel

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, ChebConv, GINConv
from torch_geometric_temporal.nn.recurrent import DCRNN, TGCN, A3TGCN
import pandas as pd
import numpy as np
from typing import Optional, Tuple

from ..dataloading.gnndataloader import GNNDataLoader


class SpatialGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_size, out_size):
        super().__init__()
        # 1st GCN layer: transforms input features -> hidden_dim
        self.conv1 = GCNConv(node_features, hidden_size)

        # 2nd GCN layer: hidden_dim -> hidden_dim (can be stacked)
        self.conv2 = GCNConv(hidden_size, hidden_size)

        # Final linear layer for prediction
        self.linear = torch.nn.Linear(hidden_size, out_size)

    def forward(self, x, edge_index, edge_weight, debug=False):

        if x.dim() == 3:
                if debug:
                    print("⚠️  Input features are 3D (e.g., sequences). Only the last dimension will be used (flattened).")
                x = x[:, :, -1]  # Flatten seq_len * feature_dim
        elif x.dim() != 2:
            raise ValueError(f"Expected 2D or 3D input for node features, got shape {x.shape}")

        # x: node features matrix (num_nodes x feature_dim)
        # edge_index: graph connectivity (2 x num_edges)

        # 1) First GCN layer + activation
        x1 = self.conv1(x, edge_index, edge_weight)
        x1 = F.relu(x1)

        # 2) Second GCN layer + activation
        x2 = self.conv2(x1, edge_index, edge_weight)
        x2 = F.relu(x2)

        # 3) Final prediction layer (no activation, regression)
        out = self.linear(x2)

        return out.squeeze(-1)


class SpatialGCNModel(DeepModel):
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
                          hidden_size: int = 256):
        self.model_hparams_set = True
        self.model = SpatialGNN(
            node_features=len(self.gnn_dataloader.feature_columns),
            hidden_size=hidden_size,
            out_size = self.gnn_dataloader.prediction_horizon
        ).to(self.device)

        return self
