
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


class TemporalGCN(nn.Module):
    """
    Temporal Graph Convolutional Network that properly uses graph structure
    across time steps for infectious disease forecasting.
    """
    def __init__(self, node_features: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, temporal_layers: int = 2, prediction_horizon: int = 1):
        super(TemporalGCN, self).__init__()
        
        self.node_features = node_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.temporal_layers = temporal_layers
        self.prediction_horizon = prediction_horizon
        
        # Spatial GCN layers
        self.spatial_convs = nn.ModuleList()
        self.spatial_convs.append(GCNConv(node_features, hidden_size))
        
        for _ in range(num_layers - 1):
            self.spatial_convs.append(GCNConv(hidden_size, hidden_size))
        
        # Temporal LSTM layers
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=temporal_layers,
            dropout=dropout if temporal_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, prediction_horizon)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [num_nodes, node_features, time_steps]
        edge_index: [2, num_edges]
        edge_weight: [num_edges] (optional)
        """
        num_nodes, node_features, time_steps = x.shape
        
        # Process each time step through spatial GCN
        spatial_outputs = []
        
        for t in range(time_steps):
            # Get features for current time step
            x_t = x[:, :, t]  # [num_nodes, node_features]
            
            # Apply spatial GCN layers
            h = x_t
            for conv in self.spatial_convs:
                h = conv(h, edge_index, edge_weight)
                h = F.relu(h)
                h = self.dropout(h)
            
            spatial_outputs.append(h)
        
        # Stack spatial outputs along time dimension
        spatial_seq = torch.stack(spatial_outputs, dim=1)  # [num_nodes, time_steps, hidden_size]
        
        # Apply temporal LSTM
        lstm_out, _ = self.temporal_lstm(spatial_seq)  # [num_nodes, time_steps, hidden_size]
        
        # Always use the last hidden state for predictions
        last_hidden = lstm_out[:, -1, :]  # [num_nodes, hidden_size]
        output = self.output_proj(last_hidden)  # [num_nodes, prediction_horizon]

        if self.prediction_horizon == 1:
            return output.squeeze(-1)  # [num_nodes]
        else:
            return output  # [num_nodes, prediction_horizon]


class TemporalGCNModel(DeepLearningModelCore):
    """
    Temporal GCN model for infectious disease forecasting.
    """
    def __init__(self, dataloader: GNNDataLoader, name: Optional[str] = None):
        super().__init__(dataloader, name=name)
        if not self.name:
            self.name = 'TemporalGCN'
        
        self.model_color = '#1F77B4'
        self.dataloader = dataloader
    
    def set_model_hparams(self, hidden_size: int = 64, num_layers: int = 2,
                         temporal_layers: int = 2, dropout: float = 0.2):
        self.model_hparams_set = True
        self.model = TemporalGCN(
            node_features=len(self.dataloader.feature_columns),
            hidden_size=hidden_size,
            num_layers=num_layers,
            temporal_layers=temporal_layers,
            dropout=dropout,
            prediction_horizon= self.dataloader.prediction_horizon
        ).to(self.device)
        
        return self
