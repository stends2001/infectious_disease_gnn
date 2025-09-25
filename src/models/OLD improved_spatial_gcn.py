import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, ChebConv, GINConv
import pandas as pd
import numpy as np
from typing import Optional, Tuple

from ..dataloading.gnndataloader import GNNDataLoader
from .modelcore import DeepLearningModelCore


class ImprovedSpatialGCN(nn.Module):
    """
    Improved Spatial Graph Convolutional Network with better graph utilization:
    - Multiple GCN layers with residual connections
    - Layer normalization
    - Graph attention mechanisms
    - Better feature processing
    """
    def __init__(self, 
                 node_features: int, 
                 hidden_size: int = 64, 
                 num_layers: int = 4, 
                 dropout: float = 0.3,
                 use_attention: bool = True,
                 use_residual: bool = True):
        super(ImprovedSpatialGCN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_size)
        
        # Build GCN layers with residual connections
        self.gcn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # First layer
        self.gcn_layers.append(GCNConv(hidden_size, hidden_size))
        self.layer_norms.append(nn.LayerNorm(hidden_size))
        self.dropouts.append(nn.Dropout(dropout))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_size, hidden_size))
            self.layer_norms.append(nn.LayerNorm(hidden_size))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Graph Attention Layer (optional)
        if use_attention:
            self.graph_attention = GATConv(hidden_size, hidden_size // 4, heads=4, dropout=dropout)
            self.attention_norm = nn.LayerNorm(hidden_size)
        
        # Output projection with residual connection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Skip connection for output
        if use_residual:
            self.skip_proj = nn.Linear(node_features, 1)

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
        
        # Store original input for skip connection
        original_input = x_t
        
        # Input projection
        h = self.input_proj(x_t)  # [num_nodes, hidden_size]
        
        # Apply GCN layers with residual connections
        for i, (conv, norm, dropout) in enumerate(zip(self.gcn_layers, self.layer_norms, self.dropouts)):
            # Graph convolution
            h_new = conv(h, edge_index, edge_weight)
            h_new = F.relu(h_new)
            h_new = norm(h_new)
            h_new = dropout(h_new)
            
            # Residual connection (except first layer)
            if self.use_residual and i > 0:
                h = h + h_new
            else:
                h = h_new
        
        # Optional graph attention
        if self.use_attention:
            h_att = self.graph_attention(h, edge_index)
            h_att = F.relu(h_att)
            h_att = self.attention_norm(h_att)
            h = h + h_att  # Residual connection
        
        # Output projection
        output = self.output_proj(h)  # [num_nodes, 1]
        
        # Skip connection from original input
        if self.use_residual:
            skip_output = self.skip_proj(original_input)  # [num_nodes, 1]
            output = output + skip_output
        
        return output.squeeze(-1)  # [num_nodes]


class ImprovedSpatialGCNModel(DeepLearningModelCore):
    """
    Improved Spatial GCN model with better graph utilization.
    """
    def __init__(self, 
                 dataloader: GNNDataLoader, 
                 name: Optional[str] = None):
        super().__init__(dataloader, name=name)
        if not self.name:
            self.name = 'ImprovedSpatialGCN'

        self.model_color = '#4ECDC4'
        self.dataloader = dataloader

    def set_model_hparams(self, 
                          hidden_size: int = 64, 
                          num_layers: int = 4, 
                          dropout: float = 0.3,
                          use_attention: bool = True,
                          use_residual: bool = True):
        self.model_hparams_set = True
        self.model = ImprovedSpatialGCN(
            node_features=len(self.dataloader.feature_columns),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention,
            use_residual=use_residual
        ).to(self.device)

        return self
