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


class ImprovedTemporalGCN(nn.Module):
    """
    Improved Temporal Graph Convolutional Network with better graph utilization:
    - Residual connections
    - Layer normalization
    - Skip connections
    - Better temporal modeling
    - Graph attention mechanisms
    """
    def __init__(self, node_features: int, hidden_size: int = 64, num_layers: int = 3, 
                 dropout: float = 0.3, temporal_layers: int = 2, prediction_horizon: int = 1,
                 use_attention: bool = True, use_residual: bool = True):
        super(ImprovedTemporalGCN, self).__init__()
        
        self.node_features = node_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.temporal_layers = temporal_layers
        self.prediction_horizon = prediction_horizon
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Input projection to hidden size
        self.input_proj = nn.Linear(node_features, hidden_size)
        
        # Spatial GCN layers with residual connections
        self.spatial_convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # First layer
        self.spatial_convs.append(GCNConv(hidden_size, hidden_size))
        self.layer_norms.append(nn.LayerNorm(hidden_size))
        self.dropouts.append(nn.Dropout(dropout))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.spatial_convs.append(GCNConv(hidden_size, hidden_size))
            self.layer_norms.append(nn.LayerNorm(hidden_size))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Graph Attention Layer (optional)
        if use_attention:
            self.graph_attention = GATConv(hidden_size, hidden_size // 4, heads=4, dropout=dropout)
            self.attention_norm = nn.LayerNorm(hidden_size)
        
        # Temporal LSTM with residual connections
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=temporal_layers,
            dropout=dropout if temporal_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Temporal attention (optional)
        if use_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        
        # Output projection with residual connection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, prediction_horizon)
        )
        
        # Skip connection for output
        if use_residual:
            self.skip_proj = nn.Linear(node_features, prediction_horizon)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [num_nodes, node_features, time_steps]
        edge_index: [2, num_edges]
        edge_weight: [num_edges] (optional)
        """
        num_nodes, node_features, time_steps = x.shape
        
        # Store original input for skip connection
        original_input = x[:, :, -1]  # Last time step
        
        # Process each time step through spatial GCN
        spatial_outputs = []
        
        for t in range(time_steps):
            # Get features for current time step
            x_t = x[:, :, t]  # [num_nodes, node_features]
            
            # Input projection
            h = self.input_proj(x_t)  # [num_nodes, hidden_size]
            
            # Apply spatial GCN layers with residual connections
            for i, (conv, norm, dropout) in enumerate(zip(self.spatial_convs, self.layer_norms, self.dropouts)):
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
            
            spatial_outputs.append(h)
        
        # Stack spatial outputs along time dimension
        spatial_seq = torch.stack(spatial_outputs, dim=1)  # [num_nodes, time_steps, hidden_size]
        
        # Apply temporal LSTM
        lstm_out, _ = self.temporal_lstm(spatial_seq)  # [num_nodes, time_steps, hidden_size]
        
        # Optional temporal attention
        if self.use_attention:
            attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
            lstm_out = lstm_out + attn_out  # Residual connection
        
        # Use the last hidden state for predictions
        last_hidden = lstm_out[:, -1, :]  # [num_nodes, hidden_size]
        output = self.output_proj(last_hidden)  # [num_nodes, prediction_horizon]
        
        # Skip connection from original input
        if self.use_residual:
            skip_output = self.skip_proj(original_input)  # [num_nodes, prediction_horizon]
            output = output + skip_output
        
        if self.prediction_horizon == 1:
            return output.squeeze(-1)  # [num_nodes]
        else:
            return output  # [num_nodes, prediction_horizon]


class ImprovedTemporalGCNModel(DeepLearningModelCore):
    """
    Improved Temporal GCN model with better graph utilization.
    """
    def __init__(self, dataloader: GNNDataLoader, name: Optional[str] = None):
        super().__init__(dataloader, name=name)
        if not self.name:
            self.name = 'ImprovedTemporalGCN'
        
        self.model_color = '#FF6B6B'
        self.dataloader = dataloader
    
    def set_model_hparams(self, hidden_size: int = 64, num_layers: int = 3,
                         temporal_layers: int = 2, dropout: float = 0.3,
                         use_attention: bool = True, use_residual: bool = True):
        self.model_hparams_set = True
        self.model = ImprovedTemporalGCN(
            node_features=len(self.dataloader.feature_columns),
            hidden_size=hidden_size,
            num_layers=num_layers,
            temporal_layers=temporal_layers,
            dropout=dropout,
            prediction_horizon=self.dataloader.prediction_horizon,
            use_attention=use_attention,
            use_residual=use_residual
        ).to(self.device)
        
        return self
