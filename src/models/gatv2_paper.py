import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Literal
from torch_geometric.nn import GATv2Conv

from ..dataloading.gnndataloader import GNNDataLoader
from .modelcore import DeepLearningModelCore


class PaperGATv2Layer(nn.Module):
    """
    Simple GATv2 layer without skip connections to avoid dimension issues.
    Matches the working GATv2 approach.
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, 
                 dropout: float = 0.0, concat: bool = True, use_edge_weights: bool = True):
        super(PaperGATv2Layer, self).__init__()
        
        self.use_edge_weights = use_edge_weights
        
        self.gatv2 = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
            edge_dim=1 if use_edge_weights else None,
            share_weights=True
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [num_nodes, in_channels]
            edge_index: [2, num_edges]
            edge_weight: [num_edges] (optional)
        
        Returns:
            output: [num_nodes, out_channels * heads]
        """
        # GATv2 forward pass
        if self.use_edge_weights and edge_weight is not None:
            return self.gatv2(x, edge_index, edge_weight)
        else:
            return self.gatv2(x, edge_index)


class PaperGRU_GATv2_Model(nn.Module):
    """
    Paper's exact architecture: 2 GRU layers + 2 GATv2 layers with skip connections.
    """
    def __init__(self, 
                 node_features: int,
                 gru_hidden_size: int = 32,
                 gatv2_hidden_size: int = 64,
                 gatv2_heads: int = 2,
                 dropout: float = 0.2,
                 use_edge_weights: bool = True):
        super(PaperGRU_GATv2_Model, self).__init__()
        
        self.node_features = node_features
        self.gru_hidden_size = gru_hidden_size
        self.gatv2_hidden_size = gatv2_hidden_size
        self.gatv2_heads = gatv2_heads
        
        # Two GRU layers (as in paper)
        self.gru1 = nn.GRU(
            input_size=node_features,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0  # No dropout between layers since we have only 1 layer each
        )
        
        self.gru2 = nn.GRU(
            input_size=gru_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        # Two GATv2 layers (as in paper, but without skip connections to avoid dimension issues)
        self.gatv2_1 = PaperGATv2Layer(
            in_channels=gru_hidden_size,
            out_channels=gatv2_hidden_size // gatv2_heads,
            heads=gatv2_heads,
            dropout=dropout,
            concat=True,
            use_edge_weights=use_edge_weights
        )
        
        # Second GATv2 layer
        self.gatv2_2 = PaperGATv2Layer(
            in_channels=gatv2_hidden_size,  # Output of first GATv2 layer
            out_channels=gatv2_hidden_size // gatv2_heads,
            heads=gatv2_heads,
            dropout=dropout,
            concat=True,
            use_edge_weights=use_edge_weights
        )
        
        # Output layer
        self.output_layer = nn.Linear(gatv2_hidden_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass matching paper's architecture.
        
        Args:
            x: [num_nodes, node_features, time_steps] - matches working GATv2 input format
            edge_index: [2, num_edges]
            edge_weight: [num_edges] (optional)
        
        Returns:
            output: [num_nodes]
        """
        num_nodes, node_features, time_steps = x.shape
        
        # Convert to [num_nodes, time_steps, node_features] for GRU processing
        # This matches the working GATv2 approach
        x_reshaped = x.permute(0, 2, 1)  # [num_nodes, time_steps, node_features]
        
        # Process temporal dimension for each node through GRU layers
        # First GRU layer
        gru1_out, _ = self.gru1(x_reshaped)  # [num_nodes, time_steps, gru_hidden_size]
        
        # Second GRU layer  
        gru2_out, _ = self.gru2(gru1_out)  # [num_nodes, time_steps, gru_hidden_size]
        
        # Use the last time step for spatial processing (as in paper)
        h_gru = gru2_out[:, -1, :]  # [num_nodes, gru_hidden_size]
        
        # First GATv2 layer
        h_gatv2_1 = self.gatv2_1(h_gru, edge_index, edge_weight)  # [num_nodes, gatv2_hidden_size]
        
        # Second GATv2 layer
        h_gatv2_2 = self.gatv2_2(h_gatv2_1, edge_index, edge_weight)  # [num_nodes, gatv2_hidden_size]
        
        # Output layer
        output = self.output_layer(h_gatv2_2)  # [num_nodes, 1]
        
        return output.squeeze(-1)  # [num_nodes]


class PaperGRU_GATv2_Model_Wrapper(DeepLearningModelCore):
    """
    Wrapper for the paper's GRU-GATv2 model.
    """
    def __init__(self, dataloader, name: Optional[str] = None):
        super().__init__(dataloader, name=name)
        if not self.name:
            self.name = 'Paper_GRU_GATv2'
        
        self.model_color = '#FF6B35'
        self.dataloader = dataloader
    
    def set_model_hparams(self, 
                         gru_hidden_size: int = 32,
                         gatv2_hidden_size: int = 64,
                         gatv2_heads: int = 2,
                         dropout: float = 0.2,
                         use_edge_weights: bool = True,
                         **kwargs):
        self.model_hparams_set = True
        self.model = PaperGRU_GATv2_Model(
            node_features=len(self.dataloader.feature_columns),
            gru_hidden_size=gru_hidden_size,
            gatv2_hidden_size=gatv2_hidden_size,
            gatv2_heads=gatv2_heads,
            dropout=dropout,
            use_edge_weights=use_edge_weights
        ).to(self.device)
    