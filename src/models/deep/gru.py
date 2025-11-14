import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from typing import Optional, Tuple
from torch_geometric.nn import GCNConv, GATConv, ChebConv, GINConv, GATv2Conv
from torch_geometric_temporal.nn.recurrent import DCRNN, TGCN, A3TGCN
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from ...dataloading.deepdataloader import DeepDataLoader
from .deepmodel import DeepModel
from .strategies.recurrent_strategy import RecurrentStrategy
from torch_geometric.utils import add_self_loops

class GRUModule(nn.Module):
    """
    GRU-based spatiotemporal model that processes time sequences using GRU units.
    """
    def __init__(self,
                 node_features: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 prediction_horizon: int,
                 self_loops: bool):
        super().__init__()

        self.node_features = node_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        self.self_loops = self_loops

        # === Temporal GRU layer ===
        self.temporal_gru = nn.GRU(
            input_size=node_features,  # GRU expects input of shape (batch_size, seq_len, features)
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True  # Input format will be (batch, seq, feature)
        )

        # === Output layer ===#
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, prediction_horizon)
        )

        self.output_scale = nn.Parameter(torch.tensor(3.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x_t: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None,                
                hidden_state: Optional[torch.Tensor] = None,
                debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one time step.

        x_t: [batch_size, num_nodes, node_features]
        hidden_state: [num_layers, batch_size, hidden_size]
        """
        if debug:
            print("input shape:", x_t.shape)

        # Add time dim: (batch_size, seq_len, features)
        # x_t = x_t.unsqueeze(1)  # Reshape to (batch_size, 1, num_nodes, node_features)
        h = x_t.squeeze(-1)

        if hidden_state is None:
            gru_out, new_hidden = self.temporal_gru(h)
        else:
            gru_out, new_hidden = self.temporal_gru(h, hidden_state)

        # Use the output of the last time step (we only have 1 timestep)
        last_hidden = gru_out  # [batch_size, num_nodes, hidden_size]

        # Project to output space
        output = self.output_proj(last_hidden)
        output = output * self.output_scale

        return output, new_hidden

class GRUModel(DeepModel):
    """
    Wrapper to manage the model and training strategy.
    """
    def __init__(self, 
                 dataloader: DeepDataLoader, 
                 name: Optional[str] = None):
        super().__init__(dataloader, name=name)
        
        if not self.name:
            self.name = 'GRU'

        self.dataloader = dataloader
        

        self._set_strategy(RecurrentStrategy())

    def set_model_hparams(self,
                          hidden_size: int = 64,
                          num_layers: int = 2,
                          dropout: float = 0.2,
                          self_loops: bool = False):
        # Set up model with the given hyperparameters
        self.model = GRUModule(
            node_features=len(self.dataloader.feature_columns),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            prediction_horizon=self.dataloader.horizon_size,
            self_loops=self_loops
        ).to(self.device)

        model_hparams_config = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'self_loops': self_loops
        }

        self.config_info = {'model_hparams': model_hparams_config}
        self._state = {'model_initialized': True}

        return self
