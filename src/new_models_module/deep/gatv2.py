import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, ChebConv, GINConv, GATv2Conv
from torch_geometric_temporal.nn.recurrent import DCRNN, TGCN, A3TGCN
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from ...dataloading.gnndataloader import GNNDataLoader
from .deepmodel import DeepModel
from ...utils.constants import paired_colors
from .strategies.recurrent_strategy import RecurrentStrategy

class GATv2Module(nn.Module):
    """
    Recurrent GATv2-based spatiotemporal model.
    Processes a single time step at a time and keeps hidden state between steps.
    """
    def __init__(self,
                 node_features: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 temporal_layers: int = 1,
                 prediction_horizon: int = 1,
                 heads: int = 2):
        super().__init__()

        self.node_features = node_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.temporal_layers = temporal_layers
        self.prediction_horizon = prediction_horizon

        # === Spatial GATv2 layers ===
        self.spatial_convs = nn.ModuleList()
        self.spatial_convs.append(GATv2Conv(node_features, hidden_size, heads=heads, dropout=dropout))

        for _ in range(num_layers - 1):
            self.spatial_convs.append(
                GATv2Conv(hidden_size * heads, hidden_size, heads=heads, dropout=dropout)
            )

        self.total_gat_out_dim = hidden_size * heads

        # === Temporal LSTM ===
        self.temporal_rnn = nn.LSTM(
            input_size=self.total_gat_out_dim,
            hidden_size=hidden_size,
            num_layers=temporal_layers,
            dropout=dropout if temporal_layers > 1 else 0,
        )

        # === Output layer ===
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
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                debug: bool = False
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for one time step.

        x_t: [num_nodes, node_features]
        edge_index: [2, num_edges]
        hidden_state: Tuple (h, c), each [num_layers, num_nodes, hidden_size]
        """
        # --- 1. Spatial GCN processing ---
        if debug:
            print("input shape:", x_t.shape)

        h = x_t.squeeze(-1)

        for gat in self.spatial_convs:
            h = gat(h, edge_index)
            h = F.elu(h)
            h = self.dropout(h)

        # Add time dim: [1, num_nodes, features]
        h = h.unsqueeze(0)

        if hidden_state is None:
            lstm_out, new_hidden = self.temporal_rnn(h)
        else:
            lstm_out, new_hidden = self.temporal_rnn(h, hidden_state)

        last_hidden = lstm_out.squeeze(0)  # [num_nodes, hidden_size]
        output = self.output_proj(last_hidden)
        output = output * self.output_scale

        return output, new_hidden

class GATv2Model(DeepModel):
    """
    Purely spatial GCN model that does not use temporal axis.
    Useful to validate the use of graph-structure
    """
    def __init__(self, 
                 dataloader: GNNDataLoader, 
                 name: Optional[str] = None):
        super().__init__(dataloader, name=name)
        if not self.name:
            self.name = 'GATv2'

        self.model_color = paired_colors[3]
        self.dataloader = dataloader
        self.config_info['model'] = 'gatv2model'

        self._set_strategy(RecurrentStrategy())

    def set_model_hparams(self, hidden_size: int = 64, num_layers: int = 2,
                         temporal_layers: int = 2, dropout: float = 0.2):
        self.model_hparams_set = True
        self.model = GATv2Module(
            node_features=len(self.dataloader.feature_columns),
            hidden_size=hidden_size,
            num_layers=num_layers,
            temporal_layers=temporal_layers,
            dropout=dropout,
            prediction_horizon= self.dataloader.horizon_size
        ).to(self.device)
        
        model_hparams_config = {'hidden_size': hidden_size,
                                'num_layers' : num_layers,
                                'temporal_layers':temporal_layers,
                                'dropout':dropout}

        self.config_info['model_hparams'] = model_hparams_config
        self._state['model_initialized'] = True


        return self