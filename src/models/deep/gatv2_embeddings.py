import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, ChebConv, GINConv, GATv2Conv
from torch_geometric_temporal.nn.recurrent import DCRNN, TGCN, A3TGCN
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from ...dataloading.deepdataloader import DeepDataLoader
from .deepmodel import DeepModel
from .strategies.recurrent_strategy import RecurrentStrategy

class GATv2EmbeddedModule(nn.Module):
    """
    Recurrent GATv2-based spatiotemporal model.
    Processes a single time step at a time and keeps hidden state between steps.
    """
    def __init__(self,
                 num_nodes: int,
                 node_features: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 temporal_layers: int,
                 num_heads: int,
                 prediction_horizon: int,
                 embedding_dim: int
        ):
        super().__init__()

        self.num_nodes          = num_nodes
        self.node_features      = node_features
        self.hidden_size        = hidden_size
        self.num_layers         = num_layers
        self.temporal_layers    = temporal_layers
        self.prediction_horizon = prediction_horizon
        self.embedding_dim      = embedding_dim

        # === (Learnable) node embeddings ===
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)

        first_layer_input_dim= node_features + embedding_dim

        # === Spatial GATv2 layers ===
        self.spatial_convs = nn.ModuleList()
        self.spatial_convs.append(GATv2Conv(first_layer_input_dim, hidden_size, heads=num_heads, dropout=dropout))

        for _ in range(num_layers - 1):
            self.spatial_convs.append(
                GATv2Conv(hidden_size * num_heads, hidden_size, heads=num_heads, dropout=dropout)
            )

        self.total_gat_out_dim = hidden_size * num_heads

        # === Temporal LSTM ===
        self.temporal_rnn = nn.LSTM(
            input_size=self.total_gat_out_dim,
            hidden_size=hidden_size,
            num_layers=temporal_layers,
            dropout=dropout if temporal_layers > 1 else 0,
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
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                node_ids: Optional[torch.Tensor]= None,
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

        # Get node embeddings
        if node_ids is None:
            node_ids = torch.arange(h.size(0), device=h.device)       
        node_emb = self.node_embeddings(node_ids)  # [num_nodes, embedding_dim] 

        # Concatenate features with embeddings
        h = torch.cat([h, node_emb], dim=-1)  # [num_nodes, node_features + embedding_dim]


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

class GATv2EmbeddingsModel(DeepModel):
    """
    Purely spatial GCN model that does not use temporal axis.
    Useful to validate the use of graph-structure
    """
    def __init__(self, 
                 dataloader: DeepDataLoader, 
                 name: Optional[str] = None):
        super().__init__(dataloader, name=name)
        
        if not self.name:
            self.name = 'GATv2'

        self.dataloader = dataloader
        

        self._set_strategy(RecurrentStrategy())

    def set_model_hparams(self, 
                          hidden_size: int = 64, 
                          num_layers: int = 2,
                          temporal_layers: int = 2, 
                          dropout: float = 0.2,
                          num_heads: int = 2,
                          embedding_dim: int = 16):
        self.model_hparams_set = True
        self.model = GATv2EmbeddedModule(
            num_nodes=400,
            node_features=len(self.dataloader.feature_columns),
            hidden_size=hidden_size,
            num_layers=num_layers,
            temporal_layers=temporal_layers,
            dropout=dropout,
            num_heads = num_heads,
            prediction_horizon= self.dataloader.horizon_size,
            embedding_dim = embedding_dim
        ).to(self.device)
        
        model_hparams_config = {'hidden_size': hidden_size,
                                'num_layers' : num_layers,
                                'temporal_layers':temporal_layers,
                                'dropout':dropout,
                                'embedding_dim': embedding_dim,
                                'num_heads':num_heads}

        self.config_info['model_hparams'] = model_hparams_config
        self._state['model_initialized'] = True


        return self