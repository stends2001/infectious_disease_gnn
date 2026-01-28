import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import pandas as pd
import numpy as np
from typing import Optional,Literal

from .strategies import StandardGNNStrategy
from ..basestrategy import desequentialize_x

from ..deepmodel import DeepModel
from ....dataloading import GraphDataLoaderManager

from ....utils.textformatting import warning_emoji

class GATv2Module(nn.Module):
    """
    """
    def __init__(self,
                 node_features: int,
                 hidden_size: int,
                 num_layers: int,
                 num_heads: int,
                 dropout: float,
                 horizon_size: int,
                 seq_length: int
        ):
        super().__init__()

        self.node_features      = node_features
        self.hidden_size        = hidden_size
        self.num_layers         = num_layers
        self.num_heads          = num_heads
        self.horizon_size = horizon_size
        self.seq_length = seq_length

        # === Spatial GCN layers ===
        self.spatial_convs = nn.ModuleList()
        self.spatial_convs.append(GATv2Conv(node_features, hidden_size, heads=num_heads, dropout=dropout))

        for _ in range(num_layers - 1):
            self.spatial_convs.append(
                GATv2Conv(hidden_size * num_heads, hidden_size, heads=num_heads, dropout=dropout)
            )

        # === Output layer ===
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * num_heads, hidden_size // 2),  # FIXED
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None,
                debug: bool = False
                ) -> torch.Tensor:
        """
        Forward pass.

        x: [num_nodes, node_features, tt]
        edge_index: [2, num_edges]
        """

        h = desequentialize_x(x, self.seq_length)

        # Apply GAT layers
        for gat in self.spatial_convs:
            h = gat(h, edge_index)
            h = F.elu(h)
            h = self.dropout(h)

        # Project to output
        output = self.output_proj(h)
 

        return output

class GATv2Model(DeepModel):
    """
    Simple spatial GCN model without temporal components.
    """
    def __init__(self, 
                 dataloadermanager: GraphDataLoaderManager, 
                 name: Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = 'GATv2Model'        

        super().__init__(dataloadermanager, name=name, deepfamily='gnn' , verbose=verbose, strategy=StandardGNNStrategy(), model_color='#B87200')                 

    def set_model_hparams(self, 
                          hidden_size: int = 64, 
                          num_layers: int = 2,
                          num_heads: int = 2,
                          dropout: float = 0.2):
        self.model_hparams_set = True
        self.model = GATv2Module(
            node_features=len(self.column_registration.get_by_type('feature')),
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads = num_heads,
            dropout=dropout,
            horizon_size=self.dataloadermanager.dataorchestrator.config.horizon_size,
            seq_length          = self.dataloadermanager.dataorchestrator.config.sequence_length            
        ).to(self.device)
        
        model_hparams_config = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        }

        self.config_info['model_hparams'] = model_hparams_config
        self._update_status('model_hparams_set')
        # return self