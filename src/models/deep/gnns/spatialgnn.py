import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
from typing import Optional,Literal

from .strategies import StandardGNNStrategy

from ..deepmodel import DeepModel
from ....dataloading import GraphDataLoaderManager

from ....utils.textformatting import warning_emoji

class SimpleGCNModule(nn.Module):
    """
    Simple spatial-only GCN model.
    Processes graph structure without temporal dynamics.
    """
    def __init__(self,
                 node_features: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 horizon_size: int,
        ):
        super().__init__()

        self.node_features      = node_features
        self.hidden_size        = hidden_size
        self.num_layers         = num_layers
        self.horizon_size = horizon_size

        # === Spatial GCN layers ===
        self.spatial_convs = nn.ModuleList()
        self.spatial_convs.append(GCNConv(node_features, hidden_size))

        for _ in range(num_layers - 1):
            self.spatial_convs.append(GCNConv(hidden_size, hidden_size))

        # === Output layer ===
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon_size),
            # nn.Sigmoid()  # outputs probability [0, 1]
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

        if x.ndim == 3:
            if x.shape[-1] != 1:
                print(f'{warning_emoji} This spatial GNN takes in only one sequence of data. No time axis expected')

        # if debug:
        #     print(f'input size: {x.shape}')

        h = x.squeeze(-1)

        # if debug:
        #     print(f'squeezed input size: {h.shape}')
        if debug:
            print(f'Edge index: {edge_index.shape}')
            print(f'Edge sample: {edge_index[:, :5]}')
            print(f'Node features before GCN: {h[:3, :3]}')
        # Apply GCN layers
        for gcn in self.spatial_convs:
            h = gcn(h, edge_index, edge_weight)
            h = F.relu(h)
            h = self.dropout(h)
        if debug:
            print(f'Node features after GCN: {h[:3, :3]}')
            print(f'did features change? {torch.norm(h - x.squeeze(-1))}')
        # if debug:
        #     print(f'convolved input size: {h.shape}')

        # Project to output
        output = self.output_proj(h)

        if debug:
            print(f'output size: {output.shape}')        

        return output

class SpatialGNNModel(DeepModel):
    """
    Simple spatial GCN model without temporal components.
    """
    def __init__(self, 
                 dataloadermanager: GraphDataLoaderManager, 
                 name: Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = 'SpatialGNN'        

        super().__init__(dataloadermanager, name=name, verbose=verbose, strategy=StandardGNNStrategy())                 

    def set_model_hparams(self, 
                          hidden_size: int = 64, 
                          num_layers: int = 2,
                          dropout: float = 0.2):
        self.model_hparams_set = True
        self.model = SimpleGCNModule(
            node_features=len(self.column_registration.get_by_type('feature')),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon_size=self.dataloadermanager.dataorchestrator.config.horizon_size
        ).to(self.device)
        
        model_hparams_config = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        }

        self.config_info['model_hparams'] = model_hparams_config
        self._update_status('model_hparams_set')
        return self