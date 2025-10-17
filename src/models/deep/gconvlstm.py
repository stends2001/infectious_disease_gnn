from .deepmodel import DeepModel
from ...dataloading.deepdataloader import DeepDataLoader
from .strategies.recurrentlstm_strategy import RecurrentLSTMStrategy

from typing import Optional

import torch 
from torch_geometric_temporal.nn.recurrent import GConvLSTM
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvLSTM
from typing import Optional, Tuple


class GConvLSTMModule(nn.Module):
    """
    Recurrent GCN-based spatiotemporal model using GConvLSTM.
    Processes a single time step at a time and maintains (h, c) hidden state.
    """
    def __init__(self,
                 node_features: int,
                 hidden_size: int = 64,
                 dropout: float = 0.2,
                 prediction_horizon: int = 1,
                 K = 1):
        super().__init__()

        self.node_features = node_features
        self.hidden_size = hidden_size
        self.prediction_horizon = prediction_horizon

        # === Temporal GConvLSTM ===
        self.recurrent = GConvLSTM(in_channels=node_features,
                                   out_channels=hidden_size,
                                   K=K)  # K is the filter size

        # === Output projection ===
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
                cell_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                debug: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for one time step.

        x_t: [num_nodes, node_features]
        edge_index: [2, num_edges]
        hidden_state: Tuple (h, c), each [num_nodes, hidden_size]
        """
        if debug:
            print("input shape:", x_t.shape)

        x_t = x_t.squeeze(-1)

        h, c = hidden_state, cell_state

        h_out, c_out = self.recurrent(x_t, edge_index, edge_weight, h, c)

        h_out = F.relu(h_out)
        h_out = self.dropout(h_out)

        output = self.output_proj(h_out)
        output = output * self.output_scale

        return output, h_out, c_out

    


class GConvLSTMModel(DeepModel):
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
        
        self._set_strategy(RecurrentLSTMStrategy())

    def set_model_hparams(self, hidden_size: int = 64, dropout: float = 0.2, K: int = 1):
        self.model_hparams_set = True
        self.model = GConvLSTMModule(
            node_features=len(self.dataloader.feature_columns),
            hidden_size=hidden_size,
            dropout=dropout,
            prediction_horizon= self.dataloader.horizon_size,
            K = K
        ).to(self.device)
        
        model_hparams_config = {'hidden_size': hidden_size,
                                'dropout' : dropout,
                                'K':K}

        self.config_info['model_hparams'] = model_hparams_config
        self._state['model_initialized'] = True


        return self