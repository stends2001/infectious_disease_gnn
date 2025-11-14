import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric_temporal.nn.recurrent import A3TGCN
from typing import Optional, Tuple
from torch_geometric.utils import add_self_loops

from src.models.deep.strategies.base import Strategy
from src.models.deep.strategies.standard_strategy import StandardStrategy
from src.models.deep.strategies.recurrent_strategy import RecurrentStrategy
from src.models.deep.strategies.recurrentlstm_strategy import RecurrentLSTMStrategy

from .graphneuralnetwork import GraphNeuralNetwork
from ...UPDATED_dataloading.dataloaders.deeploader.graphdataloadermanager import GraphDataLoaderManager


class GATv2Module(nn.Module):
    """
    Recurrent GATv2-based spatiotemporal model.
    Processes a single time step at a time and keeps hidden state between steps.
    """
    def __init__(self,
                 node_features: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 temporal_layers: int,
                 num_heads: int,
                 prediction_horizon: int,
                 self_loops : bool
        ):
        super().__init__()

        self.node_features = node_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.temporal_layers = temporal_layers
        self.prediction_horizon = prediction_horizon
        self.self_loops = self_loops

        # === Spatial GATv2 layers ===
        self.spatial_convs = nn.ModuleList()
        self.spatial_convs.append(GATv2Conv(node_features, hidden_size, heads=num_heads, dropout=dropout))

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
                debug: bool = False,
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

        if self.self_loops:
            edge_index, edge_weight = add_self_loops(edge_index, edge_attr=edge_weight, num_nodes=x_t.size(0))

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

class GATv2Model(GraphNeuralNetwork):
    """
    Purely spatial GCN model that does not use temporal axis.
    Useful to validate the use of graph-structure
    """
    def __init__(self, 
                 dataloadermanager: GraphDataLoaderManager, 
                 name: Optional[str] = None):
        super().__init__(dataloadermanager, name=name)
        
        if not self.name:
            self.name = 'GATv2'

        self.dataloadermanager = dataloadermanager
        

        self._set_strategy(RecurrentStrategy())

    def set_model_hparams(self, 
                          hidden_size: int = 64, 
                          num_layers: int = 2,
                          temporal_layers: int = 2, 
                          dropout: float = 0.2,
                          num_heads: int = 2, 
                          self_loops:bool = False):
        self.model_hparams_set = True
        self.model = GATv2Module(
            node_features   = len(self.column_registration.get_by_type('feature')),
            hidden_size     = hidden_size,
            num_layers      = num_layers,
            temporal_layers = temporal_layers,
            dropout         = dropout,
            num_heads       = num_heads,
            prediction_horizon= self.dataloadermanager.dataorchestrator.config.horizon_size,
            self_loops = self_loops
        ).to(self.device)
        
        model_hparams_config = {'hidden_size': hidden_size,
                                'num_layers' : num_layers,
                                'temporal_layers':temporal_layers,
                                'dropout':dropout,
                                'num_heads':num_heads,
                                'self_loops':self_loops}

        self.config_info['model_hparams'] = model_hparams_config
        self._state['model_initialized'] = True


        return self