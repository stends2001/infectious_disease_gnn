import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric_temporal.nn.recurrent import A3TGCN
from typing import Optional, Tuple, Literal
from torch_geometric.utils import add_self_loops

from .strategies import Strategy, RecurrentGRUStrategy

from .graphneuralnetwork import GraphNeuralNetwork
from ...dataloading.dataloaders.deeploader.graphdataloadermanager import GraphDataLoaderManager

class NodewiseGRUModule(nn.Module):
    def __init__(self, node_features, hidden_size, num_layers, dropout, prediction_horizon):
        super().__init__()

        self.gru = nn.GRU(
            input_size=node_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False  # [seq_len, num_nodes, features]
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, prediction_horizon)
        )

    def forward(self, x_t, edge_index=None, edge_weight=None, hidden_state=None, debug=False):
        h = x_t.squeeze(-1)  # [num_nodes, features]

        if hidden_state is None:
            gru_out, new_hidden = self.gru(h)
        else:
            gru_out, new_hidden = self.gru(h, hidden_state)

        last_hidden = gru_out.squeeze(0)  # [num_nodes, hidden_size]
        output = self.output_proj(last_hidden)

        return output, new_hidden

class NodeGRUModel(GraphNeuralNetwork):
    """
    Pure temporal GRU baseline (no spatial structure).
    Each node processes independently.
    """
    def __init__(self,
                 dataloadermanager: GraphDataLoaderManager,
                 name: Optional[str] = None,
                 verbose: Literal[-1, 0, 1, 2] = -1):

        if not name:
            name = 'GRU'

        super().__init__(dataloadermanager, name=name, verbose=verbose)
        self._set_strategy(RecurrentGRUStrategy())

    def set_model_hparams(self,
                          hidden_size: int = 128,
                          num_layers: int = 2,
                          dropout: float = 0.2):
        self.model_hparams_set = True
        self.model = NodewiseGRUModule(
            node_features=len(self.column_registration.get_by_type('feature')),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            prediction_horizon=self.dataloadermanager.dataorchestrator.config.horizon_size
        ).to(self.device)

        self.config_info['model_hparams'] = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        }

        self._update_status('model_hparams_set')
        return self
