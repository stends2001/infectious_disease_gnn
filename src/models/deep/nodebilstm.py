import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
from torch_geometric.utils import add_self_loops

from .strategies import Strategy, RecurrentStrategy

from .graphneuralnetwork import GraphNeuralNetwork
from ...dataloading import GraphDataLoaderManager

class NodewiseBiLSTMModule(nn.Module):
    def __init__(self, node_features, hidden_size, num_layers, dropout, prediction_horizon):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=node_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False,
            bidirectional=True
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, prediction_horizon)
        )

    def forward(self, x_t, edge_index=None, edge_weight=None, hidden_state=None, debug=False):
        h = x_t.squeeze(-1)  # [num_nodes, features]

        if hidden_state is None:
            lstm_out, new_hidden = self.lstm(h)
        else:
            lstm_out, new_hidden = self.lstm(h, hidden_state)

        last_hidden = lstm_out.squeeze(0)  # [num_nodes, hidden_size * 2]
        output = self.output_proj(last_hidden)

        return output, new_hidden

class NodeBiLSTMModel(GraphNeuralNetwork):
    """
    Pure temporal bidirectional LSTM baseline (no spatial structure).
    Each node processes independently.
    """
    def __init__(self,
                 dataloadermanager: GraphDataLoaderManager,
                 name: Optional[str] = None,
                 verbose: Literal[-1, 0, 1, 2] = -1):

        if not name:
            name = 'BiLSTM'

        super().__init__(dataloadermanager, name=name, verbose=verbose)
        self._set_strategy(RecurrentStrategy())

    def set_model_hparams(self,
                          hidden_size: int = 128,
                          num_layers: int = 2,
                          dropout: float = 0.2):
        self.model_hparams_set = True
        self.model = NodewiseBiLSTMModule(
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
