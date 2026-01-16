import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

from .strategies import RecurrentLSTMStrategy

from ..deepmodel import DeepModel
from ....dataloading import DeepDataLoaderManager

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

    def forward(self, x_t, hidden_state=None, debug=False):
        h = x_t.squeeze(-1)  # [num_nodes, features]

        if hidden_state is None:
            lstm_out, new_hidden = self.lstm(h)
        else:
            lstm_out, new_hidden = self.lstm(h, hidden_state)

        last_hidden = lstm_out.squeeze(0)  # [num_nodes, hidden_size * 2]
        output = self.output_proj(last_hidden)

        return output, new_hidden

class NodeBiLSTMModel(DeepModel):
    """
    """
    def __init__(self,
                 dataloadermanager: DeepDataLoaderManager,
                 name: Optional[str] = None,
                 verbose: Literal[-1, 0, 1, 2] = -1):

        if not name:
            name = 'BiLSTM'

        super().__init__(dataloadermanager, name=name, verbose=verbose, strategy=RecurrentLSTMStrategy())

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
