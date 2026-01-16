import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal

from .strategies import RecurrentLSTMStrategy

from ..deepmodel import DeepModel
from ....dataloading import DeepDataLoaderManager

class NodewiseLSTMModule(nn.Module):
    def __init__(self, node_features, hidden_size, num_layers, dropout, prediction_horizon):
        super().__init__()
        
        # Single LSTM applied independently to each node
        self.lstm = nn.LSTM(
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
    
    def forward(self, x_t, hidden_state=None, debug=False):
        # x_t: [num_nodes, node_features, num_seq]
        
        h = x_t.squeeze(-1)  # [num_nodes, features]
        
        if hidden_state is None:
            lstm_out, new_hidden = self.lstm(h)
        else:
            lstm_out, new_hidden = self.lstm(h, hidden_state)
            
        h, c = new_hidden
        last_hidden = lstm_out.squeeze(0)  # [num_nodes, hidden_size]
        output = self.output_proj(last_hidden)
        
        return output, h, c

class NodeLSTMModel(DeepModel):
    """
    Pure temporal LSTM baseline (no spatial structure).
    Each node processes independently.
    """
    def __init__(self, 
                 dataloadermanager: DeepDataLoaderManager, 
                 name: Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):        
        
        if not name:
            name = 'LSTM_Baseline'
        
        super().__init__(dataloadermanager, name=name, verbose=verbose, strategy = RecurrentLSTMStrategy())
        
        self.dataloadermanager = dataloadermanager
        # self._set_strategy(RecurrentLSTMStrategy())  # Same strategy as GATv2

    def set_model_hparams(self, 
                          hidden_size: int = 128, 
                          num_layers: int = 2,
                          dropout: float = 0.2):
        self.model_hparams_set = True
        self.model = NodewiseLSTMModule(
            node_features=len(self.column_registration.get_by_type('feature')),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            prediction_horizon=self.dataloadermanager.dataorchestrator.config.horizon_size
        ).to(self.device)
        
        model_hparams_config = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        }
        
        self.config_info['model_hparams'] = model_hparams_config
        self._update_status('model_hparams_set')
        return self
    

    