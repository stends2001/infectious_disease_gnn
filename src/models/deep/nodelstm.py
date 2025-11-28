import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric_temporal.nn.recurrent import A3TGCN
from typing import Optional, Tuple
from torch_geometric.utils import add_self_loops

from .strategies import Strategy, RecurrentStrategy

from .graphneuralnetwork import GraphNeuralNetwork
from ...dataloading.dataloaders.deeploader.graphdataloadermanager import GraphDataLoaderManager

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
    
    def forward(self, x_t, edge_index=None, edge_weight=None, hidden_state=None, debug=False):
        # x_t: [num_nodes, node_features, num_seq]
        # Ignore edge_index - we don't use spatial info
        
        h = x_t.squeeze(-1)  # [num_nodes, features]
        
        if hidden_state is None:
            lstm_out, new_hidden = self.lstm(h)
        else:
            lstm_out, new_hidden = self.lstm(h, hidden_state)
        
        last_hidden = lstm_out.squeeze(0)  # [num_nodes, hidden_size]
        output = self.output_proj(last_hidden)
        
        return output, new_hidden

class NodeLSTMModel(GraphNeuralNetwork):
    """
    Pure temporal LSTM baseline (no spatial structure).
    Each node processes independently.
    """
    def __init__(self, 
                 dataloadermanager: GraphDataLoaderManager, 
                 name: Optional[str] = None):
        super().__init__(dataloadermanager, name=name)
        
        if not self.name:
            self.name = 'LSTM_Baseline'
        
        self.dataloadermanager = dataloadermanager
        self._set_strategy(RecurrentStrategy())  # Same strategy as GATv2

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
        self._state['model_initialized'] = True
        
        return self