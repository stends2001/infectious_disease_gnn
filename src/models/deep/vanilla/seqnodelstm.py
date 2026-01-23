import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal

from .strategies import SequentializedLSTMStrategy

from ..deepmodel import DeepModel
from ....dataloading import DeepDataLoaderManager

class SeqLSTMModule(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, num_nodes):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False  # We'll permute to [sequence_length, N, F]
        )
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.num_nodes = num_nodes
    
    def forward(self, x):
        """
        Args:
            x: [N, sequence_length, F] where N is num_nodes
        
        Returns:
            y_hat: [N, output_dim]
        """
        # LSTM expects [sequence_length, batch, features]
        # Your input is [N, sequence_length, F]
        # Permute to [sequence_length, N, F]
        x = x.permute(2, 0, 1)  # [sequence_length, N, F]
        
        # Process entire sequence
        # out: [sequence_length, N, hidden_dim]
        # h_n: [num_layers, N, hidden_dim] - final hidden state
        out, (h_n, c_n) = self.lstm(x)
        
        # Take the output from the last timestep
        # out[-1]: [N, hidden_dim]
        last_output = out[-1]
        
        # Map to output dimension
        y_hat = self.fc(last_output)  # [N, output_dim]
        
        return y_hat
    
class SeqNodeLSTMModel(DeepModel):
    """
    Pure temporal LSTM baseline (no spatial structure).
    Each node processes independently.
    """
    def __init__(self, 
                 dataloadermanager: DeepDataLoaderManager, 
                 name: Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):        
        
        if not name:
            name = 'SequentializedNodeLSTM'
        
        super().__init__(dataloadermanager, name=name, verbose=verbose, deepfamily = 'vanilla', strategy = SequentializedLSTMStrategy(), model_color='#1f78b4')
        
        self.dataloadermanager = dataloadermanager

    def set_model_hparams(self, 
                          hidden_size: int = 128, 
                          num_layers: int = 2,
                          dropout: float = 0.2):
        self.model_hparams_set = True
        self.model = SeqLSTMModule(
            input_dim=len(self.column_registration.get_by_type('feature')),
            hidden_dim=hidden_size,
            num_layers=num_layers,
            # dropout=dropout,
            num_nodes= 400,
            output_dim=self.dataloadermanager.dataorchestrator.config.horizon_size
        ).to(self.device)
        
        model_hparams_config = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        }
        
        self.config_info['model_hparams'] = model_hparams_config
        self._update_status('model_hparams_set')
        return self
    

        