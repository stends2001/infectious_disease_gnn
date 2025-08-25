
import torch, random, numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric_temporal.nn.recurrent import A3TGCN
import pandas as pd

from ..dataloading.gnnsequencedataloader import GNNSequenceDataLoader
from .modelcore import DeepLearningModelCore


class A3TGCN_LSTM_Seq(torch.nn.Module):
    def __init__(self, node_features, periods, a3tgcn_hidden_size, lstm_hidden_size, lstm_num_layers, seq_len=5):
        super(A3TGCN_LSTM_Seq, self).__init__()
        self.periods = periods
        self.seq_len = seq_len
        
        # A3TGCN processes input of shape [num_nodes, node_features, periods]
        self.tgnn = A3TGCN(in_channels=node_features,
                           out_channels=a3tgcn_hidden_size,
                           periods=periods)
        
        # LSTM processes sequence dimension with tgnn embeddings as features per node
        self.lstm = torch.nn.LSTM(input_size=a3tgcn_hidden_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)
        
        # Final linear layer for node-wise prediction
        self.linear = torch.nn.Linear(lstm_hidden_size, 1)
        
    def forward(self, x, edge_index, edge_weight = None):
        """
        x: [seq_len, num_nodes, node_features, periods]
        edge_index: [2, num_edges]
        doesn't do anything with edge_weight
        """
        seq_len, num_nodes, node_features, periods = x.shape
        assert seq_len == self.seq_len and periods == self.periods, "Input dimensions mismatch"
        
        embeddings = []
        for t in range(seq_len):
            # For each snapshot in the sequence: shape [num_nodes, node_features, periods]
            x_t = x[t]
            # A3TGCN expects this shape as input directly
            h_t = self.tgnn(x_t, edge_index)  # [num_nodes, 32]
            embeddings.append(h_t)
        
        # Stack embeddings along sequence dim: [num_nodes, seq_len, 32]
        h = torch.stack(embeddings, dim=1)
        
        # LSTM expects input shape (batch, seq_len, features)
        # Treat each node as a batch item:
        lstm_out, _ = self.lstm(h)  # [num_nodes, seq_len, lstm_hidden_size]
        
        # Use last output along sequence for each node
        last_out = lstm_out[:, -1, :]  # [num_nodes, lstm_hidden_size]
        
        out = self.linear(last_out)  # [num_nodes, 1]
        return out.squeeze(-1)       # [num_nodes]


class A3TGCNModel_cleaned(DeepLearningModelCore):
    """

    """
    def __init__(self, dataloader: GNNSequenceDataLoader, name= None):
        super().__init__(dataloader, name= name)
        if not self.name:
            self.name = f'A3TGCN'

        self.model_color = '#2CA02C'
        self.dataloader  = dataloader

    def set_model_hparams(self, 
                            a3tgcn_hidden_size: int = 32,
                            lstm_hidden_size: int   = 64, 
                            lstm_num_layers: int    = 2
                            ):
        
        """ 
        by default, using the following:

        - optimizer: Adam
        - scheduler: step decau using `lr_scheudler.StepLR` 
        """

        self.model = A3TGCN_LSTM_Seq(
            node_features=len(self.dataloader.feature_columns),
            a3tgcn_hidden_size = a3tgcn_hidden_size,
            lstm_hidden_size=lstm_hidden_size, 
            lstm_num_layers = lstm_num_layers,
            periods=self.dataloader.periods,
            seq_len=self.dataloader.seq_len  # Make sure your dataloader sets this attribute!
        ).to(self.device)

        return self