
import torch, random, numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from ..dataloading.gnndataloader import GNNDataLoader
from .modelcore import ModelCore
from ..metrics.spike_weighted_mse import spike_weighted_mse
import torch_geometric.nn as pyg_nn

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch_geometric.nn as pyg_nn
from .modelcore import DeepLearningModelCore

class ArchitectureTemporalGATLSTM(torch.nn.Module):
    def __init__(self, node_features, periods, lstm_hidden_size=32, gat_hidden_size=32, num_heads=4):
        super(ArchitectureTemporalGATLSTM, self).__init__()
        self.periods = periods
        
        # Graph Attention Layer (GAT)
        # num_heads: Number of attention heads, typically 4 or 8.
        self.gat = GATConv(node_features, gat_hidden_size, heads=num_heads, concat=True)

        # LSTM layer for sequential modeling
        self.lstm = torch.nn.LSTM(input_size=gat_hidden_size * num_heads, hidden_size=lstm_hidden_size, num_layers=2, batch_first=True)

        # Output layer for prediction (1 output per node)
        self.linear = torch.nn.Linear(lstm_hidden_size, 1)

    def forward(self, x, edge_index, edge_weight):
        # x: [num_nodes, node_features, periods]  # Node features across time
        gatt_out_seq = []

        for t in range(self.periods):
            # Get node features for time t (shape: [num_nodes, node_features])
            xt = x[:, :, t]

            # Apply GAT to the features at this time step (output shape: [num_nodes, gat_hidden_size * num_heads])
            xt_gat = self.gat(xt, edge_index)
            xt_gat = torch.relu(xt_gat)  # Apply ReLU activation after GAT

            gatt_out_seq.append(xt_gat)

        # Stack the graph attention outputs along the time dimension (shape: [num_nodes, periods, gat_hidden_size * num_heads])
        gatt_out_seq = torch.stack(gatt_out_seq, dim=1)

        # LSTM expects input shape: [batch_size, seq_length, features]
        lstm_out, _ = self.lstm(gatt_out_seq)

        # Use the last time step output for prediction (shape: [num_nodes, lstm_hidden_size])
        last_out = lstm_out[:, -1, :]

        # Output prediction (shape: [num_nodes, 1])
        out = self.linear(last_out)

        return out.squeeze(-1)  # Return shape [num_nodes,]

class GATLSTMModel_Cleaned(DeepLearningModelCore):
    """

    """
    def __init__(self, dataloader: GNNDataLoader, name= None):
        super().__init__(dataloader, name= name)
        if not self.name:
            self.name = f'GAT_LSTM'

        self.model_color = '#9467BD'
        self.dataloader  = dataloader

    def set_model_hparams(self, 
                            gat_hidden_size: int = 128,
                            lstm_hidden_size: int = 64,
                            num_heads: int   = 4
                            ):
        
        """ 
        by default, using the following:

        - optimizer: Adam
        - scheduler: step decau using `lr_scheudler.StepLR` 
        """
        self.model_hparams_set = True

        self.model = ArchitectureTemporalGATLSTM(
            node_features=len(self.dataloader.feature_columns),
                                periods=self.dataloader.periods,
                                lstm_hidden_size=gat_hidden_size,
                                gat_hidden_size=lstm_hidden_size,
                                num_heads=num_heads
        ).to(self.device)        

        return self