import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Optional, Tuple, Literal
from torch_geometric.utils import add_self_loops

from ..strategies.gatv2lstm import StatelessGATv2LSTMStrategy, StatefullGATv2LSTMStrategy

from ....graphconstruction import has_self_loops
from ..debugging import ModelDebuggingReport, DebuggingLine
from ..deepmodel import DeepModel
from ....dataloading.dataloaders.deepdataloaders.graphdataloader import GraphDataLoaderManager


class GATv2LSTMArchitecture(nn.Module):
    def __init__(self,
                 num_features:      int, 
                 num_nodes:         int,
                 hidden_size:       int,
                 gat_hidden_dim:    int,
                 num_layers:        int,
                 dropout:           float,
                 temporal_layers:   int,
                 num_heads:         int,
                 horizon_size:      int, 
                 seq_length:        int,
                 self_loops :       bool,
        ): 
        super().__init__()

        self.num_features   = num_features
        self.num_nodes      = num_nodes 
        self.hidden_size    = hidden_size 
        self.gat_hidden_dim = gat_hidden_dim
        self.num_layers     = num_layers 
        self.dropout        = dropout
        self.temporal_layers= temporal_layers
        self.num_heads      = num_heads 
        self.horizon_size   = horizon_size
        self.seq_length     = seq_length
        self.self_loops     = self_loops

        # === Temporal LSTM ===
        self.lstm = nn.LSTM(
            input_size  = self.num_features,
            hidden_size = hidden_size,
            num_layers  = temporal_layers,
            dropout     = dropout if temporal_layers > 1 else 0,
            batch_first = False  # We'll use [seq_len, num_nodes, features] ==> dont think this is the case in my nodelstm model
        )

        self.lstm_out_norm = nn.LayerNorm(hidden_size)

        # GATv2 module
        self.gat1 = GATv2Conv(hidden_size, gat_hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATv2Conv(gat_hidden_dim * num_heads, gat_hidden_dim, heads=num_heads, concat=True)        

        # Final output layer
        self.fc = nn.Linear(gat_hidden_dim * num_heads + hidden_size, horizon_size)

    def forward(self,
                x:              torch.Tensor,
                edge_index:     torch.Tensor,
                edge_weight:    Optional[torch.Tensor] = None,
                hidden_state:   Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        output: Tensor
        h: Tensor 
        c: Tensor 
        # x shape: [num_nodes, num_features, seq_len]
        x_seq = x.permute(2, 0, 1)  # [seq_len, num_nodes, num_features]

        # Run LSTM
        if hidden_state is None:
            lstm_out, (h, c) = self.lstm(x_seq)
        else:
            lstm_out, (h, c) = self.lstm(x_seq, hidden_state)
        
        # lstm_out: [seq_len, num_nodes, hidden_size]
        # h: [num_layers, num_nodes, hidden_size]
        # c: [num_layers, num_nodes, hidden_size]
        
        # Take the last layer's hidden state
        ht_lstm = h[-1]  # [num_nodes, hidden_size]
        ht_lstm = self.lstm_out_norm(ht_lstm)

        # GATv2 forward
        h_gat = F.relu(self.gat1(ht_lstm, edge_index))
        h_gat = F.relu(self.gat2(h_gat, edge_index))

        # Concatenate LSTM embedding (skip connection)
        h_cat = torch.cat([h_gat, ht_lstm], dim=-1)

        # Final output
        output = self.fc(h_cat)  # [num_nodes, horizon_size]
        
        return output, h, c

    def debug(self,
              x:              torch.Tensor,
              edge_index:     torch.Tensor,
              edge_weight:    Optional[torch.Tensor] = None,
              hidden_state:   Optional[Tuple[torch.Tensor, torch.Tensor]] = None
              ):
        output: Tensor
        lstm_out: Tensor 
        ht_lstm: Tensor
        h: Tensor 
        c: Tensor 

        x_seq = x.permute(2, 0, 1)  # [seq_len, num_nodes, features]
        
        dbl1 = DebuggingLine(list(x_seq.shape), [self.seq_length, self.num_nodes, self.num_features])
        
        # Run a forward pass to get actual shapes
        if hidden_state is None:
            lstm_out, (h, c) = self.lstm(x_seq)
        else:
            lstm_out, (h, c) = self.lstm(x_seq, hidden_state)
            
        ht_lstm = h[-1]
        ht_lstm = self.lstm_out_norm(ht_lstm)
        # Add dropout between GAT layers
        h_gat = F.relu(self.gat1(ht_lstm, edge_index))
        h_gat = F.dropout(h_gat, p=self.dropout, training=self.training)
        h_gat = F.relu(self.gat2(h_gat, edge_index))
        h_gat = F.dropout(h_gat, p=self.dropout, training=self.training)
        h_cat = torch.cat([h_gat, ht_lstm], dim=-1)
        output = self.fc(h_cat)
        
        dbl2 = DebuggingLine(list(lstm_out.shape), [self.seq_length, self.num_nodes, self.hidden_size])
        dbl3 = DebuggingLine(list(ht_lstm.shape), [self.num_nodes, self.hidden_size])
        dbl4 = DebuggingLine(list(output.shape), [self.num_nodes, self.horizon_size])

        debugging_report = ModelDebuggingReport([dbl1, dbl2, dbl3, dbl4])

        return output, debugging_report   

class GATv2LSTMModel(DeepModel):
    """
    Purely spatial GCN model that does not use temporal axis.
    Useful to validate the use of graph-structure
    """
    def __init__(self, 
                 dataloadermanager: GraphDataLoaderManager, 
                 name: Optional[str] = None,
                 reset:             Literal['epoch','dataset'] = 'epoch',                 
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = 'GATV2LSTMModel'

        strategy = StatelessGATv2LSTMStrategy() if reset == 'epoch' else StatefullGATv2LSTMStrategy()

        super().__init__(
            dataloadermanager   = dataloadermanager, 
            name                = name, 
            verbose             = verbose, 
            deepfamily          = 'gnn', 
            strategy            = strategy
        )

    def set_model_hparams(self, 
                          hidden_size: int = 128, 
                          gat_hidden_dim : int = 64,
                          num_layers: int = 2,
                          temporal_layers: int = 2, 
                          dropout: float = 0.2,
                          num_heads: int = 2, 
                          self_loops:bool = False):
        self.model_hparams_set = True
        
        _num_features   = len(self.column_registration.get_by_type('feature'))
        _num_nodes      = self.dataloadermanager.dataorchestrator.data_context.num_nodes
        _horizon_size   = self.dataloadermanager.dataorchestrator.config.horizon_size
        _seq_length     = self.dataloadermanager.dataorchestrator.config.sequence_length

        self.model = GATv2LSTMArchitecture(
            num_features        = _num_features,
            num_nodes           = _num_nodes,
            num_layers          = num_layers,
            temporal_layers     = temporal_layers,
            hidden_size         = hidden_size,
            gat_hidden_dim      = gat_hidden_dim,
            dropout             = dropout,
            num_heads           = num_heads,
            horizon_size        = _horizon_size,
            self_loops          = self_loops,
            seq_length          = _seq_length
        ).to(self.device)
        
        model_hparams_config = {'hidden_size': hidden_size,
                                'num_layers' : num_layers,
                                'temporal_layers':temporal_layers,
                                'dropout':dropout,
                                'num_heads':num_heads,
                                'self_loops':self_loops}

        self.config_info['model_hparams'] = model_hparams_config
        self._update_status('model_hparams_set')
        return self