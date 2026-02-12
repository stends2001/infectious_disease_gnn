import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Optional, Tuple, Literal
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GCNConv
from ..strategies.gcn import StandardGNNStrategy

from ....graphconstruction import has_self_loops
from ..debugging import ModelDebuggingReport, DebuggingLine
from ..deepmodel import DeepModel
from ....dataloading.dataloaders.deepdataloaders.graphdataloader import GraphDataLoaderManager


class TGCNArchitecture(nn.Module):
    """
    Temporal Graph Convolutional Network (TGCN)
    - Alternates GCN (spatial) and 1D temporal convolution (temporal)
    - Fully deterministic, preserves node variance
    """

    def __init__(self,
                 num_features: int,
                 num_nodes: int,
                 gcn_hidden_dim: int,
                 temporal_hidden_dim: int,
                 horizon_size: int,
                 seq_length: int,
                 num_gcn_layers: int = 1,
                 num_temporal_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()

        self.num_features = num_features
        self.num_nodes = num_nodes
        self.gcn_hidden_dim = gcn_hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.horizon_size = horizon_size
        self.seq_length = seq_length
        self.num_gcn_layers = num_gcn_layers
        self.num_temporal_layers = num_temporal_layers
        self.dropout = dropout

        # --- Spatial: stack of GCN layers ---
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(num_features, gcn_hidden_dim))
        for _ in range(1, num_gcn_layers):
            self.gcn_layers.append(GCNConv(gcn_hidden_dim, gcn_hidden_dim))

        # --- Temporal: 1D Conv layers over time dimension ---
        temporal_layers = []
        in_channels = gcn_hidden_dim
        for _ in range(num_temporal_layers):
            temporal_layers.append(nn.Conv1d(in_channels, temporal_hidden_dim, kernel_size=3, padding=1))
            temporal_layers.append(nn.ReLU())
            temporal_layers.append(nn.Dropout(dropout))
            in_channels = temporal_hidden_dim
        self.temporal_conv = nn.Sequential(*temporal_layers)

        # --- Output layer ---
        self.fc = nn.Linear(temporal_hidden_dim, horizon_size)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        """
        x: [num_nodes, num_features, seq_length]
        edge_index: [2, num_edges]
        """

        # --- Apply GCN at each timestep ---
        gcn_out_seq = []
        for t in range(self.seq_length):
            x_t = x[:, :, t]  # [num_nodes, num_features]
            h = x_t
            for gcn in self.gcn_layers:
                h = F.relu(gcn(h, edge_index, edge_weight))
                h = F.dropout(h, p=self.dropout, training=self.training)
            gcn_out_seq.append(h)

        # Stack: [num_nodes, gcn_hidden_dim, seq_length]
        h_seq = torch.stack(gcn_out_seq, dim=2)

        # --- Temporal convolution over time ---
        h_temporal = self.temporal_conv(h_seq)  # [num_nodes, temporal_hidden_dim, seq_length]

        # Take last timestep for prediction
        last_step = h_temporal[:, :, -1]  # [num_nodes, temporal_hidden_dim]

        # --- Output ---
        out = self.fc(last_step)  # [num_nodes, horizon_size]
        return out
    
    def debug(self, x: Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) ->  Tuple[Tensor, 'ModelDebuggingReport']:
        output:   Tensor

        if self.seq_length > 1:
            print('Please note that spatial GCN only takes the final sequence of data. Input [num_nodes, num_features, seq_len] will be used as [num_nodes, num_features, -1]')

        x_t = x[:, :, -1] 
        
        dbl1 = DebuggingLine(list(x_t.shape),       [self.num_nodes, self.num_features])  

        h = x_t      
        
        debugging_report = ModelDebuggingReport([dbl1])


        # for conv in self.convs:
        #     h = conv(h, edge_index, edge_weight)
        #     h = F.relu(h)
        #     h = self.dropout(h)

        output = x_t   

        return output, debugging_report
    


class TGCNModel(DeepModel):
    """
    Simple spatial GCN model without temporal components.
    """
    def __init__(self, 
                 dataloadermanager: GraphDataLoaderManager, 
                 name: Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = 'SpatialGNN'        

        super().__init__(dataloadermanager, name=name, deepfamily='gnn' , verbose=verbose, strategy=StandardGNNStrategy())                 

    def set_model_hparams(self, 
                          gcn_hidden_dim: int = 64, 
                          t_hidden_dim: int = 32,
                          num_layers: int = 1,
                          num_layers_t: int = 2,
                          emb_size: int = 32,
                          dropout: float = 0.2):
        self.model_hparams_set = True

        _num_features   = len(self.column_registration.get_by_type('feature'))
        _num_nodes      = self.dataloadermanager.dataorchestrator.data_context.num_nodes
        _horizon_size   = self.dataloadermanager.dataorchestrator.config.horizon_size
        _seq_length     = self.dataloadermanager.dataorchestrator.config.sequence_length

        self.model = TGCNArchitecture(
            num_features    = _num_features,
            num_nodes       = _num_nodes,
            gcn_hidden_dim     = gcn_hidden_dim,
            temporal_hidden_dim= t_hidden_dim,
            num_gcn_layers      = num_layers,
            num_temporal_layers = num_layers_t,
            dropout         = dropout,
            horizon_size    = _horizon_size,
            seq_length      =_seq_length
    
        ).to(self.device)

        model_hparams_config = {
        }

        self.config_info['model_hparams'] = model_hparams_config
        self._update_status('model_hparams_set')
        return self