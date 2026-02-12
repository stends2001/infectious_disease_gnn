import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Optional, Tuple, Literal
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GCNConv
from ..strategies.gcn import StandardGNNStrategy

from ..debugging import ModelDebuggingReport, DebuggingLine
from ..deepmodel import DeepModel
from ....dataloading.dataloaders.deepdataloaders.graphdataloader import GraphDataLoaderManager

class SimpleGCNModule(nn.Module):
    """
    Pure spatial GCN.
    Uses only the last timestep of sequential input.
    """

    def __init__(self,
                 num_features: int,
                 num_nodes: int,
                 hidden_size: int,
                 emb_size: int,
                 num_layers: int,
                 dropout: float,
                 horizon_size: int,
                 seq_length: int):

        super().__init__()

        self.num_features = num_features
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon_size = horizon_size
        self.seq_length = seq_length
        self.emb_size = emb_size



        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features + emb_size, hidden_size))

        self.node_emb = nn.Embedding(num_nodes, emb_size)

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_size, hidden_size))

        self.dropout = nn.Dropout(dropout)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon_size)
        )

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None):

        # x: [num_nodes, num_features, seq_len]

        # --- collapse time ---
        x = x[:, :, -1]  # use last timestep
        # alternative: x = x.mean(dim=2)

        node_ids = torch.arange(self.num_nodes, device=x.device)
        emb = self.node_emb(node_ids)
        h = torch.cat([x, emb], dim=-1)

        for conv in self.convs:
            h = conv(h, edge_index, edge_weight)
            h = F.relu(h)
            h = self.dropout(h)

        output = self.output_proj(h)

        return output
    
    def debug(self, x: Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) ->  Tuple[Tensor, 'ModelDebuggingReport']:
        output:   Tensor

        if self.seq_length > 1:
            print('Please note that spatial GCN only takes the final sequence of data. Input [num_nodes, num_features, seq_len] will be used as [num_nodes, num_features, -1]')

        x_t = x[:, :, -1] 
        
        dbl1 = DebuggingLine(list(x_t.shape),       [self.num_nodes, self.num_features])  

        h = x_t      
        
        debugging_report = ModelDebuggingReport([dbl1])


        for conv in self.convs:
            h = conv(h, edge_index, edge_weight)
            h = F.relu(h)
            h = self.dropout(h)

        output = self.output_proj(h)        

        return output, debugging_report



class SimpleGCNModel(DeepModel):
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
                          hidden_size: int = 64, 
                          num_layers: int = 2,
                          emb_size: int = 32,
                          dropout: float = 0.2):
        self.model_hparams_set = True

        _num_features   = len(self.column_registration.get_by_type('feature'))
        _num_nodes      = self.dataloadermanager.dataorchestrator.data_context.num_nodes
        _horizon_size   = self.dataloadermanager.dataorchestrator.config.horizon_size
        _seq_length     = self.dataloadermanager.dataorchestrator.config.sequence_length

        self.model = SimpleGCNModule(
            num_features    = _num_features,
            num_nodes       = _num_nodes,
            hidden_size     = hidden_size,
            emb_size= emb_size,
            num_layers      = num_layers,
            dropout         = dropout,
            horizon_size    = _horizon_size,
            seq_length      =_seq_length
    
        ).to(self.device)

        model_hparams_config = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        }

        self.config_info['model_hparams'] = model_hparams_config
        self._update_status('model_hparams_set')
        return self