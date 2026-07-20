import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from typing import Optional

from ..strategies.gcn import StandardGNNStrategy
from ..deepmodel import DeepModel
from ....dataloading import GraphDataBuilder

class GATModule(nn.Module):
    """
    GCN - Module

    Parameters
    ----------
    hidden_size: int
        hidden_size of the linear layer
    num_layers: int
        number of GCN-layers
    num_heads: int
        number of attention heads
    dropout_p: float
        dropout rate
    self_loops: bool
        whether to include self loops in MPNN
    norm_edges: bool
        whether to normalize edge weights    
    residuals: bool
        whether to keep skip connections

    num_features: int
        number of features in dataloader-snapshot
    num_nodes: int
        number of nodes in graph structure
    seq_length: int
        number of sequences in dataloader-snapshot
    horizon_size: int
        number of steps to predict
    num_quantiles: int
        number of quantiles to predict

    Forward
    -------

    The forward pass is divided into the following parts:
    - input projection
        - temporal-flattening
        - linear-layer
        - layer-norm
        - relu-activation
    - convolutional-layers
        - GCNConv
        - layer-norm
        - relu-activation
        - dropout
        - optional residual connection
    - output projection
        - linear-layer
    """
    def __init__(self,
                 hidden_size:   int,
                 num_layers:    int,
                 num_heads:     int,
                 dropout_p:     float,
                 self_loops:    bool,
                 norm_edges:    bool,
                 residuals:     bool,
                 num_features:  int,
                 num_nodes:     int,
                 seq_length:    int,
                 horizon_size:  int,
                 num_quantiles: int):

        super().__init__()

        ### set model - params ###
        self.hidden_size    = hidden_size 
        self.num_layers     = num_layers 
        self.num_heads      = num_heads
        self.dropout_p      = dropout_p 
        self.self_loops     = self_loops
        self.norm_edges     = norm_edges
        self.residuals      = residuals

        ### get data - params ###
        self.num_features   = num_features
        self.num_nodes      = num_nodes
        self.seq_length     = seq_length
        self.horizon_size   = horizon_size
        self.num_quantiles  = num_quantiles

        flat_features = num_features * seq_length

        # input projection: [] -> []
        self.input_proj     = nn.Linear(flat_features, hidden_size)
        self.input_norm     = nn.LayerNorm(hidden_size)
        self.activation     = nn.ReLU()

        # Deep spatial stack with residual connections.
        # Each layer: GATv2Conv → LayerNorm → ReLU → Dropout + residual

        convs       = []
        norms       = []
        for layer in range(num_layers):
            convs.append(GATv2Conv(
                in_channels = hidden_size,
                out_channels = hidden_size,
                heads    =num_heads,
                concat=False,
                add_self_loops=self_loops
))
            norms.append(nn.LayerNorm(hidden_size))

        self.convs  = nn.ModuleList(convs)
        self.norms  = nn.ModuleList(norms)
        self.dropout= nn.Dropout(self.dropout_p)

        # Output projection: linear layer [hidden_size] → [horizon * quantiles]
        self.output_proj = nn.Linear(hidden_size, horizon_size * num_quantiles)

    def forward(self,
                x:              torch.Tensor,
                edge_index:     torch.Tensor,
                edge_weight:    Optional[torch.Tensor] = None) -> torch.Tensor:
        
        h:      torch.Tensor
        output: torch.Tensor

        # Flatten all timesteps into a single feature vector per node
        # [num_nodes, num_features, seq_len] → [num_nodes, num_features * seq_len]
        h = x.reshape(self.num_nodes, -1)

        # Project to hidden dim
        h = self.input_proj(h)              # [num_nodes, hidden_size]
        h = self.input_norm(h)
        h = self.activation(h)

        # Deep spatial convolution with residual connections
        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_res   = h                                     # store for residual
            h       = conv(h, edge_index)                   # spatial aggregation
            h       = norm(h) 
            h       = self.activation(h)
            h       = self.dropout(h)

            # residual: preserves node-specific signal
            if self.residuals:
                h       = h + h_res                 

        h_out = h

        # Project to forecasts
        output = self.output_proj(h_out)                                                # [num_nodes, horizon_size * num_quantiles]
        output = output.view(self.num_nodes, self.horizon_size, self.num_quantiles)     # [num_nodes, horizon_size,  num_quantiles] 

        return output

class GATModel(DeepModel):
    """
    """
    _expected_dataloadermanager = 'GraphDataBuilder'
    def __init__(self,
                 dataloadermanager: GraphDataBuilder,
                 name:              str           = 'gatmodel',
                 num_quantiles:     int = 1):

        super().__init__(
            dataloadermanager   = dataloadermanager,
            name                = name,
            verbose             = 2,
            strategy            = StandardGNNStrategy()
        )
        self.num_quantiles = num_quantiles

    def set_model_hparams(self,
                          hidden_size:  int   = 64,
                          num_layers:   int   = 3,
                          num_heads:    int   = 4,
                          dropout:      float = 0.2,
                          self_loops:   bool  = False,
                          norm_edges:   bool  = True,
                          residuals:    bool  = True):
        """
        """
        _num_features   = len(self.column_registration.get_entries_names_by_type('feature'))
        _num_nodes      = len(self.dataloadermanager.dataorchestrator.data_context.local_shapedata)
        _horizon_size   = self.dataloadermanager.dataorchestrator.config.horizon_size
        _seq_length     = self.dataloadermanager.dataorchestrator.config.sequence_length
        _num_quantiles  = self.num_quantiles

        self.model = GATModule(
            hidden_size     = hidden_size,
            num_layers      = num_layers,
            num_heads       = num_heads,
            dropout_p       = dropout,
            self_loops      = self_loops,
            norm_edges      = norm_edges,
            residuals       = residuals,

            num_features    = _num_features,
            num_nodes       = _num_nodes,
            seq_length      = _seq_length,
            horizon_size    = _horizon_size,
            num_quantiles   = _num_quantiles
        ).to(self.device)

        self.config_info['model_hparams'] = {
            'hidden_size':  hidden_size,
            'num_layers':   num_layers,
            'num_heads' :   num_heads,
            'dropout'   :   dropout,
            'self_loops':   self_loops,
            'norm_edges':   norm_edges,
            'residuals' :   residuals
        }

        self._update_status('model_hparams_set')
