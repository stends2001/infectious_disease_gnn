import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
from typing import Optional, Tuple, Literal

from ..strategies.gcn import StandardGNNStrategy
from ..debugging import ModelDebuggingReport, DebuggingLine
from ..deepmodel import DeepModel
from ....dataloading.dataloaders.deepdataloaders.graphdataloader import GraphDataLoaderManager


class SpatialGCNModule(nn.Module):
    """
    NOTE
    ----
    Updates w.r.t. GCN2:
    - dropout in final layer has been removed

    NOTE
    ----
    By intention, temporal ordering is destoryed by flattening the feature vector.
    """

    def __init__(self,
                 num_features:  int,
                 num_nodes:     int,
                 seq_length:    int,
                 hidden_size:   int,
                 num_layers:    int,
                 dropout:       float,
                 horizon_size:  int,
                 num_quantiles: int,
                 self_loops:    bool,
                 norm_edges:    bool):

        super().__init__()

        self.num_features   = num_features
        self.num_nodes      = num_nodes
        self.seq_length     = seq_length
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        self.horizon_size   = horizon_size
        self.num_quantiles  = num_quantiles
        self.dropout_rate   = dropout

        # flatten features -> destroy temporal ordering
        num_flat_features = num_features * seq_length

        # input projection -> linear layer
        self.input_proj = nn.Linear(num_flat_features, hidden_size)

        # input normalization -> layer normalization
        self.input_norm = nn.LayerNorm(hidden_size)

        ##########################
        ### deep spatial stack ###
        ##########################        
        # Deep spatial stack with residual connections.
        # Each layer: GCNConv → LayerNorm → ReLU → Dropout + residual

        # 1. GCNConv - layers
        self.convs = nn.ModuleList([
            GCNConv(
                hidden_size,
                hidden_size,
                add_self_loops  = self_loops,
                normalize       = norm_edges
            )
            for _ in range(num_layers)
            ])

        # 2. LayerNorm - layers
        self.norms = nn.ModuleList([
            nn.LayerNorm(
                hidden_size
                )
            for _ in range(num_layers)
        ])

        # 3. Dropout - layers
        self.dropout = nn.Dropout(dropout)

        # Output projection: hidden → horizon * quantiles
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, horizon_size * num_quantiles)
        )

    def forward(self,
                x:              torch.Tensor,
                edge_index:     torch.Tensor,
                edge_weight:    Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x            : [num_nodes, num_features, seq_len]
        edge_index   : [2, num_edges]
        edge_weight  : [num_edges] or None

        Returns
        -------
        output : [num_nodes, horizon_size, num_quantiles]
                 or [num_nodes, horizon_size] when num_quantiles == 1
        """

        # Flatten all timesteps into a single feature vector per node
        # [num_nodes, num_features, seq_len] → [num_nodes, num_features * seq_len]
        h = x.reshape(self.num_nodes, -1)

        # Project to hidden dim
        h = self.input_proj(h)              # [num_nodes, hidden_size]
        h = self.input_norm(h)
        h = F.relu(h)
        h = self.dropout(h)

        # Deep spatial convolution with residual connections
        for conv, norm in zip(self.convs, self.norms):
            h_res   = h                                 # store for residual
            h       = conv(h, edge_index, edge_weight)  # spatial aggregation
            h       = norm(h)
            h       = F.relu(h)
            h       = self.dropout(h)
            h       = h + h_res                         # residual: preserves node-specific signal

        # Project to forecasts
        output = self.output_proj(h)                    # [num_nodes, horizon_size * num_quantiles]
        output = output.view(self.num_nodes, self.horizon_size, self.num_quantiles)   

        return output

    def debug(self,
              x:            Tensor,
              edge_index:   torch.Tensor,
              edge_weight:  Optional[torch.Tensor] = None) -> Tuple[Tensor, ModelDebuggingReport]:

        h = x.reshape(self.num_nodes, -1)

        dbl_input = DebuggingLine(
            list(h.shape),
            [self.num_nodes, self.num_features * self.seq_length]
        )

        h = F.relu(self.input_norm(self.input_proj(h)))
        h = self.dropout(h)

        dbl_proj = DebuggingLine(list(h.shape), [self.num_nodes, self.hidden_size])

        for conv, norm in zip(self.convs, self.norms):
            h_res   = h
            h       = conv(h, edge_index, edge_weight)
            h       = norm(h)
            h       = F.relu(h)
            h       = self.dropout(h)
            h       = h + h_res

        dbl_gcn = DebuggingLine(list(h.shape), [self.num_nodes, self.hidden_size])

        output = self.output_proj(h).view(self.num_nodes, self.horizon_size, self.num_quantiles)

        if self.num_quantiles == 1:
            output = output[:, :, 0]

        dbl_out = DebuggingLine(list(output.shape), [self.num_nodes, self.horizon_size])

        report = ModelDebuggingReport([dbl_input, dbl_proj, dbl_gcn, dbl_out])

        return output, report
