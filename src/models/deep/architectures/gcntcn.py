import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional
from typing import Literal

from ..strategies.gcn import StandardGNNStrategy
from ..debugging import ModelDebuggingReport, DebuggingLine
from ..deepmodel import DeepModel
from ....dataloading.dataloaders.deepdataloaders.graphdataloader import GraphDataLoaderManager

class GCNTCNArchitecture(nn.Module):

    def __init__(self,
                 num_features: int,
                 num_nodes: int,
                 gcn_hidden_dim: int,
                 gcn_layers: int,
                 tcn_hidden_dim: int,
                 tcn_layers: int,
                 kernel_size: int,
                 dropout: float,
                 horizon_size: int,
                 seq_length: int,
                 num_quantiles: int,
                 self_loops: bool,
                 norm_edges: bool):

        super().__init__()

        self.num_nodes = num_nodes
        self.horizon_size = horizon_size
        self.seq_length = seq_length
        self.num_quantiles = num_quantiles
        self.dropout = dropout

        # -------------------------
        # GCN stack
        # -------------------------
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(num_features, gcn_hidden_dim,
                    add_self_loops=self_loops,
                    normalize=norm_edges)
        )

        for _ in range(gcn_layers - 1):
            self.convs.append(
                GCNConv(gcn_hidden_dim, gcn_hidden_dim,
                        add_self_loops=self_loops,
                        normalize=norm_edges)
            )

        self.gcn_norms = nn.ModuleList([
            nn.LayerNorm(gcn_hidden_dim)
            for _ in range(gcn_layers)
        ])

        # -------------------------
        # Temporal Convolution Stack
        # -------------------------

        tcn_blocks = []
        in_channels = gcn_hidden_dim

        for _ in range(tcn_layers):
            tcn_blocks.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=tcn_hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size - 1
                )
            )
            tcn_blocks.append(nn.ReLU())
            tcn_blocks.append(nn.Dropout(dropout))
            in_channels = tcn_hidden_dim

        self.tcn = nn.Sequential(*tcn_blocks)
        self.temporal_norm = nn.LayerNorm(tcn_hidden_dim)

        # -------------------------
        # Output projection
        # -------------------------
        self.fc = nn.Linear(tcn_hidden_dim,
                            horizon_size * num_quantiles)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None):

        # x: [num_nodes, num_features, seq_len]

        seq_outputs = []

        # ---- Spatial encoding per timestep ----
        for t in range(x.shape[2]):
            h_t = x[:, :, t]  # [num_nodes, num_features]

            for conv, norm in zip(self.convs, self.gcn_norms):
                h_t = conv(h_t, edge_index, edge_weight)
                h_t = norm(h_t)
                h_t = F.relu(h_t)
                h_t = F.dropout(h_t, p=self.dropout,
                                training=self.training)

            seq_outputs.append(h_t)

        # [seq_len, num_nodes, gcn_hidden_dim]
        x_seq = torch.stack(seq_outputs, dim=0)

        # reshape for Conv1D:
        # [num_nodes, channels, seq_len]
        x_seq = x_seq.permute(1, 2, 0)

        # ---- Temporal convolution ----
        tcn_out = self.tcn(x_seq)

        # Remove extra padding tail
        tcn_out = tcn_out[:, :, :self.seq_length]

        # Take last timestep representation
        ht = tcn_out[:, :, -1]  # [num_nodes, tcn_hidden_dim]

        ht = self.temporal_norm(ht)

        output = self.fc(ht)

        output = output.view(self.num_nodes,
                             self.horizon_size,
                             self.num_quantiles)

        return output
    

class GCNTCNModel(DeepModel):
    """
    """

    def __init__(self,
                 dataloadermanager: GraphDataLoaderManager,
                 name:              Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):

        if not name:
            name = 'GCNLSTMModel'

        self._expected_dataloadermanager = 'GraphDataLoaderManager'

        # Reuse StatelessGATv2LSTMStrategy — it already handles
        # (x, edge_index, edge_weight, hidden) -> (y_hat, h, c)
        # which is exactly this model's forward signature
        super().__init__(
            dataloadermanager   = dataloadermanager,
            name                = name,
            verbose             = verbose,
            strategy            = StandardGNNStrategy()
        )

    def set_model_hparams(self,
                          gcn_hidden_dim:    int   = 64,
                          gcn_layers:        int   = 1,
                          tcn_hidden_dim:   int = 32,
                          tcn_layers: int = 2, 
                          kernel_size: int = 3,
                          dropout:           float = 0.2,
                          self_loops:       bool = False,
                          norm_edges:       bool = False):
        """
        Parameters
        ----------

        """
        _num_features   = len(self.column_registration.get_by_type('feature'))
        _num_nodes      = self.dataloadermanager.dataorchestrator.data_context.num_nodes
        _horizon_size   = self.dataloadermanager.dataorchestrator.config.horizon_size
        _seq_length     = self.dataloadermanager.dataorchestrator.config.sequence_length
        _num_quantiles  = max(self.dataloadermanager.dataorchestrator.config._num_quantiles,1)

        self.model = GCNTCNArchitecture(
            num_features        = _num_features,
            num_nodes           = _num_nodes,
            gcn_hidden_dim      = gcn_hidden_dim,
            gcn_layers          = gcn_layers,
            tcn_hidden_dim      = tcn_hidden_dim, 
            tcn_layers          = tcn_layers,
            kernel_size         = kernel_size,
            dropout             = dropout,
            horizon_size        = _horizon_size,
            seq_length          = _seq_length,
            num_quantiles       = _num_quantiles,
            self_loops      = self_loops,
            norm_edges      = norm_edges
        ).to(self.device)

        self.config_info['model_hparams'] = {
            'gcn_hidden_dim':   gcn_hidden_dim,
            'gcn_layers':       gcn_layers,
            'dropout':          dropout,
        }

        self._update_status('model_hparams_set')
