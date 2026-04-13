import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
from typing import Optional, Tuple, Literal

from ..strategies.gatv2lstm import StatelessGATv2LSTMStrategy
from ..debugging import ModelDebuggingReport, DebuggingLine
from ..deepmodel import DeepModel
from ....dataloading.dataloaders.deepdataloaders.graphdataloader import GraphDataLoaderManager


class GCNLSTMArchitecture(nn.Module):
    """
    Spatial GCN followed by a temporal LSTM.

    Design rationale
    ----------------
    - GCN aggregates neighbourhood information at each timestep independently.
      No attention — edge weights from the graph constructor are used directly
      via GCN's symmetric normalisation, so commuter/geographic weights
      actually matter here (unlike GATv2 which re-learns its own weights).
    - LSTM then models temporal dynamics across the sequence of
      graph-enriched node representations.
    - LayerNorm after LSTM hidden state stabilises training.
    - Single linear output head — no hidden bottleneck, avoids extra
      parameters on an already-small feature space.

    Parameters
    ----------
    num_features    : int   — number of input features per node per timestep
    num_nodes       : int   — number of nodes in the graph
    gcn_hidden_dim  : int   — output dim of each GCN layer
    gcn_layers      : int   — number of stacked GCN layers (keep at 1-2 to
                              avoid over-smoothing on geographic graphs)
    lstm_hidden_size: int   — LSTM hidden state size
    lstm_layers     : int   — number of stacked LSTM layers
    dropout         : float — applied after each GCN layer and inside LSTM
    horizon_size    : int   — number of forecast steps
    seq_length      : int   — input sequence length
    """

    def __init__(self,
                 num_features:      int,
                 num_nodes:         int,
                 gcn_hidden_dim:    int,
                 gcn_layers:        int,
                 lstm_hidden_size:  int,
                 lstm_layers:       int,
                 dropout:           float,
                 horizon_size:      int,
                 seq_length:        int,
                 num_quantiles:     int,
                 self_loops:        bool,
                 norm_edges:        bool):

        super().__init__()

        self.num_features       = num_features
        self.num_nodes          = num_nodes
        self.gcn_hidden_dim     = gcn_hidden_dim
        self.gcn_layers         = gcn_layers
        self.lstm_hidden_size   = lstm_hidden_size
        self.lstm_layers        = lstm_layers
        self.dropout            = dropout
        self.horizon_size       = horizon_size
        self.seq_length         = seq_length
        self.num_quantiles      = num_quantiles

        self.self_loops     = self_loops
        self.norm_edges     = norm_edges        

        # --- GCN stack ---
        # First layer: raw features -> gcn_hidden_dim
        # Subsequent layers: gcn_hidden_dim -> gcn_hidden_dim
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, gcn_hidden_dim, add_self_loops = self_loops, normalize=norm_edges))
        for _ in range(gcn_layers - 1):
            self.convs.append(GCNConv(gcn_hidden_dim, gcn_hidden_dim, add_self_loops = self_loops, normalize=norm_edges))

        # BatchNorm per GCN layer — stabilises training when edge weights
        # vary a lot (commuter graph has high weight variance)
        self.gcn_norms = nn.ModuleList([
            nn.LayerNorm(gcn_hidden_dim) for _ in range(gcn_layers)
        ])

        # --- Temporal LSTM ---
        # input: gcn_hidden_dim per timestep
        # dropout only applied between stacked LSTM layers, not on last layer
        self.lstm = nn.LSTM(
            input_size  = gcn_hidden_dim,
            hidden_size = lstm_hidden_size,
            num_layers  = lstm_layers,
            dropout     = dropout if lstm_layers > 1 else 0.0,
            batch_first = False  # expects [seq_len, num_nodes, features]
        )

        self.lstm_out_norm = nn.LayerNorm(lstm_hidden_size)

        # Dropout applied to LSTM output before projection
        self.out_dropout = nn.Dropout(dropout)

        # --- Output projection ---
        self.fc = nn.Linear(lstm_hidden_size, horizon_size * self.num_quantiles)

    def forward(self,
                x:              torch.Tensor,
                edge_index:     torch.Tensor,
                edge_weight:    Optional[torch.Tensor] = None,
                hidden_state:   Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x           : [num_nodes, num_features, seq_len]
        edge_index  : [2, num_edges]
        edge_weight : [num_edges]  — used directly by GCNConv
        hidden_state: optional (h, c) tuple from previous step

        Returns
        -------
        output  : [num_nodes, horizon_size]
        h       : [lstm_layers, num_nodes, lstm_hidden_size]
        c       : [lstm_layers, num_nodes, lstm_hidden_size]
        """
        seq_outputs = []

        # Apply GCN stack at each timestep independently
        for t in range(x.shape[2]):
            h_t = x[:, :, t]   # [num_nodes, num_features]

            for conv, norm in zip(self.convs, self.gcn_norms):
                h_t = conv(h_t, edge_index, edge_weight)
                h_t = norm(h_t)
                h_t = F.relu(h_t)
                h_t = F.dropout(h_t, p=self.dropout, training=self.training)

            seq_outputs.append(h_t)

        # [seq_len, num_nodes, gcn_hidden_dim]
        x_seq = torch.stack(seq_outputs, dim=0)

        # LSTM over time
        if hidden_state is None:
            lstm_out, (h, c) = self.lstm(x_seq)
        else:
            lstm_out, (h, c) = self.lstm(x_seq, hidden_state)

        # Take last layer's hidden state: [num_nodes, lstm_hidden_size]
        ht = h[-1]
        ht = self.lstm_out_norm(ht)
        ht = self.out_dropout(ht)

        output = self.fc(ht)


        output = output.view(self.num_nodes, self.horizon_size, self.num_quantiles)

        return output, h, c
        

class GCNLSTMModel(DeepModel):
    """
    GCN + LSTM model for spatiotemporal epidemic forecasting.

    Compared to GATv2LSTM
    ---------------------
    - Uses GCNConv instead of GATv2Conv: no learned attention weights,
      so pre-computed edge weights (commuter flows, geographic distances)
      are used directly. This makes graph structure differences more
      interpretable.
    - BatchNorm after each GCN layer rather than just LayerNorm at the end:
      helps when edge weight distributions differ across graph types.
    - Simpler — fewer hyperparameters to tune, less likely to overfit
      on a small feature set.

    Usage
    -----
    model = GCNLSTMModel(graphdataloader, name='gcnlstm_ig_1')
    model.set_model_hparams()
    model.set_global_hparams(**global_hparams)
    model.train()
    model.forecast()
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
            strategy            = StatelessGATv2LSTMStrategy()
        )

    def set_model_hparams(self,
                          gcn_hidden_dim:    int   = 64,
                          gcn_layers:        int   = 1,
                          lstm_hidden_size:  int   = 64,
                          lstm_layers:       int   = 1,
                          dropout:           float = 0.2,
                          self_loops:       bool = False,
                          norm_edges:       bool = False):
        """
        Parameters
        ----------
        gcn_hidden_dim   : output dimension of each GCN layer.
                           64 is a reasonable default for ~5 features.
                           Don't go above 128 — you'll overfit.
        gcn_layers       : 1 is recommended to start. 2 risks over-smoothing
                           on dense graphs (commuter has 3905 edges).
        lstm_hidden_size : LSTM hidden state size. Match to gcn_hidden_dim
                           to avoid an information bottleneck or explosion
                           at the GCN->LSTM boundary.
        lstm_layers      : 1 is almost always enough for seq_length=4.
                           2 adds parameters without much benefit here.
        dropout          : 0.2-0.3. Higher if you see the train/val gap
                           widen early in training.
        """
        _num_features   = len(self.column_registration.get_by_type('feature'))
        _num_nodes      = self.dataloadermanager.dataorchestrator.data_context.num_nodes
        _horizon_size   = self.dataloadermanager.dataorchestrator.config.horizon_size
        _seq_length     = self.dataloadermanager.dataorchestrator.config.sequence_length
        _num_quantiles  = max(self.dataloadermanager.dataorchestrator.config._num_quantiles,1)

        self.model = GCNLSTMArchitecture(
            num_features        = _num_features,
            num_nodes           = _num_nodes,
            gcn_hidden_dim      = gcn_hidden_dim,
            gcn_layers          = gcn_layers,
            lstm_hidden_size    = lstm_hidden_size,
            lstm_layers         = lstm_layers,
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
            'lstm_hidden_size': lstm_hidden_size,
            'lstm_layers':      lstm_layers,
            'dropout':          dropout,
            'self_loops':       self_loops,
            'norm_edges':       norm_edges
        }

        self._update_status('model_hparams_set')
