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


class SpatialDominantGCNModule(nn.Module):
    """
    Spatially dominant GCN with no temporal unit.

    Design rationale
    ----------------
    The key difference from SimpleGCNModule is that temporal information
    is NOT collapsed to a single timestep before the GCN. Instead:

    1. All seq_len timesteps are flattened into a wide feature vector per
       node: [nodes, features * seq_len]. This preserves all temporal
       information but treats it as unstructured — no LSTM, no TCN, no
       learned temporal ordering.

    2. A linear input projection maps this to the hidden dimension.

    3. Multiple GCN layers with residual connections then repeatedly
       propagate and refine representations spatially. Residuals are
       critical here: without them, stacking 3+ GCN layers causes
       over-smoothing where all nodes converge to the graph mean,
       destroying any node-specific signal.

    4. The output projection maps the final spatial representation to
       forecasts.

    The consequence of this design is that the graph is the *only*
    structured inductive bias. There is no temporal unit that can absorb
    the predictive burden and render the graph irrelevant. If graph
    structure matters at all in the data, it will show up here.

    Parameters
    ----------
    num_features  : int   — input features per node per timestep
    num_nodes     : int   — number of nodes in the graph
    seq_length    : int   — number of input timesteps (flattened into features)
    hidden_size   : int   — hidden dimension throughout the GCN stack
    num_layers    : int   — number of GCN layers (2-4 recommended; >4 risks over-smoothing
                            even with residuals)
    dropout       : float — applied after each GCN layer
    horizon_size  : int   — number of forecast steps
    num_quantiles : int   — number of output quantiles (1 = point forecast)
    self_loops    : bool  — whether GCNConv adds self-loops (if False, a node's
                            own features only enter through the residual connection)
    norm_edges    : bool  — whether GCNConv applies symmetric normalisation
                            (D^{-1/2} A D^{-1/2}); recommended True when edge
                            weights vary in scale across graphs
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

        flat_features = num_features * seq_length

        # Project flattened temporal features into hidden space.
        # Simple linear — no temporal structure imposed.
        self.input_proj = nn.Linear(flat_features, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        # Deep spatial stack with residual connections.
        # Each layer: GCNConv → LayerNorm → ReLU → Dropout + residual
        self.convs = nn.ModuleList([
            GCNConv(
                hidden_size,
                hidden_size,
                add_self_loops  = self_loops,
                normalize       = norm_edges
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Output projection: hidden → horizon * quantiles
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
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


class GCN2Model(DeepModel):
    """
    Spatially dominant GCN model — no temporal unit.

    The central design choice is that no LSTM, TCN, or attention unit is
    present to absorb temporal dynamics. All input timesteps are flattened
    into features and the graph convolution stack is the sole source of
    structured inductive bias. This forces the graph to matter if it
    contains any predictive signal.

    Use this model to test whether graph structure is genuinely useful
    when the temporal escape hatch is closed. If geographic or commuter
    graphs outperform the identity graph here but not in GCNLSTM or
    GATv2LSTM, the conclusion is that spatial signal exists in the data
    but the temporal unit learns to ignore it.

    Usage
    -----
    model = SpatialDominantGCNModel(dataloadermanager)
    model.set_model_hparams(hidden_size=64, num_layers=3, dropout=0.2,
                            self_loops=False, norm_edges=True)
    model.set_global_hparams(lr=1e-3, n_epochs=50, loss='mse')
    model.train()
    model.forecast('test')
    """

    def __init__(self,
                 dataloadermanager: GraphDataLoaderManager,
                 name:              Optional[str]           = None,
                 verbose:           Literal[-1, 0, 1, 2]   = -1):

        if not name:
            name = 'SpatialDominantGCN'

        self._expected_dataloadermanager = 'GraphDataLoaderManager'

        super().__init__(
            dataloadermanager   = dataloadermanager,
            name                = name,
            verbose             = verbose,
            strategy            = StandardGNNStrategy()
        )

    def set_model_hparams(self,
                          hidden_size:  int   = 64,
                          num_layers:   int   = 3,
                          dropout:      float = 0.2,
                          self_loops:   bool  = False,
                          norm_edges:   bool  = True) -> 'GCN2Model':
        """
        Set model hyperparameters and instantiate the architecture.

        Parameters
        ----------
        hidden_size : int
            Hidden dimension used throughout the GCN stack and input
            projection. Also sets the bottleneck of the output projection
            (hidden_size // 2).
        num_layers : int
            Number of GCN layers. Each layer applies one round of
            neighbourhood aggregation. 2-4 is sensible; beyond 4, residuals
            help but over-smoothing risk increases.
        dropout : float
            Dropout rate applied after each GCN layer and inside the
            output projection.
        self_loops : bool
            Whether GCNConv adds self-loops before aggregation. If False,
            a node's own features enter the next layer only via the
            residual connection — useful for testing whether the model
            relies on graph neighbours vs. self-information.
        norm_edges : bool
            Whether GCNConv applies symmetric normalisation
            (D^{-1/2} A D^{-1/2}). Recommended True when comparing graphs
            with different edge weight scales (e.g. commuter flows vs.
            geographic distances).

        Returns
        -------
        self — for method chaining
        """

        _num_features   = len(self.column_registration.get_by_type('feature'))
        _num_nodes      = len(self.dataloadermanager.dataorchestrator.data_context.local_shapedata)
        _horizon_size   = self.dataloadermanager.dataorchestrator.config.horizon_size
        _seq_length     = self.dataloadermanager.dataorchestrator.config.sequence_length
        _num_quantiles  = max(self.dataloadermanager.dataorchestrator.config._num_quantiles, 1)

        self.model = SpatialDominantGCNModule(
            num_features    = _num_features,
            num_nodes       = _num_nodes,
            seq_length      = _seq_length,
            hidden_size     = hidden_size,
            num_layers      = num_layers,
            dropout         = dropout,
            horizon_size    = _horizon_size,
            num_quantiles   = _num_quantiles,
            self_loops      = self_loops,
            norm_edges      = norm_edges,
        ).to(self.device)

        self.config_info['model_hparams'] = {
            'hidden_size':  hidden_size,
            'num_layers':   num_layers,
            'dropout':      dropout,
            'self_loops':   self_loops,
            'norm_edges':   norm_edges,
        }

        self._update_status('model_hparams_set')
        return self