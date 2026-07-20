import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
from typing import Optional, Tuple, Literal

from ..strategies.gcn import StandardGNNStrategy
from ..debugging import ModelDebuggingReport, DebuggingLine
from ..deepmodel import DeepModel
from ....dataloading.databuilders.deepdataloaders.graphdataloader import GraphDataLoaderManager


class DecoupledGCNLayer(nn.Module):
    """
    A single GCN layer that keeps self-information and neighbour-information
    completely separate, then combines them via a learned per-node gate.

    Standard GCNConv with self-loops computes:
        h_i = W * mean(h_j for j in N(i) ∪ {i})
    which irreversibly mixes self and neighbour signals before any
    nonlinearity. This layer instead computes:
        h_self_i  = W_s * h_i                                    (own representation)
        h_neigh_i = GCNConv_no_loops * h_i                       (neighbour aggregation)
        alpha_i   = sigmoi
        d(gate([h_self_i; h_neigh_i; emb_i]))  (per-node gate)
        h_i       = alpha_i * h_self_i + (1 - alpha_i) * h_neigh_i

    The node embedding emb_i is critical: without it, the gate sees nearly
    identical inputs for every node (hidden representations converge in norm
    during training) and collapses to a uniform alpha. The embedding gives
    each node a fixed learned identity that the gate can use to produce
    genuinely node-specific mixing coefficients.

    Parameters
    ----------
    in_dim      : int   — input feature dimension
    out_dim     : int   — output feature dimension
    num_nodes   : int   — number of nodes (for embedding table)
    emb_dim     : int   — dimension of node embedding fed to gate
    norm_edges  : bool  — whether neighbour GCNConv normalises edge weights
    """

    def __init__(self,
                 in_dim:        int,
                 out_dim:       int,
                 num_nodes:     int,
                 emb_dim:       int,
                 norm_edges:    bool):
        super().__init__()

        self.num_nodes = num_nodes

        # Self-branch: plain linear projection, no graph involved
        self.self_proj = nn.Linear(in_dim, out_dim, bias=True)

        # Neighbour-branch: GCNConv without self-loops
        # normalize=norm_edges applies D^{-1/2} A D^{-1/2} weighting
        self.neigh_conv = GCNConv(
            in_dim,
            out_dim,
            add_self_loops  = False,    # deliberately excluded — self handled separately
            normalize       = norm_edges,
            bias            = True
        )

        # Per-node embedding: gives the gate node-specific identity so it can
        # learn different alpha values for different nodes even when their
        # hidden representations are similar in norm
        self.node_emb = nn.Embedding(num_nodes, emb_dim)

        # Gate: maps [h_self; h_neigh; emb] → scalar per node
        # sigmoid output gives alpha ∈ (0, 1)
        # alpha → 1 means rely on self, alpha → 0 means rely on neighbours
        self.gate = nn.Linear(out_dim * 2 + emb_dim, 1, bias=True)

        # Initialise gate bias slightly positive so alpha starts near 0.73
        # (sigmoid(1) ≈ 0.73), biasing toward self-information initially.
        # This mirrors the identity graph result — start from a reasonable
        # prior and let the data move alpha toward neighbours if warranted.
        nn.init.constant_(self.gate.bias, 1.0)

    def forward(self,
                h:              Tensor,
                edge_index:     Tensor,
                edge_weight:    Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        h            : [num_nodes, in_dim]
        edge_index   : [2, num_edges]
        edge_weight  : [num_edges] or None

        Returns
        -------
        h_out  : [num_nodes, out_dim]
        alpha  : [num_nodes, 1]  — gate values for inspection
        """

        h_self  = self.self_proj(h)                             # [num_nodes, out_dim]
        h_neigh = self.neigh_conv(h, edge_index, edge_weight)   # [num_nodes, out_dim]

        node_ids    = torch.arange(self.num_nodes, device=h.device)
        emb         = self.node_emb(node_ids)                   # [num_nodes, emb_dim]

        gate_input  = torch.cat([h_self, h_neigh, emb], dim=-1) # [num_nodes, out_dim*2 + emb_dim]
        alpha       = torch.sigmoid(self.gate(gate_input))      # [num_nodes, 1]

        h_out = alpha * h_self + (1 - alpha) * h_neigh          # [num_nodes, out_dim]

        return h_out, alpha


class DecoupledGCNModule(nn.Module):
    """
    Spatially dominant GCN using decoupled self/neighbour layers.

    Identical to SpatialDominantGCNModule in overall structure — temporal
    information is flattened into features, spatial convolution dominates —
    but each GCN layer is replaced by a DecoupledGCNLayer that separates
    self-information from neighbour aggregation and combines them via a
    learned per-node gate.

    After training, gate values (alpha per node per layer) are inspectable
    via get_gate_values(). Nodes where alpha is consistently low are genuinely
    using neighbour information. Correlating these with persistence CCC tests
    the hypothesis that hard nodes benefit from spatial information.

    Parameters
    ----------
    num_features  : int
    num_nodes     : int
    seq_length    : int
    hidden_size   : int
    num_layers    : int   — number of DecoupledGCNLayer stacks
    dropout       : float
    horizon_size  : int
    num_quantiles : int
    emb_dim       : int   — node embedding dimension fed to each layer's gate
    norm_edges    : bool
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
                 emb_dim:       int,
                 norm_edges:    bool):

        super().__init__()

        self.num_features   = num_features
        self.num_nodes      = num_nodes
        self.seq_length     = seq_length
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        self.horizon_size   = horizon_size
        self.num_quantiles  = num_quantiles

        flat_features = num_features * seq_length

        # Input projection: flattened temporal features → hidden space
        self.temporal_encoder = nn.GRU(
            input_size   = num_features,
            hidden_size  = hidden_size,
            num_layers   = 2,
            batch_first  = False,
            dropout      = dropout,
        )
        self.input_norm = nn.LayerNorm(hidden_size)

        # Stack of decoupled layers — each layer has its own node embedding
        # table so each layer can learn independent gating behaviour
        self.layers = nn.ModuleList([
            DecoupledGCNLayer(hidden_size, hidden_size, num_nodes, emb_dim, norm_edges)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon_size * num_quantiles)
        )

        # Storage for gate values — populated during forward, readable after
        # via get_gate_values(). List of [num_nodes, 1] tensors, one per layer.
        self._last_alphas: list[Tensor] = []

    def forward(self,
                x:              Tensor,
                edge_index:     Tensor,
                edge_weight:    Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        x            : [num_nodes, num_features, seq_len]
        edge_index   : [2, num_edges]
        edge_weight  : [num_edges] or None

        Returns
        -------
        output : [num_nodes, horizon_size, num_quantiles]
        """

        # Flatten temporal dimension into features
        x_seq = x.permute(2, 0, 1)              # [seq_len, num_nodes, num_features]
        _, h_n = self.temporal_encoder(x_seq)   # h_n: [num_gru_layers, num_nodes, hidden_size]
        h = h_n[-1]                             # take last layer: [num_nodes, hidden_size]
        h = self.input_norm(h)
        h = F.relu(h)
        h = self.dropout(h)

        # Decoupled spatial layers with residual connections
        self._last_alphas = []
        for layer, norm in zip(self.layers, self.norms):
            h_res           = h
            h, alpha        = layer(h, edge_index, edge_weight)
            h               = norm(h)
            h               = F.relu(h)
            h               = self.dropout(h)
            h               = h + h_res
            self._last_alphas.append(alpha.detach())

        # Output projection
        # Always returns [num_nodes, horizon_size, num_quantiles] — no squeeze
        # when num_quantiles == 1, consistent with other models in this codebase
        output = self.output_proj(h)                # [num_nodes, horizon_size * num_quantiles]
        output = output.view(self.num_nodes, self.horizon_size, self.num_quantiles)

        return output

    def get_gate_values(self) -> list[Tensor]:
        """
        Return the gate values (alpha) from the most recent forward pass.

        Returns
        -------
        List of [num_nodes, 1] tensors, one per layer.
        alpha → 1.0 means the node relied on its own representation.
        alpha → 0.0 means the node relied on neighbour aggregation.

        Usage
        -----
        model.model.eval()
        with torch.no_grad():
            model.model(x, edge_index, edge_weight)
        alphas     = model.model.get_gate_values()
        mean_alpha = torch.stack(alphas).mean(dim=0).squeeze()  # [num_nodes]
        """
        return self._last_alphas

    def debug(self,
              x:            Tensor,
              edge_index:   Tensor,
              edge_weight:  Optional[Tensor] = None) -> Tuple[Tensor, ModelDebuggingReport]:

        h = x.reshape(self.num_nodes, -1)

        dbl_input = DebuggingLine(
            list(h.shape),
            [self.num_nodes, self.num_features * self.seq_length]
        )

        h = F.relu(self.input_norm(self.input_proj(h)))
        h = self.dropout(h)

        dbl_proj = DebuggingLine(list(h.shape), [self.num_nodes, self.hidden_size])

        for layer, norm in zip(self.layers, self.norms):
            h_res   = h
            h, _    = layer(h, edge_index, edge_weight)
            h       = norm(h)
            h       = F.relu(h)
            h       = self.dropout(h)
            h       = h + h_res

        dbl_gcn = DebuggingLine(list(h.shape), [self.num_nodes, self.hidden_size])

        output = self.output_proj(h).view(self.num_nodes, self.horizon_size, self.num_quantiles)

        dbl_out = DebuggingLine(
            list(output.shape),
            [self.num_nodes, self.horizon_size, self.num_quantiles]
        )

        report = ModelDebuggingReport([dbl_input, dbl_proj, dbl_gcn, dbl_out])

        return output, report


class DecoupledGCNModel(DeepModel):
    """
    Spatially dominant GCN with decoupled self/neighbour gating.

    Each graph convolution layer explicitly separates the node's own
    representation from its neighbours' aggregated representation, then
    combines them via a learned per-node sigmoid gate. A node embedding is
    concatenated to the gate input so each node can learn a distinct mixing
    coefficient independent of whether its hidden representation happens to
    be similar to its neighbours'.

    The key diagnostic is get_gate_values() on the underlying module after
    forecasting. Nodes where alpha is consistently low are genuinely using
    neighbour information. Correlating alpha with persistence CCC per node
    tests whether hard nodes are the ones benefiting from spatial information.

    Usage
    -----
    model = DecoupledGCNModel(dataloadermanager)
    model.set_model_hparams(hidden_size=64, num_layers=3, dropout=0.2,
                            emb_dim=8, norm_edges=True)
    model.set_global_hparams(lr=1e-3, n_epochs=50, loss='mse')
    model.train()
    model.forecast('test')

    # Inspect gate values after forecasting
    alphas     = model.model.get_gate_values()
    mean_alpha = torch.stack(alphas).mean(dim=0).squeeze()  # [num_nodes]
    # low alpha  → node relies on neighbours
    # high alpha → node relies on self
    """
    _expected_dataloadermanager = 'GraphDataLoaderManager'
    def __init__(self,
                 dataloadermanager: GraphDataLoaderManager,
                 name:              str           = 'DecoupledGCN',
                 verbose:           Literal[-1, 0, 1, 2]   = -1):

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
                          emb_dim:      int   = 8,
                          norm_edges:   bool  = True):
        """
        Parameters
        ----------
        hidden_size : int
            Hidden dimension throughout. Output projection bottleneck is
            hidden_size // 2.
        num_layers : int
            Number of DecoupledGCNLayers. Each layer has its own node
            embedding table so each learns independent gating behaviour.
            2-4 recommended.
        dropout : float
            Applied after each layer and inside the output projection.
        emb_dim : int
            Dimension of the node embedding concatenated to the gate input.
            Larger values give the gate more node-specific signal but add
            parameters. 8-16 is sufficient for NUTS3-scale graphs.
        norm_edges : bool
            Whether the neighbour GCNConv applies symmetric normalisation.
            No self_loops parameter — self-loops are always handled by
            the self_proj branch, never by the neighbour conv.

        Returns
        -------
        self — for method chaining
        """

        _num_features   = len(self.column_registration.get_by_type('feature'))
        _num_nodes      = len(self.dataloadermanager.dataorchestrator.data_context.local_shapedata)
        _horizon_size   = self.dataloadermanager.dataorchestrator.config.horizon_size
        _seq_length     = self.dataloadermanager.dataorchestrator.config.sequence_length
        _num_quantiles  = max(self.dataloadermanager.dataorchestrator.config._num_quantiles, 1)

        self.model = DecoupledGCNModule(
            num_features    = _num_features,
            num_nodes       = _num_nodes,
            seq_length      = _seq_length,
            hidden_size     = hidden_size,
            num_layers      = num_layers,
            dropout         = dropout,
            horizon_size    = _horizon_size,
            num_quantiles   = _num_quantiles,
            emb_dim         = emb_dim,
            norm_edges      = norm_edges,
        ).to(self.device)

        self.config_info['model_hparams'] = {
            'hidden_size':  hidden_size,
            'num_layers':   num_layers,
            'dropout':      dropout,
            'emb_dim':      emb_dim,
            'norm_edges':   norm_edges,
        }

        self._update_status('model_hparams_set')