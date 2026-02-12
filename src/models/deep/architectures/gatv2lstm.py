import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Optional, Tuple, Literal
from torch_geometric.utils import add_self_loops

from ..strategies.gatv2lstm import StatelessGATv2LSTMStrategy, StatefullGATv2LSTMStrategy

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

        # ONLY ONE GAT layer to prevent over-smoothing
        self.gat1 = GATv2Conv(num_features, gat_hidden_dim, heads=num_heads, concat=True)

        # === Temporal LSTM ===
        self.lstm = nn.LSTM(
            input_size  = gat_hidden_dim * num_heads,
            hidden_size = hidden_size,
            num_layers  = temporal_layers,
            dropout     = dropout if temporal_layers > 1 else 0,
            batch_first = False
        )

        self.lstm_out_norm = nn.LayerNorm(hidden_size)

        # Final output layer
        self.fc = nn.Linear(hidden_size, horizon_size)

    def forward(self,
                x:              torch.Tensor,
                edge_index:     torch.Tensor,
                edge_weight:    Optional[torch.Tensor] = None,
                hidden_state:   Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        
        # x shape: [num_nodes, num_features, seq_len]
        
        # Apply GAT at EACH timestep (only one layer)
        seq_outputs = []
        for t in range(x.shape[2]):
            x_t = x[:, :, t]  # [num_nodes, num_features]
            h_gat = F.relu(self.gat1(x_t, edge_index))
            h_gat = F.dropout(h_gat, p=self.dropout, training=self.training)
            seq_outputs.append(h_gat)
        
        # Stack to get [seq_len, num_nodes, gat_hidden_dim * num_heads]
        x_seq = torch.stack(seq_outputs, dim=0)
        
        # Run LSTM on graph-enriched features
        if hidden_state is None:
            lstm_out, (h, c) = self.lstm(x_seq)
        else:
            lstm_out, (h, c) = self.lstm(x_seq, hidden_state)
        
        # Take last layer's hidden state
        ht_lstm = h[-1]  # [num_nodes, hidden_size]
        ht_lstm = self.lstm_out_norm(ht_lstm)

        # Final output
        output = self.fc(ht_lstm)  # [num_nodes, horizon_size]
        
        return output, h, c

    def debug(self,
              x:              torch.Tensor,
              edge_index:     torch.Tensor,
              edge_weight:    Optional[torch.Tensor] = None,
              hidden_state:   Optional[Tuple[torch.Tensor, torch.Tensor]] = None
              ):
        
        print("\n" + "="*80)
        print(" "*20 + "COMPREHENSIVE MODEL DEBUG")
        print("="*80)
        
        # ============= 1. EDGE INDEX DEBUGGING =============
        print("\n[1] EDGE INDEX ANALYSIS")
        print("-" * 80)
        print(f"  Edge index shape: {edge_index.shape}")
        print(f"  Edge index dtype: {edge_index.dtype}")
        print(f"  Edge index device: {edge_index.device}")
        print(f"  First 10 edges:\n{edge_index[:, :10]}")
        print(f"  Last 10 edges:\n{edge_index[:, -10:]}")
        print(f"  Unique source nodes: {edge_index[0].unique().numel()}/{self.num_nodes}")
        print(f"  Unique target nodes: {edge_index[1].unique().numel()}/{self.num_nodes}")
        print(f"  Total edges: {edge_index.shape[1]}")
        
        # Check if identity graph
        is_identity = (edge_index[0] == edge_index[1]).all()
        print(f"  Is identity graph (all self-loops)? {is_identity}")
        
        # Check for self-loops
        from torch_geometric.utils import contains_self_loops, remove_self_loops
        has_self_loops = contains_self_loops(edge_index)
        print(f"  Contains self-loops: {has_self_loops}")
        
        if has_self_loops:
            edge_no_loops, _ = remove_self_loops(edge_index)
            num_self_loops = edge_index.shape[1] - edge_no_loops.shape[1]
            print(f"  Self-loop edges: {num_self_loops}")
            print(f"  Non-self-loop edges: {edge_no_loops.shape[1]}")
        
        # Degree distribution
        from torch_geometric.utils import degree
        in_degree = degree(edge_index[1], num_nodes=self.num_nodes)
        out_degree = degree(edge_index[0], num_nodes=self.num_nodes)
        print(f"  In-degree  - mean: {in_degree.float().mean():.2f}, std: {in_degree.float().std():.2f}, min: {in_degree.min()}, max: {in_degree.max()}")
        print(f"  Out-degree - mean: {out_degree.float().mean():.2f}, std: {out_degree.float().std():.2f}, min: {out_degree.min()}, max: {out_degree.max()}")
        
        # ============= 2. INPUT DATA ANALYSIS =============
        print("\n[2] INPUT DATA ANALYSIS")
        print("-" * 80)
        print(f"  Input shape: {x.shape} (expected: [{self.num_nodes}, {self.num_features}, {self.seq_length}])")
        print(f"  Input dtype: {x.dtype}")
        print(f"  Input device: {x.device}")
        print(f"  Input statistics:")
        print(f"    Mean: {x.mean():.4f}")
        print(f"    Std:  {x.std():.4f}")
        print(f"    Min:  {x.min():.4f}")
        print(f"    Max:  {x.max():.4f}")
        print(f"  Input per feature statistics:")
        for feat in range(x.shape[1]):
            print(f"    Feature {feat}: mean={x[:, feat, :].mean():.4f}, std={x[:, feat, :].std():.4f}")
        
        # Check for NaN/Inf
        has_nan = torch.isnan(x).any()
        has_inf = torch.isinf(x).any()
        print(f"  Contains NaN: {has_nan}")
        print(f"  Contains Inf: {has_inf}")
        
        # ============= 3. GAT LAYER ANALYSIS =============
        print("\n[3] GAT PROCESSING (timestep by timestep)")
        print("-" * 80)
        
        seq_outputs = []
        gat1_activations = []
        
        for t in range(x.shape[2]):
            x_t = x[:, :, t]  # [num_nodes, num_features]
            
            if t == 0:
                print(f"  Timestep {t}:")
                print(f"    Input to GAT1: mean={x_t.mean():.4f}, std={x_t.std():.4f}")
            
            # GAT layer
            h_gat1 = self.gat1(x_t, edge_index)
            if t == 0:
                print(f"    After GAT1 (before ReLU): mean={h_gat1.mean():.4f}, std={h_gat1.std():.4f}")
                print(f"    GAT1 output shape: {h_gat1.shape}")
            
            h_gat1 = F.relu(h_gat1)
            gat1_activations.append(h_gat1.clone())
            
            if t == 0:
                print(f"    After ReLU: mean={h_gat1.mean():.4f}, std={h_gat1.std():.4f}")
                sparsity = (h_gat1 == 0).float().mean()
                print(f"    ReLU sparsity: {sparsity:.2%} zeros")
            
            h_gat1 = F.dropout(h_gat1, p=self.dropout, training=self.training)
            
            if t == 0:
                print(f"    After dropout: mean={h_gat1.mean():.4f}, std={h_gat1.std():.4f}")
            
            seq_outputs.append(h_gat1)
        
        # Analyze GAT outputs across time
        print(f"\n  GAT output statistics across all timesteps:")
        gat1_stack = torch.stack(gat1_activations, dim=0)
        print(f"    GAT1: mean={gat1_stack.mean():.4f}, std={gat1_stack.std():.4f}")
        
        # Check if GAT is collapsing representations
        node_variance_gat1 = gat1_stack.var(dim=1).mean()  # Variance across nodes per timestep
        print(f"    Node variance GAT1: {node_variance_gat1:.4f} (low = nodes too similar)")
        
        # ============= 4. LSTM PROCESSING =============
        print("\n[4] LSTM PROCESSING")
        print("-" * 80)
        
        x_seq = torch.stack(seq_outputs, dim=0)
        print(f"  LSTM input shape: {x_seq.shape} (expected: [{self.seq_length}, {self.num_nodes}, {self.gat_hidden_dim * self.num_heads}])")
        print(f"  LSTM input statistics:")
        print(f"    Mean: {x_seq.mean():.4f}")
        print(f"    Std:  {x_seq.std():.4f}")
        
        dbl1 = DebuggingLine(list(x_seq.shape), [self.seq_length, self.num_nodes, self.gat_hidden_dim * self.num_heads])
        
        if hidden_state is None:
            lstm_out, (h, c) = self.lstm(x_seq)
        else:
            lstm_out, (h, c) = self.lstm(x_seq, hidden_state)
        
        print(f"  LSTM output shape: {lstm_out.shape}")
        print(f"  LSTM hidden state shape: {h.shape} (num_layers={self.temporal_layers})")
        print(f"  LSTM cell state shape: {c.shape}")
        print(f"  LSTM output statistics:")
        print(f"    Mean: {lstm_out.mean():.4f}")
        print(f"    Std:  {lstm_out.std():.4f}")
        
        ht_lstm = h[-1]  # Last layer
        print(f"  Final hidden state (h[-1]) statistics:")
        print(f"    Mean: {ht_lstm.mean():.4f}")
        print(f"    Std:  {ht_lstm.std():.4f}")
        
        ht_lstm = self.lstm_out_norm(ht_lstm)
        print(f"  After LayerNorm:")
        print(f"    Mean: {ht_lstm.mean():.4f}")
        print(f"    Std:  {ht_lstm.std():.4f}")
        
        # ============= 5. FINAL OUTPUT =============
        print("\n[5] FINAL OUTPUT")
        print("-" * 80)
        
        output = self.fc(ht_lstm)
        print(f"  Output shape: {output.shape} (expected: [{self.num_nodes}, {self.horizon_size}])")
        print(f"  Output statistics:")
        print(f"    Mean: {output.mean():.4f}")
        print(f"    Std:  {output.std():.4f}")
        print(f"    Min:  {output.min():.4f}")
        print(f"    Max:  {output.max():.4f}")
        
        # Check output variance across nodes
        output_node_variance = output.var(dim=0).mean()
        print(f"  Node variance in output: {output_node_variance:.4f}")
        print(f"    (low variance = model predicting similar values for all nodes)")
        
        # ============= 6. GRADIENT FLOW CHECK =============
        print("\n[6] GRADIENT FLOW")
        print("-" * 80)
        if self.training:
            print(f"  Model is in training mode")
            print(f"  GAT1 weights require grad: {self.gat1.lin_l.weight.requires_grad}")
            print(f"  LSTM weights require grad: {next(self.lstm.parameters()).requires_grad}")
            print(f"  FC weights require grad: {self.fc.weight.requires_grad}")
        else:
            print(f"  Model is in eval mode")
        
        # ============= 7. SHAPE VERIFICATION =============
        print("\n[7] SHAPE VERIFICATION")
        print("-" * 80)
        dbl2 = DebuggingLine(list(lstm_out.shape), [self.seq_length, self.num_nodes, self.hidden_size])
        dbl3 = DebuggingLine(list(ht_lstm.shape), [self.num_nodes, self.hidden_size])
        dbl4 = DebuggingLine(list(output.shape), [self.num_nodes, self.horizon_size])
        
        debugging_report = ModelDebuggingReport([dbl1, dbl2, dbl3, dbl4])
        
        print("\n" + "="*80)
        print(" "*25 + "DEBUG COMPLETE")
        print("="*80 + "\n")
        
        return output, debugging_report


class GATv2LSTMModel(DeepModel):
    """
    GAT+LSTM model for spatiotemporal forecasting
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