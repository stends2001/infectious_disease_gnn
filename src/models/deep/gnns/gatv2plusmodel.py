import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Optional, Tuple, Literal
from torch_geometric.utils import add_self_loops

from .strategies import GATv2PLUSStrategy
from ..deepmodel import DeepModel
from ....dataloading import GraphDataLoaderManager


class GATv2PLUSModule(nn.Module):
    """
    Recurrent GATv2-based spatiotemporal model following the paper architecture.
    Architecture: GRU → GATv2 (with skip connections) → FC output
    """
    def __init__(self,
                 node_features: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 temporal_layers: int,
                 num_heads: int,
                 prediction_horizon: int,
                 self_loops: bool = True
        ):
        super().__init__()

        self.node_features = node_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.temporal_layers = temporal_layers
        self.prediction_horizon = prediction_horizon
        self.self_loops = self_loops
        self.num_heads = num_heads

        # === Temporal GRU (processes time window) ===
        # Input: [num_nodes, time_window, node_features]
        # Output: [num_nodes, hidden_size] (last hidden state)
        self.temporal_gru = nn.GRU(
            input_size=node_features,
            hidden_size=hidden_size,
            num_layers=temporal_layers,
            dropout=dropout if temporal_layers > 1 else 0,
            batch_first=True  # Expect [batch, seq, features]
        )

        # === Spatial GATv2 layers (with skip connections) ===
        self.spatial_convs = nn.ModuleList()
        
        # First GATv2 layer: input is GRU output
        # No edge_dim - GATv2 learns attention weights from node features only
        self.spatial_convs.append(
            GATv2Conv(hidden_size, hidden_size, heads=num_heads, dropout=dropout, concat=True)
        )

        # Subsequent layers: input is concatenated [GATv2_output + GRU_embedding]
        gat_input_dim = hidden_size * num_heads + hidden_size  # concat output + skip
        
        for _ in range(num_layers - 1):
            self.spatial_convs.append(
                GATv2Conv(gat_input_dim, hidden_size, heads=num_heads, dropout=dropout, concat=True)
            )

        # Final dimension after last GATv2 layer
        self.final_gat_dim = hidden_size * num_heads

        # === Output layer ===
        self.output_proj = nn.Sequential(
            nn.Linear(self.final_gat_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, prediction_horizon)
        )

        self.output_scale = nn.Parameter(torch.tensor(3.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x_window: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None,
                debug: bool = False,
                ) -> torch.Tensor:
        """
        Forward pass following paper architecture: GRU → GATv2 → Output
        
        Args:
            x_window: [num_nodes, time_window, node_features] OR [num_nodes, node_features, time_window]
            edge_index: [2, num_edges]
            edge_weight: Optional edge weights
            
        Returns:
            output: [num_nodes, prediction_horizon]
        """
        
        if debug:
            print(f"Input shape (raw): {x_window.shape}")
        
        # Handle different input formats
        if x_window.dim() == 2:
            # Shape: [num_nodes, node_features]
            # This is single timestep from recurrent strategy - expand time dim
            x_window = x_window.unsqueeze(1)  # [num_nodes, 1, node_features]
        elif x_window.dim() == 3:
            # Check if shape is [num_nodes, features, time] and fix it
            if x_window.shape[1] > x_window.shape[2] and x_window.shape[2] < 20:
                # Likely [nodes, features, time] -> permute to [nodes, time, features]
                x_window = x_window.permute(0, 2, 1)
            # If last dimension is 1, squeeze and add back properly
            elif x_window.shape[-1] == 1:
                # Shape is [nodes, features, 1] -> squeeze to [nodes, features] -> expand to [nodes, 1, features]
                x_window = x_window.squeeze(-1).unsqueeze(1)
        
        if debug:
            print(f"Input shape (after fix): {x_window.shape}")
        
        # === 1. TEMPORAL: GRU processes time window ===
        # x_window: [num_nodes, time_window, node_features]
        gru_out, _ = self.temporal_gru(x_window)
        
        # Take last hidden state (output at final timestep)
        h_gru = gru_out[:, -1, :]  # [num_nodes, hidden_size]
        
        if debug:
            print(f"GRU output shape: {h_gru.shape}")
        
        # === 2. SPATIAL: GATv2 with skip connections ===
        if self.self_loops:
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_attr=edge_weight, num_nodes=x_window.size(0)
            )
        
        h = h_gru
        
        for i, gat in enumerate(self.spatial_convs):
            # Apply GATv2 (don't pass edge_weight - GATv2 computes its own attention)
            h_gat = gat(h, edge_index)  # No edge_attr parameter
            h_gat = F.relu(h_gat)
            h_gat = self.dropout(h_gat)
            
            # Skip connection: concatenate with original GRU embedding
            if i < len(self.spatial_convs) - 1:
                # Not the last layer - concatenate for next layer input
                h = torch.cat([h_gat, h_gru], dim=-1)
            else:
                # Last layer - just use GATv2 output
                h = h_gat
            
            if debug:
                print(f"GATv2 layer {i+1} output shape: {h.shape}")
        
        # === 3. OUTPUT: Map to prediction horizon ===
        output = self.output_proj(h)
        output = output * self.output_scale
        
        if debug:
            print(f"Final output shape: {output.shape}")
        
        return output


class GATv2PLUSModel(DeepModel):
    """
    Spatiotemporal GNN model following paper architecture.
    Processes temporal patterns first (GRU), then spatial patterns (GATv2).
    """
    def __init__(self, 
                 dataloadermanager: GraphDataLoaderManager, 
                 name: Optional[str] = None,
                 verbose: Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = 'GATv2PLUSmodel'

        super().__init__(
            dataloadermanager, 
            name=name, 
            deepfamily='gnn', 
            verbose=verbose, 
            strategy=GATv2PLUSStrategy(), 
            model_color="darkgreen"
        )

    def set_model_hparams(self, 
                          hidden_size: int = 128, 
                          num_layers: int = 2,
                          temporal_layers: int = 2, 
                          dropout: float = 0.2,
                          num_heads: int = 2, 
                          self_loops: bool = True):  # Changed default to True per paper
        
        self.model_hparams_set = True
        self.model = GATv2PLUSModule(
            node_features=len(self.column_registration.get_by_type('feature')),
            hidden_size=hidden_size,
            num_layers=num_layers,
            temporal_layers=temporal_layers,
            dropout=dropout,
            num_heads=num_heads,
            prediction_horizon=self.dataloadermanager.dataorchestrator.config.horizon_size,
            self_loops=self_loops
        ).to(self.device)
        
        model_hparams_config = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'temporal_layers': temporal_layers,
            'dropout': dropout,
            'num_heads': num_heads,
            'self_loops': self_loops
        }

        self.config_info['model_hparams'] = model_hparams_config
        self._update_status('model_hparams_set')
        return self