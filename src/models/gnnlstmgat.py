
import torch, random, numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from ..dataloading.gnndataloader import GNNDataLoader
from .modelcore import ModelCore
from ..metrics.spike_weighted_mse import spike_weighted_mse
import torch_geometric.nn as pyg_nn

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch_geometric.nn as pyg_nn

class ArchitectureTemporalGATLSTM(torch.nn.Module):
    def __init__(self, node_features, periods, lstm_hidden_size=32, gat_hidden_size=32, num_heads=4):
        super(ArchitectureTemporalGATLSTM, self).__init__()
        self.periods = periods
        
        # Graph Attention Layer (GAT)
        # num_heads: Number of attention heads, typically 4 or 8.
        self.gat = GATConv(node_features, gat_hidden_size, heads=num_heads, concat=True)

        # LSTM layer for sequential modeling
        self.lstm = torch.nn.LSTM(input_size=gat_hidden_size * num_heads, hidden_size=lstm_hidden_size, num_layers=2, batch_first=True)

        # Output layer for prediction (1 output per node)
        self.linear = torch.nn.Linear(lstm_hidden_size, 1)

    def forward(self, x, edge_index):
        # x: [num_nodes, node_features, periods]  # Node features across time
        gatt_out_seq = []

        for t in range(self.periods):
            # Get node features for time t (shape: [num_nodes, node_features])
            xt = x[:, :, t]

            # Apply GAT to the features at this time step (output shape: [num_nodes, gat_hidden_size * num_heads])
            xt_gat = self.gat(xt, edge_index)
            xt_gat = torch.relu(xt_gat)  # Apply ReLU activation after GAT

            gatt_out_seq.append(xt_gat)

        # Stack the graph attention outputs along the time dimension (shape: [num_nodes, periods, gat_hidden_size * num_heads])
        gatt_out_seq = torch.stack(gatt_out_seq, dim=1)

        # LSTM expects input shape: [batch_size, seq_length, features]
        lstm_out, _ = self.lstm(gatt_out_seq)

        # Use the last time step output for prediction (shape: [num_nodes, lstm_hidden_size])
        last_out = lstm_out[:, -1, :]

        # Output prediction (shape: [num_nodes, 1])
        out = self.linear(last_out)

        return out.squeeze(-1)  # Return shape [num_nodes,]


# Example Model Usage
class GATLSTMModel(ModelCore):
    def __init__(self, dataloader:GNNDataLoader, name=None):
        super().__init__(dataloader, name=name)
        if not self.name:
            self.name = f'GAT_LSTM'
        self.model_color = "purple"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataloader = dataloader

    def _init_model(self, hidden_size: int = 128, lr=0.001, num_heads=4):
        self.model = ArchitectureTemporalGATLSTM(
            node_features=len(self.dataloader.feature_columns),
            periods=self.dataloader.periods,
            lstm_hidden_size=hidden_size,
            gat_hidden_size=hidden_size,
            num_heads=num_heads
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

    def train(self, n_epochs=100, patience=15, min_delta=1e-4):
        self.model.train()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(n_epochs):
            total_loss = 0
            for snapshot in self.dataloader.dataset_train:
                snapshot = snapshot.to(self.device)
                self.optimizer.zero_grad()

                y_hat = self.model(snapshot.x, snapshot.edge_index)
                loss = spike_weighted_mse(y_hat, snapshot.y)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            train_mse = total_loss / len(list(self.dataloader.dataset_train))

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for snapshot in self.dataloader.dataset_val:
                    snapshot = snapshot.to(self.device)
                    y_hat = self.model(snapshot.x, snapshot.edge_index)
                    loss = spike_weighted_mse(y_hat, snapshot.y)
                    val_loss += loss.item()
            
            val_mse = val_loss / len(list(self.dataloader.dataset_val))
            
            self.model.train()
            val_improved = val_mse < (best_val_loss - min_delta)
            
            if val_improved:
                best_val_loss = val_mse
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print(f"Epoch {epoch} train MSE: {train_mse:.4f}, val MSE: {val_mse:.4f} ✓ (new best)")
            else:
                patience_counter += 1
                print(f"Epoch {epoch} train MSE: {train_mse:.4f}, val MSE: {val_mse:.4f} (patience: {patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print(f"Early stopping: Validation loss hasn't improved for {patience} epochs")
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                        print(f"Restored model from best validation loss: {best_val_loss:.4f}")
                    break
            
            self.scheduler.step()

    def forecast(self):
        self.model.eval()
        loss = 0
        step = 0

        predictions = []
        labels = []

        for snapshot in self.dataloader.dataset_test:
            snapshot = snapshot.to(self.device)
            y_hat = self.model(snapshot.x, snapshot.edge_index)
            loss = loss + spike_weighted_mse(y_hat, snapshot.y)
            labels.append(snapshot.y)
            predictions.append(y_hat)
            step += 1

        loss = loss / (step+1)
        loss = float(loss)
        print("Test MSE: {:.4f}".format(loss))

        eval_df = self.dataloader.test_df

        tensor_list_cpu = [t.detach().cpu() for t in predictions]
        stacked = torch.stack(tensor_list_cpu)
        df = pd.DataFrame(stacked.numpy())  # shape: [50, 300]
        long_df_predictions = df.reset_index().melt(id_vars='index', var_name='position', value_name='preds')
        long_df_predictions = long_df_predictions.rename(columns={'index': 'timestamp_idx', 'position': 'id'})

        ts = eval_df['timestamp'].unique()[self.dataloader.periods:]
        timestamps = {idx: tss for idx, tss in enumerate(ts)}

        cutperiods = eval_df[eval_df['timestamp'].isin(ts)]
        long_df_predictions['timestamp'] = long_df_predictions['timestamp_idx'].map(timestamps)
        self.evaluation_df = pd.merge(cutperiods, long_df_predictions, on=['timestamp', 'id'])

        return self
