from ...basestrategy import Strategy

from typing import Tuple 
import torch


class GATv2PLUSStrategy(Strategy):
    """
    Windowed strategy for GNN models that process entire time windows at once.
    
    Unlike RecurrentGNNStrategy, this doesn't manage hidden state between timesteps
    because the model receives the full temporal window as input.
    
    This matches the paper architecture where GRU processes X_t ∈ R^(n×d×f)
    where d is the full lookback window.
    """
    
    def training_step(self, model, snapshot, optimizer, loss_fn) -> float:
        """
        Training step for windowed model.
        
        Args:
            model: The GATv2Module (expects x_window input)
            snapshot: Data snapshot with x_window [num_nodes, time_window, features]
            optimizer: PyTorch optimizer
            loss_fn: Loss function
            
        Returns:
            loss value (float)
        """
        optimizer.zero_grad()
        
        # snapshot.x should now be [num_nodes, time_window, features]
        y_hat = model(
            snapshot.x,  # Full time window
            snapshot.edge_index, 
            snapshot.edge_weight
        )
        
        loss = loss_fn(y_hat, snapshot.y)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def validation_step(self, model, snapshot, loss_fn) -> float:
        """
        Validation step for windowed model.
        
        Args:
            model: The GATv2Module
            snapshot: Data snapshot with x_window
            loss_fn: Loss function
            
        Returns:
            loss value (float)
        """
        y_hat = model(
            snapshot.x,
            snapshot.edge_index,
            snapshot.edge_weight
        )
        
        loss = loss_fn(y_hat, snapshot.y)
        return loss.item()
    
    def forecast_step(self, model, snapshot, loss_fn) -> Tuple[torch.Tensor, float]:
        """
        Forecasting step for windowed model.
        
        Args:
            model: The GATv2Module
            snapshot: Data snapshot with x_window
            loss_fn: Loss function
            
        Returns:
            Tuple of (predictions, loss_value)
        """
        y_hat = model(
            snapshot.x,
            snapshot.edge_index,
            snapshot.edge_weight
        )
        
        loss = loss_fn(y_hat, snapshot.y).item()
        return y_hat, loss
    
    def reset_state(self):
        """
        No-op for windowed strategy since there's no hidden state to reset.
        Kept for interface compatibility.
        """
        pass

    def __repr__(self) -> str:
        return "windowed strategy"