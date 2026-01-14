from ...basestrategy import Strategy

from typing import Tuple 
import torch


class RecurrentGNNStrategy(Strategy):
    """Recurrent strategy - manages hidden state across all operations"""
    
    def __init__(self):
        self.hidden_state = None
    
    def _update_hidden_state(self, hidden_state, device):
        """Helper to detach and move hidden state to device"""
        hidden_state = tuple(h.detach() for h in hidden_state)
        if hidden_state is not None:
            hidden_state = tuple(h.to(device) for h in hidden_state)
        return hidden_state
    
    def training_step(self, model, snapshot, optimizer, loss_fn) -> float:
        optimizer.zero_grad()
        y_hat, self.hidden_state = model(
            snapshot.x, 
            snapshot.edge_index, 
            snapshot.edge_weight,
            self.hidden_state
        )
        self.hidden_state = self._update_hidden_state(self.hidden_state, snapshot.x.device)
        
        loss = loss_fn(y_hat, snapshot.y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def validation_step(self, model, snapshot, loss_fn) -> float:
        y_hat, self.hidden_state = model(
            snapshot.x,
            snapshot.edge_index,
            snapshot.edge_weight,
            self.hidden_state
        )
        self.hidden_state = self._update_hidden_state(self.hidden_state, snapshot.x.device)
        
        loss = loss_fn(y_hat, snapshot.y)
        return loss.item()
    
    def forecast_step(self, model, snapshot, loss_fn) -> Tuple[torch.Tensor, float]:
        y_hat, self.hidden_state = model(
            snapshot.x,
            snapshot.edge_index,
            snapshot.edge_weight,
            self.hidden_state
        )
        self.hidden_state = self._update_hidden_state(self.hidden_state, snapshot.x.device)
        
        loss = loss_fn(y_hat, snapshot.y).item()
        return y_hat, loss
    
    def reset_state(self):
        """Reset hidden state"""
        self.hidden_state = None

    def __repr__(self) -> str:
        return "recurrent strategy"