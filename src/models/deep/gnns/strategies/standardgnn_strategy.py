from ...basestrategy import Strategy

from typing import Tuple 
import torch

class StandardGNNStrategy(Strategy):
    """Standard (non-recurrent) strategy - no hidden state management"""
    
    def training_step(self, model, snapshot, optimizer, loss_fn) -> float:
        optimizer.zero_grad()
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
        loss = loss_fn(y_hat, snapshot.y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def validation_step(self, model, snapshot, loss_fn) -> float:
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
        loss = loss_fn(y_hat, snapshot.y)
        return loss.item()
    
    def forecast_step(self, model, snapshot, loss_fn) -> Tuple[torch.Tensor, float]:
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
        loss = loss_fn(y_hat, snapshot.y).item()
        return y_hat, loss
    
    def reset_state(self):
        """No state to reset"""
        pass

    def __repr__(self) -> str:
        return "standard strategy"