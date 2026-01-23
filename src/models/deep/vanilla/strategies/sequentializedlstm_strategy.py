from typing import Tuple 
import torch

from ...basestrategy import Strategy

class SequentializedLSTMStrategy(Strategy):
    """

    """
    
    def __init__(self):
        # No persistent hidden state - each window is processed independently
        pass
    
    def training_step(self, model, snapshot, optimizer, loss_fn) -> float:
        optimizer.zero_grad()
        
        # Model processes the entire window and returns prediction
        # No hidden state passed in - the window contains all temporal context
        y_hat = model(snapshot.x)  # [N, horizon_size]
        
        loss = loss_fn(y_hat, snapshot.y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def validation_step(self, model, snapshot, loss_fn) -> float:
        # Same as training but no gradient updates
        y_hat = model(snapshot.x)
        loss = loss_fn(y_hat, snapshot.y)
        return loss.item()
    
    def forecast_step(self, model, snapshot, loss_fn) -> Tuple[torch.Tensor, float]:
        y_hat = model(snapshot.x)
        loss = loss_fn(y_hat, snapshot.y).item()
        return y_hat, loss
    
    def reset_state(self):
        # No state to reset - each window is independent
        pass
    
    def __repr__(self) -> str:
        return "Sequentialized LSTM strategy"