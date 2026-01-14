from ...basestrategy import Strategy

from typing import Tuple 
import torch

class RecurrentLSTMStrategy(Strategy):
    def __init__(self):
        self.hidden_state = None
        self.cell_state = None
    
    def _detach_and_move(self, state, device):
        if state is None:
            return None
        return state.detach().to(device)

    def training_step(self, model, snapshot, optimizer, loss_fn) -> float:
        optimizer.zero_grad()
        
        # Pass hidden state as tuple or None
        hidden = (self.hidden_state, self.cell_state) if self.hidden_state is not None else None
        
        y_hat, h, c = model(snapshot.x, hidden)
        
        self.hidden_state = self._detach_and_move(h, snapshot.x.device)
        self.cell_state = self._detach_and_move(c, snapshot.x.device)

        loss = loss_fn(y_hat, snapshot.y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def validation_step(self, model, snapshot, loss_fn) -> float:

        # Pass hidden state as tuple or None
        hidden = (self.hidden_state, self.cell_state) if self.hidden_state is not None else None
        
        y_hat, h, c = model(snapshot.x, hidden)        

        self.hidden_state = self._detach_and_move(h, snapshot.x.device)
        self.cell_state = self._detach_and_move(c, snapshot.x.device)

        loss = loss_fn(y_hat, snapshot.y)
        return loss.item()

    def forecast_step(self, model, snapshot, loss_fn) -> Tuple[torch.Tensor, float]:

        # Pass hidden state as tuple or None
        hidden = (self.hidden_state, self.cell_state) if self.hidden_state is not None else None
        
        y_hat, h, c = model(snapshot.x, hidden)     

        self.hidden_state = self._detach_and_move(h, snapshot.x.device)
        self.cell_state = self._detach_and_move(c, snapshot.x.device)

        loss = loss_fn(y_hat, snapshot.y).item()
        return y_hat, loss

    def reset_state(self):
        self.hidden_state = None
        self.cell_state = None

    def __repr__(self) -> str:
        return "recurrent LSTM strategy"
