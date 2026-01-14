from ...basestrategy import Strategy
from typing import Tuple
import torch

class RecurrentGRUStrategy(Strategy):
    """Recurrent strategy for GRU models only (single hidden tensor)."""

    def __init__(self):
        self.hidden_state = None

    def _detach_and_move(self, hidden_state, device):
        if hidden_state is None:
            return None
        return hidden_state.detach().to(device)

    def training_step(self, model, snapshot, optimizer, loss_fn) -> float:
        optimizer.zero_grad()

        y_hat, h = model(
            snapshot.x,
            self.hidden_state
        )

        self.hidden_state = self._detach_and_move(h, snapshot.x.device)

        loss = loss_fn(y_hat, snapshot.y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def validation_step(self, model, snapshot, loss_fn) -> float:
        y_hat, h = model(
            snapshot.x,
            self.hidden_state
        )

        self.hidden_state = self._detach_and_move(h, snapshot.x.device)
        loss = loss_fn(y_hat, snapshot.y)
        return loss.item()

    def forecast_step(self, model, snapshot, loss_fn) -> Tuple[torch.Tensor, float]:
        y_hat, h = model(
            snapshot.x,
            self.hidden_state
        )

        self.hidden_state = self._detach_and_move(h, snapshot.x.device)
        loss = loss_fn(y_hat, snapshot.y).item()
        return y_hat, loss

    def reset_state(self):
        self.hidden_state = None

    def __repr__(self) -> str:
        return "recurrent GRU strategy"