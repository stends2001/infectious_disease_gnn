from .basestrategy import Strategy
from torch.optim.optimizer import Optimizer
from ....dataloading.databuilders.graphdatabuilder.datacontainers import Data
from ..deepmodel.loss.losshandler import LossHandler

from typing import Tuple, Optional
import torch

from torch import Tensor
class StandardGNNStrategy(Strategy):
    """Standard (non-recurrent) strategy - no hidden state management"""
    
    def _detach_and_move(self, state: Optional[Tensor], device: torch.device) -> Optional[Tensor]:
        if state is None:
            return None
        return state.detach().to(device)    

    def training_step(self, model: torch.nn.Module, snapshot: Data, optimizer: Optimizer, loss_fn: LossHandler) -> float:
        y_hat:  Tensor
        loss:   Tensor        
        
        optimizer.zero_grad()
        assert snapshot.graph is not None
        y_hat = model(snapshot.x, snapshot.graph.edge_index, snapshot.graph.edge_weight)

        loss    = loss_fn(y_hat, snapshot.y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def validation_step(self, model: torch.nn.Module, snapshot: Data, loss_fn: LossHandler) -> float:
        y_hat:  Tensor
        loss:   Tensor      

        assert snapshot.graph is not None
        y_hat       = model(snapshot.x, snapshot.graph.edge_index, snapshot.graph.edge_weight)      

        loss = loss_fn(y_hat, snapshot.y)
        return loss.item()
    
    def forecast_step(self, model: torch.nn.Module, snapshot: Data, loss_fn: LossHandler) -> Tuple[torch.Tensor, float]:
        y_hat:  Tensor
        loss:   Tensor      

        assert snapshot.graph is not None
        y_hat = model(snapshot.x, snapshot.graph.edge_index, snapshot.graph.edge_weight)
        loss = loss_fn(y_hat, snapshot.y)
        return y_hat, loss.item()


    def reset_state(self):
        """No state to reset"""
        pass

    def __repr__(self) -> str:
        return "standard strategy"