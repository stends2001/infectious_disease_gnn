from .basestrategy import Strategy
from torch.optim.optimizer import Optimizer
from ....dataloading.databuilders.graphdatabuilder.datacontainers import Data
from ...utils.loss.losshandler import LossHandler

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any, TYPE_CHECKING
import torch
from src.utils.textformatting import warning_emoji

from torch import Tensor

if TYPE_CHECKING:
    from ..debugging import ModelDebuggingReport

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
    
    def debug(self, model: torch.nn.Module, snapshot: Data) -> Tuple[Tensor, 'ModelDebuggingReport']:
        y_hat:  Tensor     
        rep:    'ModelDebuggingReport'  

        assert snapshot.graph is not None
        y_hat, rep = model.debug(snapshot.x, snapshot.graph.edge_index, snapshot.graph.edge_weight)
        return y_hat, rep

    def reset_state(self):
        """No state to reset"""
        pass

    def __repr__(self) -> str:
        return "standard strategy"