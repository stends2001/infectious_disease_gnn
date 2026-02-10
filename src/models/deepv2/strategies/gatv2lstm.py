import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any, TYPE_CHECKING 

from .basestrategy import Strategy
from ...utils.loss.baseloss import BaseLoss
from ....dataloading.dataloaders.deepdataloaders.datacontainers import DeepData, GraphData
from ....utils.textformatting import warning_emoji

if TYPE_CHECKING:
    from ..debugging import ModelDebuggingReport
    
class StatelessGATv2LSTMStrategy(Strategy[GraphData]):
    """
    """    

    def __init__(self):
        self.hidden_state: Optional[Tensor] = None
        self.cell_state: Optional[Tensor]   = None
    
    def _detach_and_move(self, state: Optional[Tensor], device: torch.device) -> Optional[Tensor]:
        if state is None:
            return None
        return state.detach().to(device)

    def training_step(self, model: torch.nn.Module, snapshot: GraphData, optimizer: Optimizer, loss_fn: BaseLoss) -> float:
        y_hat:  Tensor
        h:      Tensor
        c:      Tensor
        loss:   Tensor        
        
        optimizer.zero_grad()
        
        hidden      = (self.hidden_state, self.cell_state) if self.hidden_state is not None else None
        y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight, hidden)
        
        self.hidden_state = self._detach_and_move(h, snapshot.x.device)
        self.cell_state   = self._detach_and_move(c, snapshot.x.device)

        loss    = loss_fn(y_hat, snapshot.y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def validation_step(self, model: torch.nn.Module, snapshot: GraphData, loss_fn: BaseLoss) -> float:
        y_hat:  Tensor
        h:      Tensor
        c:      Tensor
        loss:   Tensor      

        hidden      = (self.hidden_state, self.cell_state) if self.hidden_state is not None else None
        y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight, hidden)       

        self.hidden_state   = self._detach_and_move(h, snapshot.x.device)
        self.cell_state     = self._detach_and_move(c, snapshot.x.device)

        loss = loss_fn(y_hat, snapshot.y)
        return loss.item()

    def forecast_step(self, model: torch.nn.Module, snapshot: GraphData, loss_fn: BaseLoss) -> Tuple[torch.Tensor, float]:
        y_hat:  Tensor
        h:      Tensor
        c:      Tensor
        loss:   Tensor  

        hidden = (self.hidden_state, self.cell_state) if self.hidden_state is not None else None
        y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight, hidden)  

        self.hidden_state   = self._detach_and_move(h, snapshot.x.device)
        self.cell_state     = self._detach_and_move(c, snapshot.x.device)

        loss = loss_fn(y_hat, snapshot.y)
        return y_hat, loss.item()

    def reset_state_epoch(self):
        """Reset hidden states at epoch boundaries"""
        self.hidden_state = None
        self.cell_state   = None

    def debug(self, model: torch.nn.Module, snapshot: GraphData) -> Tuple[Tensor, 'ModelDebuggingReport']:
        y_hat:  Tensor     
        rep:    'ModelDebuggingReport'  

        y_hat, rep = model.debug(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
        return y_hat, rep

    def __repr__(self) -> str:
        return "TODO" 
    
class StatefullGATv2LSTMStrategy(Strategy[GraphData]):

    def __init__(self):
        self.hidden_state: Optional[Tensor] = None
        self.cell_state: Optional[Tensor]   = None
    
    def _detach_and_move(self, state: Optional[Tensor], device: torch.device) -> Optional[Tensor]:
        if state is None:
            return None
        return state.detach().to(device)

    def training_step(self, model: torch.nn.Module, snapshot: GraphData, optimizer: Optimizer, loss_fn: BaseLoss) -> float:
        y_hat:  Tensor
        h:      Tensor
        c:      Tensor
        loss:   Tensor        
        
        optimizer.zero_grad()
        
        hidden      = (self.hidden_state, self.cell_state) if self.hidden_state is not None else None
        y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight, hidden)    
        
        self.hidden_state = self._detach_and_move(h, snapshot.x.device)
        self.cell_state   = self._detach_and_move(c, snapshot.x.device)

        loss    = loss_fn(y_hat, snapshot.y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def validation_step(self, model: torch.nn.Module, snapshot: GraphData, loss_fn: BaseLoss) -> float:
        y_hat:  Tensor
        h:      Tensor
        c:      Tensor
        loss:   Tensor      

        hidden      = (self.hidden_state, self.cell_state) if self.hidden_state is not None else None
        y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight, hidden)    

        self.hidden_state   = self._detach_and_move(h, snapshot.x.device)
        self.cell_state     = self._detach_and_move(c, snapshot.x.device)

        loss = loss_fn(y_hat, snapshot.y)
        return loss.item()

    def forecast_step(self, model: torch.nn.Module, snapshot: GraphData, loss_fn: BaseLoss) -> Tuple[torch.Tensor, float]:
        y_hat:  Tensor
        h:      Tensor
        c:      Tensor
        loss:   Tensor  

        hidden = (self.hidden_state, self.cell_state) if self.hidden_state is not None else None
        y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight, hidden)    

        self.hidden_state   = self._detach_and_move(h, snapshot.x.device)
        self.cell_state     = self._detach_and_move(c, snapshot.x.device)

        loss = loss_fn(y_hat, snapshot.y)
        return y_hat, loss.item()
    
    def debug(self, model: torch.nn.Module, snapshot: GraphData) -> Tuple[Tensor, 'ModelDebuggingReport']:
        y_hat:  Tensor     
        rep:    'ModelDebuggingReport'  

        y_hat, rep = model.debug(snapshot.x, snapshot.edge_index, snapshot.edge_weight) 
        return y_hat, rep

    def __repr__(self) -> str:
        return "TODO" 