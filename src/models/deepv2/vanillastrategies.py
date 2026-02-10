from torch import Tensor 
from typing import Optional 
import torch 
from torch.optim.optimizer import Optimizer

from ..utils.loss.baseloss import BaseLoss
from ...dataloading.dataloaders.deepdataloaders.datacontainers import DeepData, GraphData
from .basestrategy import Strategy 


class StatelessLSTMStrategy(Strategy):
    """
    LSTM models with which hidden states are resetted each epoch

    For LSTMS, parameters to watch out for are:
    - hidden_state
    - cell_state
    """    

    def __init__(self):
        self.hidden_state: Optional[Tensor] = None
        self.cell_state: Optional[Tensor]   = None
    
    def _detach_and_move(self, state: Optional[Tensor], device: torch.device) -> Optional[Tensor]:
        if state is None:
            return None
        return state.detach().to(device)

    def training_step(self, model: torch.nn.Module, snapshot: DeepData | GraphData, optimizer: Optimizer, loss_fn: BaseLoss) -> float:
        y_hat:  Tensor
        h:      Tensor
        c:      Tensor
        loss:   Tensor        
        
        optimizer.zero_grad()
        
        hidden      = (self.hidden_state, self.cell_state) if self.hidden_state is not None else None
        y_hat, h, c = model(snapshot.x, hidden)
        
        self.hidden_state = self._detach_and_move(h, snapshot.x.device)
        self.cell_state   = self._detach_and_move(c, snapshot.x.device)

        loss    = loss_fn(y_hat, snapshot.y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def validation_step(self, model: torch.nn.Module, snapshot: DeepData | GraphData, loss_fn: BaseLoss) -> float:
        y_hat:  Tensor
        h:      Tensor
        c:      Tensor
        loss:   Tensor      

        hidden      = (self.hidden_state, self.cell_state) if self.hidden_state is not None else None
        y_hat, h, c = model(snapshot.x, hidden)        

        self.hidden_state   = self._detach_and_move(h, snapshot.x.device)
        self.cell_state     = self._detach_and_move(c, snapshot.x.device)

        loss = loss_fn(y_hat, snapshot.y)
        return loss.item()

    def forecast_step(self, model: torch.nn.Module, snapshot: DeepData | GraphData, loss_fn: BaseLoss) -> Tuple[torch.Tensor, float]:
        y_hat:  Tensor
        h:      Tensor
        c:      Tensor
        loss:   Tensor  

        hidden = (self.hidden_state, self.cell_state) if self.hidden_state is not None else None
        y_hat, h, c = model(snapshot.x, hidden)     

        self.hidden_state   = self._detach_and_move(h, snapshot.x.device)
        self.cell_state     = self._detach_and_move(c, snapshot.x.device)

        loss = loss_fn(y_hat, snapshot.y)
        return y_hat, loss.item()

    def reset_state_epoch(self) -> None:
        """Reset hidden states at epoch boundaries"""
        self.hidden_state = None
        self.cell_state     = None

    def debug(self, model: torch.nn.Module, snapshot: DeepData | GraphData):
        y_hat:  Tensor
        debugging_report: ModelDebuggingReport

        y_hat, debugging_report  = model.debug(snapshot.x)
        return y_hat, debugging_report  

    def __repr__(self) -> str:
        return "<StatelessLSTMStrategy(reset_state_epoch)>"    
    
class StatefullLSTMStrategy(Strategy):
    """
    LSTM models with which hidden states are resetted each dataset

    For LSTMS, parameters to watch out for are:
    - hidden_state
    - cell_state
    """    

    def __init__(self):
        self.hidden_state: Optional[Tensor] = None
        self.cell_state: Optional[Tensor]   = None
    
    def _detach_and_move(self, state: Optional[Tensor], device: torch.device) -> Optional[Tensor]:
        if state is None:
            return None
        return state.detach().to(device)

    def training_step(self, model: torch.nn.Module, snapshot: DeepData | GraphData, optimizer: Optimizer, loss_fn: BaseLoss) -> float:
        y_hat:  Tensor
        h:      Tensor
        c:      Tensor
        loss:   Tensor        
        
        optimizer.zero_grad()
        
        hidden      = (self.hidden_state, self.cell_state) if self.hidden_state is not None else None
        y_hat, h, c = model(snapshot.x, hidden)
        
        self.hidden_state = self._detach_and_move(h, snapshot.x.device)
        self.cell_state   = self._detach_and_move(c, snapshot.x.device)

        loss    = loss_fn(y_hat, snapshot.y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def validation_step(self, model: torch.nn.Module, snapshot: DeepData | GraphData, loss_fn: BaseLoss) -> float:
        y_hat:  Tensor
        h:      Tensor
        c:      Tensor
        loss:   Tensor      

        hidden      = (self.hidden_state, self.cell_state) if self.hidden_state is not None else None
        y_hat, h, c = model(snapshot.x, hidden)        

        self.hidden_state   = self._detach_and_move(h, snapshot.x.device)
        self.cell_state     = self._detach_and_move(c, snapshot.x.device)

        loss = loss_fn(y_hat, snapshot.y)
        return loss.item()

    def forecast_step(self, model: torch.nn.Module, snapshot: DeepData | GraphData, loss_fn: BaseLoss) -> Tuple[torch.Tensor, float]:
        y_hat:  Tensor
        h:      Tensor
        c:      Tensor
        loss:   Tensor  

        hidden = (self.hidden_state, self.cell_state) if self.hidden_state is not None else None
        y_hat, h, c = model(snapshot.x, hidden)     

        self.hidden_state   = self._detach_and_move(h, snapshot.x.device)
        self.cell_state     = self._detach_and_move(c, snapshot.x.device)

        loss = loss_fn(y_hat, snapshot.y)
        return y_hat, loss.item()

    def debug(self, model: torch.nn.Module, snapshot: DeepData | GraphData):
        y_hat:  Tensor
        debugging_report: ModelDebuggingReport

        y_hat, debugging_report  = model.debug(snapshot.x)
        return y_hat, debugging_report  

    def __repr__(self) -> str:
        return "<StatefullLSTMStrategy(reset_state_dataset)>" 