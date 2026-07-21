from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ....dataloading.databuilders.graphdatabuilder.datacontainers import Data

if TYPE_CHECKING:
    from ..deepmodel.loss.losshandler import LossHandler

class Strategy(ABC):
    """
    Base class for model strategies that handle training and forecasting
    All child classes inherit the abstractmethods: must implement all of these:
    - training_step
    - validation_step
    - forecast_step
    - reset_state
    """
    @abstractmethod
    def training_step(self, model: torch.nn.Module, snapshot: Data, optimizer: Optimizer, loss_fn: 'LossHandler') -> float:
        """
        Execute one training step

        Parameters
        ----------
        model
        snapshot
        optimizer
        loss_fn

        Returns
        -------
        loss value
        """
        pass
    
    @abstractmethod    
    def validation_step(self, model: torch.nn.Module, snapshot: Data, loss_fn: 'LossHandler') -> float:
        """
        Execute one validation step

        Parameters
        ----------
        model
        snapshot
        loss_fn

        Returns
        -------
        loss value
        """
        pass
    
    @abstractmethod
    def forecast_step(self, model: torch.nn.Module, snapshot: Data, loss_fn: 'LossHandler') -> Tuple[Tensor, float]:
        """
        Execute one validation step

        Parameters
        ----------
        model
        snapshot
        loss_fn

        Returns
        -------
        (predictions, loss_value)
        """        
        pass

    def reset_state_epoch(self) -> None:
        """
        Resets any state between epochs

        May or may not be implemented, depending on the strategy
        """        
        pass

    def reset_state_dataset(self) -> None:
        """Reset hidden states at dataset boundaries. each strategy should have this"""
        if hasattr(self, 'hidden_state'):
            self.hidden_state = None

        if hasattr(self, 'cell_state'):            
            self.cell_state   = None   