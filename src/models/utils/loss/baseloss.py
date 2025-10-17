import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, Any, List
from abc import ABC, abstractmethod

class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for all loss functions.
    
    Ensures consistency across loss implementations and provides
    common utilities like device management and validation.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
    
    @abstractmethod
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between predictions and targets.
        
        Parameters:
        -----------
        y_pred : torch.Tensor
            Model predictions, shape: [batch_size, horizon, ...]
        y_true : torch.Tensor
            Ground truth targets, same shape as y_pred
            
        Returns:
        --------
        torch.Tensor : Scalar loss value
        """
        pass
    
    def _validate_inputs(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Validate that inputs have matching shapes."""
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
            )
    
    def __repr__(self) -> str:
        kwargs_str = ', '.join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"{self.__class__.__name__}({kwargs_str})"