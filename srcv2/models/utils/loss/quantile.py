
import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, Any, List
from abc import ABC, abstractmethod

from .baseloss import BaseLoss

class QuantileLoss(BaseLoss):
    """
    Quantile loss - naturally asymmetric.
    quantile=0.5 is MAE, quantile>0.5 penalizes underestimation more.
    """
    
    def __init__(self, quantile: float = 0.7, **kwargs):
        super().__init__(quantile=quantile, **kwargs)
        if not (0 < quantile < 1):
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.quantile = quantile
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(y_pred, y_true)
        
        error = y_true - y_pred
        loss = torch.where(
            error >= 0,
            self.quantile * error,
            (self.quantile - 1) * error
        )
        return loss.mean()