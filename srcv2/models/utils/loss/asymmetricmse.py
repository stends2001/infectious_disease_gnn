
import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, Any, List
from abc import ABC, abstractmethod

from .baseloss import BaseLoss

class AsymmetricMSELoss(BaseLoss):
    """
    Asymmetric MSE that penalizes underestimation more heavily.
    When y_pred < y_true (underestimation), apply higher penalty.
    """
    
    def __init__(self, underestimate_penalty: float = 2.0, **kwargs):
        super().__init__(underestimate_penalty=underestimate_penalty, **kwargs)
        if underestimate_penalty < 1.0:
            raise ValueError(f"underestimate_penalty must be >= 1.0, got {underestimate_penalty}")
        self.penalty = underestimate_penalty
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(y_pred, y_true)
        
        error = y_pred - y_true
        squared_error = error ** 2
        
        # Apply higher weight where we underestimate (error < 0)
        weights = torch.where(error < 0, self.penalty, 1.0)
        
        return (squared_error * weights).mean()


