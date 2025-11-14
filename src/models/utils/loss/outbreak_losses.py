
import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, Any, List
from abc import ABC, abstractmethod

from .baseloss import BaseLoss

class FocalLoss(BaseLoss):
    """
    Focuses on hard-to-predict samples (outbreaks).
    Reduces loss for easy samples (zeros).
    """
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(alpha=alpha, gamma = gamma, **kwargs)
        self.alpha = alpha
        self.gamma = gamma           

        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse = (y_pred - y_true) ** 2
        # Upweight large errors (outbreaks)
        pt = torch.exp(-mse)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * mse
        return loss.mean()

class OutbreakWeightedLoss(BaseLoss):
    """
    Heavily weights outbreak periods.
    """
    def __init__(self, outbreak_threshold:float = 0.01, outbreak_weight: float = 3, normal_weight = 1, **kwargs):
        super().__init__(outbreak_threshold=outbreak_threshold, outbreak_weight = outbreak_weight, normal_weight = normal_weight, **kwargs)
        self.outbreak_threshold = outbreak_threshold
        self.outbreak_weight    = outbreak_weight
        self.normal_weight      = normal_weight
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Compute quantile threshold dynamically
        threshold = torch.quantile(y_true, self.outbreak_threshold)
        
        # Weight samples by whether they're outbreaks
        weights = torch.where(y_true > threshold, 
                             torch.tensor(self.outbreak_weight),  # High weight for outbreaks
                             torch.tensor(self.normal_weight))   # Normal weight otherwise
        
        mse = (y_pred - y_true) ** 2
        weighted_loss = (weights * mse).mean()
        return weighted_loss