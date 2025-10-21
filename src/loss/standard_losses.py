
import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, Any, List
from abc import ABC, abstractmethod

from .baseloss import BaseLoss

class MSELoss(BaseLoss):
    """Standard Mean Squared Error loss."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(y_pred, y_true)
        return self.mse(y_pred, y_true)


class MAELoss(BaseLoss):
    """Mean Absolute Error loss."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mae = nn.L1Loss()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(y_pred, y_true)
        return self.mae(y_pred, y_true)


class HuberLoss(BaseLoss):
    """Huber loss - combines MSE and MAE properties."""
    
    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(delta=delta, **kwargs)
        self.delta = delta
        self.huber = nn.HuberLoss(delta=delta)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(y_pred, y_true)
        return self.huber(y_pred, y_true)


class ExponentialDecayLoss(BaseLoss):
    """
    Applies exponential decay weights to future timesteps.
    Useful when recent predictions are more important than distant ones.
    """
    
    def __init__(self, gamma: float = 0.9, **kwargs):
        super().__init__(gamma=gamma, **kwargs)
        if not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        self.gamma = gamma

        print('exp decay initiated')
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(y_pred, y_true)
        
        print(f"y_pred shape: {y_pred.shape}")
        print(f"y_true shape: {y_true.shape}")
        
        # y_pred and y_true should be (num_nodes, horizon) = (400, 3)
        horizon = y_pred.size(-1)
        print(f"horizon: {horizon}")
        
        # Create weights: [gamma^0, gamma^1, gamma^2, ...]
        weights = torch.tensor(
            [self.gamma ** t for t in range(horizon)],
            dtype=y_pred.dtype,
            device=y_pred.device
        )
        print(f"weights initial shape: {weights.shape}")
        print(f"weights values: {weights}")
        
        # Reshape to (1, horizon) for broadcasting over nodes
        weights = weights.unsqueeze(0)
        print(f"weights after unsqueeze(0) shape: {weights.shape}")
        
        # Compute weighted squared error
        se = (y_pred - y_true) ** 2
        print(f"se shape: {se.shape}")
        
        print(f"About to multiply se * weights...")
        weighted_se = se * weights
        print(f"weighted_se shape: {weighted_se.shape}")
        
        return weighted_se.mean()
    
class PolynomialDecayLoss(BaseLoss):
    """
    Applies polynomial decay weights to future timesteps.
    More flexible than exponential decay.
    """
    
    def __init__(self, power: float = 1.0, **kwargs):
        super().__init__(power=power, **kwargs)
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}")
        self.power = power
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(y_pred, y_true)
        
        horizon = y_pred.size(1)
        weights = torch.tensor(
            [(t + 1) ** (-self.power) for t in range(horizon)],
            dtype=y_pred.dtype,
            device=y_pred.device
        )
        
        while len(weights.shape) < len(y_pred.shape):
            weights = weights.unsqueeze(-1)
        
        se = (y_pred - y_true) ** 2
        weighted_se = se * weights
        return weighted_se.mean()


class SmoothL1Loss(BaseLoss):
    """Smooth L1 loss (similar to Huber)."""
    
    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__(beta=beta, **kwargs)
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(y_pred, y_true)
        return self.smooth_l1(y_pred, y_true)

