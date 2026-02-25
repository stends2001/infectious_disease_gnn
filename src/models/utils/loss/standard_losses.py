
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
        y_pred = y_pred.squeeze(-1)
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
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(y_pred, y_true)
        
        horizon = y_pred.size(-1)  # Last dimension is horizon
        weights = torch.tensor(
            [self.gamma ** t for t in range(horizon)],
            dtype=y_pred.dtype,
            device=y_pred.device
        )  # Shape: (3,)
        
        # Reshape to match y_pred dimensions, adding singleton dims at the FRONT
        for _ in range(len(y_pred.shape) - 1):
            weights = weights.unsqueeze(0)
        # Now weights.shape = (1, 3) for 2D input
        
        se = (y_pred - y_true) ** 2
        weighted_se = se * weights
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


class WeightedMSELoss(nn.Module):
    """Weighted Mean Squared Error loss with emphasis on underpredictions when target is non-zero."""
    
    def __init__(self, high_weight: float = 10.0, low_weight: float = 2.0, **kwargs):
        """
        Args:
            high_weight (float): The weight applied when the target is non-zero but the prediction is zero.
            low_weight (float): The weight applied when both target and prediction are non-zero.
            **kwargs: Additional arguments for flexibility.
        """
        super().__init__(**kwargs)
        self.mse = nn.MSELoss(reduction='none')  # Set reduction to 'none' to compute element-wise loss
        
        self.high_weight = high_weight  # For cases where target is non-zero and prediction is zero
        self.low_weight = low_weight  # For cases where both target and prediction are non-zero
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the weighted MSE loss with more emphasis on underpredictions (target != 0, pred == 0)."""
        self._validate_inputs(y_pred, y_true)
        
        # Compute the element-wise MSE loss
        loss = self.mse(y_pred, y_true)
        
        # Create masks for different conditions
        non_zero_target_mask = (y_true != 0).float()  # 1 for non-zero target
        zero_prediction_mask = (y_pred == 0).float()  # 1 for zero predictions
        non_zero_prediction_mask = (y_pred != 0).float()  # 1 for non-zero predictions
        
        # Condition 1: target is non-zero and prediction is zero -> High penalty
        high_penalty_mask = non_zero_target_mask * zero_prediction_mask
        
        # Condition 2: target and prediction both non-zero -> Low penalty
        low_penalty_mask = non_zero_target_mask * non_zero_prediction_mask
        
        # Apply penalties
        weighted_loss = (loss * (high_penalty_mask * (self.high_weight - 1) + 1))  # High penalty for underprediction
        
        # Apply lower penalty for non-zero target and prediction
        weighted_loss += (loss * (low_penalty_mask * (self.low_weight - 1) + 1))  # Low penalty for correct predictions
        
        # Return the mean loss
        return weighted_loss.mean()

    def _validate_inputs(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Helper method to validate inputs (optional but useful)."""
        if y_pred.size() != y_true.size():
            raise ValueError(f"Shape mismatch: y_pred.shape = {y_pred.shape}, y_true.shape = {y_true.shape}")