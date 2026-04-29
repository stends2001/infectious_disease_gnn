
import torch
import torch.nn as nn
from .baseloss import BaseLoss

class MSELoss(BaseLoss):
    """Standard Mean Squared Error loss."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mse = nn.MSELoss()
    
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.mse(y_pred, y_true)

class MAELoss(BaseLoss):
    """Mean Absolute Error loss."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mae = nn.L1Loss()
    
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.mae(y_pred, y_true)

class HuberLoss(BaseLoss):
    """Huber loss - combines MSE and MAE properties."""
    
    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(delta=delta, **kwargs)
        self.delta = delta
        self.huber = nn.HuberLoss(delta=delta)
    
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.huber(y_pred, y_true)

class WeightedMSELoss(BaseLoss):
    """
    MSE loss with asymmetric penalties.
    
    Applies higher weight when target is non-zero but prediction is zero
    (missed outbreak), and lower weight when both are non-zero.
    Zero-target predictions are penalised with standard MSE weight of 1.

    Parameters
    ----------
    high_weight : float
        Weight for missed outbreaks (target != 0, pred == 0). Default 10.0.
    low_weight : float
        Weight for non-zero predictions on non-zero targets. Default 2.0.
    """

    def __init__(self, high_weight: float = 10.0, low_weight: float = 2.0, **kwargs):
        super().__init__(high_weight=high_weight, low_weight=low_weight, **kwargs)
        self.high_weight = high_weight
        self.low_weight  = low_weight
        self.mse         = nn.MSELoss(reduction='none')

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        elementwise_loss: torch.Tensor = self.mse(y_pred, y_true)

        nonzero_target   = y_true != 0
        zero_pred        = y_pred == 0

        # build weight tensor — default weight is 1
        weights = torch.ones_like(elementwise_loss)
        weights[nonzero_target &  zero_pred] = self.high_weight  # missed outbreak
        weights[nonzero_target & ~zero_pred] = self.low_weight   # non-zero prediction

        return (elementwise_loss * weights).mean()