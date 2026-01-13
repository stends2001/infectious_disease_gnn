import torch
from typing import Optional
import torch.nn as nn
from .baseloss import BaseLoss

class BCELoss(BaseLoss):
    """Binary Cross-Entropy loss for binary classification."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Using BCEWithLogitsLoss is numerically more stable than plain BCE
        self.bce = nn.BCELoss()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted logits (not passed through sigmoid yet)
            y_true: Ground truth labels (0 or 1)
        """
        self._validate_inputs(y_pred, y_true)
        return self.bce(y_pred, y_true)

class BCELogitLoss(BaseLoss):

    """parmeters include pos_weight"""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(pos_weight=pos_weight, **kwargs)     

        self.bcelogit = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight # For class imbalance!
        )        

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(y_pred, y_true)
        return self.bcelogit(y_pred, y_true)