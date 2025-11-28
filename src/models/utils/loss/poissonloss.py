import torch
import torch.nn as nn
from .baseloss import BaseLoss


class PoissonLoss(BaseLoss):
    """Poisson loss for counting tasks, handling both negative predictions and targets."""

    def __init__(self, shift: float = 1.0, **kwargs):
        """
        Args:
            shift (float): A constant to add to both predictions and targets to avoid negative values.
        """
        super().__init__(**kwargs)
        self.shift = shift
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the Poisson loss with the absolute value, and handle negative predictions and targets.
        This function shifts both predictions and targets by a constant to make them non-negative.

        Args:
            y_pred (torch.Tensor): Predicted values (can be positive or negative).
            y_true (torch.Tensor): Ground truth values (can be positive or negative).

        Returns:
            torch.Tensor: Poisson loss.
        """
        self._validate_inputs(y_pred, y_true)

        # Shift both predictions and targets to be non-negative
        y_pred = torch.abs(y_pred) + self.shift
        y_true = torch.abs(y_true) + self.shift

        # Ensure that predictions are positive
        if torch.any(y_pred <= 0):
            raise ValueError("Predicted values must be positive for Poisson loss.")

        # Poisson loss computation: y_pred - y_true * log(y_pred)
        loss = y_pred - y_true * torch.log(y_pred)

        # If ground truth values are 0, set the loss to y_pred, since log(1) = 0
        loss = torch.where(y_true == 0, y_pred, loss)

        # Return the mean loss
        return loss.mean()

    def _validate_inputs(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Helper method to validate inputs."""
        if y_pred.size() != y_true.size():
            raise ValueError(f"Shape mismatch: y_pred.shape = {y_pred.shape}, y_true.shape = {y_true.shape}")
