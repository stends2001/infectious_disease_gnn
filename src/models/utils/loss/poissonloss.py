from .baseloss import BaseLoss
import torch
import torch.nn as nn

class PoissonLoss(BaseLoss):
    """Poisson Loss (negative log-likelihood) with numerical stability improvements."""
    
    def __init__(self, epsilon: float = 1e-8, **kwargs):
        """
        Args:
            epsilon (float): Small value added to avoid log(0) errors.
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Inside your forward method, before loss calculation

        self._validate_inputs(y_pred, y_true)
        
        # Apply softplus to make sure predictions are positive
        y_pred = torch.nn.functional.softplus(y_pred)
        
        # Avoid log(0) by adding epsilon
        y_pred = torch.clamp(y_pred, min=self.epsilon)  # Clamp to avoid zero or very small values
        # print(f"Min y_pred after softplus and clamp: {y_pred.min()}")
        # print(f"Max y_pred after softplus and clamp: {y_pred.max()}")
        # Poisson loss: L = y_true * log(y_pred) - y_pred
        loss = y_true * torch.log(y_pred) - y_pred
        if torch.any(torch.isnan(loss)):
            print("Warning: NaN values detected in loss calculation!")
        return loss.mean()


class NegativeBinomialLoss(BaseLoss):
    """Negative Binomial Loss (negative log-likelihood)."""
    
    def __init__(self, alpha: float = 1.0, **kwargs):
        """
        Args:
            alpha (float): Dispersion parameter (often called theta in the NB distribution).
        """
        super().__init__(alpha=alpha, **kwargs)
        self.alpha = alpha
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(y_pred, y_true)
        
        # Ensure y_pred is positive, transform predictions using Softplus
        y_pred = torch.nn.functional.softplus(y_pred)
        
        # Negative Binomial loss: L = log(Γ(y_true + alpha)) - log(Γ(alpha)) - log(y_true!) + alpha * log(alpha) - (y_true + alpha) * log(y_pred + alpha)
        log_gamma_y = torch.lgamma(y_true + self.alpha)
        log_gamma_alpha = torch.lgamma(self.alpha)
        log_factorial_y = torch.lgamma(y_true + 1)
        
        loss = log_gamma_y - log_gamma_alpha - log_factorial_y + self.alpha * torch.log(self.alpha) - (y_true + self.alpha) * torch.log(y_pred + self.alpha)
        
        return loss.mean()
