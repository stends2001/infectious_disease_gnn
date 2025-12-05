import torch
import torch.nn as nn
import torch.nn.functional as F
from .baseloss import BaseLoss

from typing import Literal


class PoissonLoss(BaseLoss):
    """
    Poisson Loss using PyTorch's built-in PoissonNLLLoss.
    
    Appropriate for modeling discrete count data.
    Note: Negative loss values are mathematically valid when predictions are accurate!
    The loss can be negative because we're computing log-likelihood, which is bounded above by 0.
    """
    
    def __init__(self, log_input: bool = False, epsilon: float = 1e-8, **kwargs):
        """
        Args:
            log_input (bool): If True, loss expects log(λ) as input. If False, expects λ.
            epsilon (float): Small value for numerical stability.
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        # log_input=False, full=False means we compute: λ - y*log(λ)
        # This is the standard Poisson NLL without the log(y!) term
        self.poisson_nll = nn.PoissonNLLLoss(log_input=log_input, full=False, reduction='mean')
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute Poisson negative log-likelihood.
        
        Args:
            y_pred: Model predictions (continuous values, can be negative)
            y_true: Ground truth counts (non-negative integers)
            
        Returns:
            Scalar loss value (can be negative for good predictions!)
        """
        self._validate_inputs(y_pred, y_true)
        
        # Transform predictions to positive rate parameter λ using softplus
        lambda_pred = F.softplus(y_pred) + self.epsilon
        
        # PyTorch's PoissonNLLLoss expects target first, then input
        return self.poisson_nll(lambda_pred, y_true)


class NegativeBinomialLoss(BaseLoss):
    """
    Negative Binomial Loss for overdispersed count data.
    
    More flexible than Poisson - allows variance > mean, which is common
    in real-world count data (like disease cases with clustering).
    """
    
    def __init__(self, epsilon: float = 1e-8, **kwargs):
        """
        Args:
            epsilon (float): Small value for numerical stability.
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        # Learnable dispersion parameter (r)
        # Higher r → closer to Poisson, lower r → more overdispersion
        # Initialize as tensor to avoid device issues
        self.log_r = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute Negative Binomial negative log-likelihood.
        
        Args:
            y_pred: Model predictions (continuous values)
            y_true: Ground truth counts (non-negative integers)
            
        Returns:
            Scalar loss value
        """
        self._validate_inputs(y_pred, y_true)
        
        # Transform to positive mean
        mu = F.softplus(y_pred) + self.epsilon
        
        # Dispersion parameter (must be positive, on same device as input)
        r = torch.exp(self.log_r.to(y_pred.device)) + self.epsilon
        
        # Negative Binomial NLL
        # NLL = -[log Γ(y + r) - log Γ(r) + r*log(r) + y*log(μ) - (r+y)*log(r+μ)]
        t1 = torch.lgamma(y_true + r)
        t2 = torch.lgamma(r)
        t3 = r * torch.log(r + self.epsilon)
        t4 = y_true * torch.log(mu + self.epsilon)
        t5 = (r + y_true) * torch.log(r + mu + self.epsilon)
        
        loss = -(t1 - t2 + t3 + t4 - t5)
        
        return loss.mean()


class ZeroInflatedPoissonLoss(BaseLoss):
    """
    Zero-Inflated Poisson Loss for count data with excess zeros.
    
    Models two processes:
    1. Binary: Is this a structural zero? (no disease possible)
    2. Poisson: If not structural zero, what's the count?
    
    Useful when you have many zeros that don't follow Poisson distribution.
    
    NOTE: Requires model to output TWO values: (y_pred, pi_logits)
    """
    
    def __init__(self, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                pi_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Poisson rate predictions
            y_true: Ground truth counts
            pi_logits: Logits for probability of structural zero
            
        Returns:
            Scalar loss value
        """
        self._validate_inputs(y_pred, y_true)
        
        # Transform predictions
        lambda_pred = F.softplus(y_pred) + self.epsilon
        pi = torch.sigmoid(pi_logits)  # P(structural zero)
        
        # For y = 0: could be structural zero OR Poisson zero
        # For y > 0: must be from Poisson process
        
        zero_mask = (y_true == 0)
        
        # Loss for zeros: -log[π + (1-π)*exp(-λ)]
        zero_loss = -torch.log(
            pi[zero_mask] + 
            (1 - pi[zero_mask]) * torch.exp(-lambda_pred[zero_mask]) +
            self.epsilon
        )
        
        # Loss for non-zeros: -log[(1-π)] + Poisson NLL
        non_zero_mask = ~zero_mask
        non_zero_loss = (
            -torch.log(1 - pi[non_zero_mask] + self.epsilon) +
            (lambda_pred[non_zero_mask] - 
             y_true[non_zero_mask] * torch.log(lambda_pred[non_zero_mask]))
        )
        
        # Combine losses
        total_loss = torch.cat([zero_loss, non_zero_loss]).mean()
        
        return total_loss

import numpy as np
import torch
import torch.nn.functional as F

class OutbreakAwarePoissonLoss(BaseLoss):
    """
    Poisson Loss with heavy penalty for missing outbreaks.
    
    When ground truth has cases (y_true > 0), prediction errors are weighted more heavily.
    This ensures the model learns to detect outbreaks, not just predict zeros.
    """
    
    def __init__(self, log_input: bool = False, epsilon: float = 1e-8, 
                 outbreak_weight: float = 5.0, **kwargs):
        """
        Args:
            log_input (bool): If True, loss expects log(λ) as input. If False, expects λ.
            epsilon (float): Small value for numerical stability.
            outbreak_weight (float): How much more to penalize errors when cases > 0.
                                    Default 5.0 means missing an outbreak costs 5x more.
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.outbreak_weight = outbreak_weight
        # Use reduction='none' to get per-sample losses for weighting
        self.poisson_nll = nn.PoissonNLLLoss(log_input=log_input, full=False, reduction='none')
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute outbreak-aware Poisson loss.
        
        Args:
            y_pred: Model predictions (continuous values, can be negative)
            y_true: Ground truth counts (non-negative integers)
            
        Returns:
            Weighted loss that heavily penalizes missing outbreaks
        """
        self._validate_inputs(y_pred, y_true)
        
        # Transform predictions to positive rate parameter λ
        lambda_pred = F.softplus(y_pred) + self.epsilon
        
        # Compute per-sample Poisson NLL
        per_sample_loss = self.poisson_nll(lambda_pred, y_true)
        
        # Create weight mask: high weight when y_true > 0 (outbreak present)
        outbreak_mask = (y_true > 0).float()
        weights = torch.where(
            outbreak_mask.bool(),
            torch.full_like(outbreak_mask, self.outbreak_weight),  # High weight for outbreaks
            torch.ones_like(outbreak_mask)  # Normal weight for no outbreak
        )
        
        # Apply weights and return mean
        weighted_loss = per_sample_loss * weights
        
        return weighted_loss.mean()

def convert_poisson_predictions(pred_series, mode='mean', epsilon=1e-8):
    """
    Convert model outputs (stored in pandas Series/column) to integer predictions.
    
    Args:
        pred_series: pandas Series or array-like of raw model predictions
        mode: How to generate predictions:
            - 'mean': Use λ (expected value) and round
            - 'sample': Draw random samples from Poisson(λ)
            - 'mode': Use floor(λ) (the mode of Poisson when λ > 1)
        epsilon: Small value for numerical stability
    
    Returns:
        numpy array of integer predictions
    """
    # Convert to numpy if needed
    if hasattr(pred_series, 'values'):
        y_pred = pred_series.values
    else:
        y_pred = np.array(pred_series)
    
    # Convert to torch tensor
    y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
    
    # Transform to positive rate parameter λ (same as in loss)
    lambda_pred = F.softplus(y_pred_tensor) + epsilon
    lambda_pred_np = lambda_pred.numpy()
    
    if mode == 'mean':
        # Expected value of Poisson(λ) is λ, round to nearest integer
        return np.round(lambda_pred_np).astype(int)
    
    elif mode == 'sample':
        # Sample from Poisson distribution
        return np.random.poisson(lambda_pred_np).astype(int)
    
    elif mode == 'mode':
        # Mode of Poisson(λ) is floor(λ) when λ >= 1
        return np.floor(lambda_pred_np).astype(int)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'mean', 'sample', or 'mode'")