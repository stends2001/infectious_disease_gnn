from typing import Optional, Dict, Any, List, TYPE_CHECKING
import torch

from .standard_losses import MSELoss, MAELoss, HuberLoss, SmoothL1Loss, ExponentialDecayLoss, PolynomialDecayLoss, WeightedMSELoss
from .outbreak_losses import FocalLoss, OutbreakWeightedLoss
from .poissonloss import PoissonLoss, NegativeBinomialLoss,ZeroInflatedPoissonLoss, OutbreakAwarePoissonLoss
from .classification import BCELoss, BCELogitLoss
from .asymmetricmse import AsymmetricMSELoss
from .quantile import QuantileLoss
from .pinball import PinballLoss

LOSS_REGISTRY: Dict[str, type] = {
    'mse':              MSELoss,
    'mae':              MAELoss,
    'huber':            HuberLoss,
    'smooth_l1':        SmoothL1Loss,
    'exp_decay':        ExponentialDecayLoss,
    'poly_decay':       PolynomialDecayLoss,
    'asymmetric_mse':   AsymmetricMSELoss,
    'quantile' :        QuantileLoss    ,
    'weighted_mse' :    WeightedMSELoss,
    'focal' :           FocalLoss,
    'outbreakweighted': OutbreakWeightedLoss,
    'poisson':          PoissonLoss,
    'binomial':         NegativeBinomialLoss,
    'poisson3':         ZeroInflatedPoissonLoss,
    'outbreakpoisson':  OutbreakAwarePoissonLoss,
    'bce':              BCELoss,    
    'bcelogit':         BCELogitLoss,
    'pinball':          PinballLoss,
}


class LossHandler:
    """
    Factory and manager for loss functions.
    Handles instantiation, validation, and usage of loss functions.
    """
    
    def __init__(self, 
                 loss_name: str,
                 loss_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize loss handler.
        
        Parameters:
        -----------
        loss_name : str
            Name of loss function (must be in LOSS_REGISTRY)
        loss_kwargs : Optional[Dict[str, Any]]
            Keyword arguments to pass to loss function constructor
        """
        if loss_name not in LOSS_REGISTRY:
            available = ', '.join(LOSS_REGISTRY.keys())
            raise ValueError(
                f"Unknown loss '{loss_name}'. Available: {available}"
            )
        
        loss_kwargs     = loss_kwargs or {}
        self.loss_name  = loss_name
        self.loss_fn    = LOSS_REGISTRY[loss_name](**loss_kwargs)
    
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        return self.loss_fn(y_pred, y_true)
    
    def __repr__(self) -> str:
        return f"LossHandler({self.loss_fn})"
    
    @staticmethod
    def list_available_losses() -> List[str]:
        """Return list of available loss functions."""
        return list(LOSS_REGISTRY.keys())
    
    @staticmethod
    def get_loss_info(loss_name: str) -> str:
        """Get docstring for a specific loss."""
        if loss_name not in LOSS_REGISTRY:
            return f"Loss '{loss_name}' not found"
        return LOSS_REGISTRY[loss_name].__doc__ or "No documentation available"