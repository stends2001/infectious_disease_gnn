from typing import Optional, Dict, Any, List
from torch import Tensor as Tensor

from .baseloss import BaseLoss
from ...issues import InvalidLossError

class LossHandler:
    """
    Factory and manager for loss functions.
    Handles instantiation, validation, and usage of loss functions.

    Inside `call()` we delegate to the loss-function having defined 
    its `forward()` method. As such, `loss_fn(y_hat, snapshot.y)` will
    already return the loss.

    See Also
    --------
    For more loss-function specifics, see BaseLoss and its subclasses.
    """
    def __init__(self, 
                 loss_name: str,
                 loss_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize loss handler.
        
        Parameters:
        -----------
        loss_name : str
            Name of loss function (must be in BaseLoss._registry)
        loss_kwargs : Optional[Dict[str, Any]]
            Keyword arguments to pass to loss function constructor
        """
        if loss_name not in BaseLoss._registry:
            available = list(BaseLoss._registry.keys())
            raise InvalidLossError(
                f'{loss_name}', available
            )
        
        loss_kwargs     = loss_kwargs or {}
        self.loss_name  = loss_name
        self.loss_fn    = BaseLoss._registry[loss_name](**loss_kwargs)
    
    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:     
        return self.loss_fn(y_hat, y)
    
    def __repr__(self) -> str:
        return f"LossHandler({self.loss_fn})"
    
    @staticmethod
    def list_available_losses() -> List[str]:
        """Return list of available loss functions."""
        return list(BaseLoss._registry.keys())
