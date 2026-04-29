import torch
import torch.nn as nn
from typing import Dict, Type, Tuple
from abc import ABC, abstractmethod

class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for all loss functions

    Each loss class is supposed to have its own `compute()` method and,
    when necessary, also its `normalize()` method.

    Methods
    -------
    As all loss classes inherit from torch.nn.Module, the `forward()` method
    is implicityly called, which is done in `LossHandler.__call__()`. `forward()`
    is not overwritten here, and always consists of two helper methods being called;
    `normalize()` and `compute()`.    

    `normalize()` may be used by some classes that expect a different shape of predictions.
    By default we expect predictions of shape [nodes, horizon, quantile], even if n_quantiles = 1.
    In that case, there is a mapping of 1 prediction value per 1 target value.
    When predicting uncertainties though, we'll have multiple prediction values per 1 target
    value (final dimension quantile will > 1). Relevant losses (PinballLoss and PincPinballLoss) 
    therefore have their own, different version of `normalize()` in which target gets a trailing dimension
    of size 1.

    `compute()` is the represents computation of the loss value. These are implemented by every class.

    Inheritance structure ensures the registration of each subclass in `._registry` 
    """

    _registry: Dict[str, Type['BaseLoss']] = {}
        
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
    
    # init - subclass dunder. This code is reun at "class creation time", not at run-time.
    # so each class that inherits from this main BaseLoss runs this, and registers itself
    # as a loss method.
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # strip 'loss' from classname and register
        # the class "MSELoss" is registered as "mse"
        key = cls.__name__.lower().replace('loss', '')
        BaseLoss._registry[key] = cls

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred, y_true = self.normalize(y_pred, y_true)
        return self.compute(y_pred, y_true)

    def normalize(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Default normalization for point prediction losses.
        Squeezes trailing quantile dimension if present.
        Quantile losses override this with their own normalize().
        """
        if y_pred.shape[-1] != 1:
            raise ValueError(f'expected y_pred with final dimension of size 1. Got {y_pred.shape}')
        y_pred = y_pred.squeeze(-1)
        return y_pred, y_true

    @abstractmethod
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pass
    
    def __repr__(self) -> str:
        kwargs_str = ', '.join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"{self.__class__.__name__}({kwargs_str})"
    
# for __init_subclass__ to run, the classes must be imported at some point
from .standard_losses import MSELoss, MAELoss, HuberLoss, WeightedMSELoss 
from .pinball import PinballLoss
from .pinchball import PinchPinballLoss