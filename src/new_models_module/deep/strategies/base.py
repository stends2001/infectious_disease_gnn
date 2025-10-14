
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import torch
import numpy as np
from tqdm import tqdm

# ============================================================================
# TRAINING STRATEGIES - Encapsulate the differences
# ============================================================================

class Strategy(ABC):
    """Base class for model strategies that handle training and forecasting"""
    
    @abstractmethod
    def training_step(self, model: torch.nn.Module, snapshot: Any, optimizer, loss_fn) -> float:
        """Execute one training step. Returns loss value."""
        pass
    
    @abstractmethod
    def validation_step(self, model: torch.nn.Module, snapshot: Any, loss_fn) -> float:
        """Execute one validation step. Returns loss value."""
        pass
    
    @abstractmethod
    def forecast_step(self, model: torch.nn.Module, snapshot: Any, loss_fn) -> Tuple[torch.Tensor, float]:
        """Execute one forecasting step. Returns (predictions, loss_value)."""
        pass
    
    @abstractmethod
    def reset_state(self):
        """Reset any state (e.g., hidden states) between epochs or datasets"""
        pass