
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import torch
import numpy as np
from tqdm import tqdm
from ...utils.textformatting import warning_emoji

def desequentialize_x(x: torch.Tensor, expected_seq_length: int):
    """for models in which we carry over hidden states, we do not require sequentialized X."""
    if x.ndim == 3:
        if x.shape[-1] == expected_seq_length:
            x_squeezed = x[:,:,-1]
            x_squeezed.squeeze(-1)
        elif x.shape[-1] == 1:
            x_squeezed = x.squeeze(-1)
        # print(f'x is desequentialized, now has shape:{x_squeezed.shape}')            
        return x_squeezed
    else:
        print(f'{warning_emoji}desequentialize_x was called but x has unexpected dimensions')

    

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


