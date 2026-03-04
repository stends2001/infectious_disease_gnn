import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, Any, List
from abc import ABC, abstractmethod

from .baseloss import BaseLoss


import torch
import torch.nn as nn

# class PinballLoss(BaseLoss):
#     """
#     Quantile (pinball) loss for multi-quantile forecasting.

#     Parameters
#     ----------
#     quantils : list[float]
#         e.g. [0.1, 0.5, 0.9]. Must match the order of the quantile
#         dimension in model output.

#     reduction : Literal['mean', 'sum']
#         how to reduce across all elements
#     """

#     def __init__(self, quantiles: list[float], reduction: str = 'mean'):
#         super().__init__()

#         if not all(0 < q < 1 for q in quantiles):
#             raise ValueError("All quantile levels must be strictly between 0 and 1.")

#         self.quantiles = quantiles  # plain Python list, useful for labelling etc.
#         self.register_buffer('quantile_levels', torch.tensor(quantiles, dtype=torch.float32))
#         self.reduction = reduction

#     def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#         """
#         y_pred : [num_nodes, horizon_size, num_quantiles]
#         y_true : [num_nodes, horizon_size]
#         """
#         targets_expanded = y_true.unsqueeze(-1).expand_as(y_pred)
#         errors = targets_expanded - y_pred  # positive => under-prediction

#         q = self.quantile_levels.to(errors.device)
#         loss = torch.where(errors >= 0, q * errors, (q - 1) * errors)

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             raise ValueError(f"Unknown reduction '{self.reduction}'")
        

class PinballLoss(BaseLoss):
    def __init__(self, quantiles, mse_weight=1.0, reduction='mean'):
        super().__init__()

        if not all(0 < q < 1 for q in quantiles):
            raise ValueError("Quantiles must be strictly between 0 and 1.")

        self.quantiles = quantiles
        self.register_buffer(
            "quantile_levels",
            torch.tensor(quantiles, dtype=torch.float32)
        )

        if 0.5 not in quantiles:
            raise ValueError("0.5 must be included to apply MSE on median.")

        self.median_idx = quantiles.index(0.5)
        self.mse_weight = mse_weight
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, y_pred, y_true):
        # --- pinball ---
        targets_expanded = y_true.unsqueeze(-1).expand_as(y_pred)
        errors = targets_expanded - y_pred

        q = self.quantile_levels.to(errors.device)
        pinball = torch.where(errors >= 0, q * errors, (q - 1) * errors)

        if self.reduction == "mean":
            pinball_loss = pinball.mean()
        elif self.reduction == "sum":
            pinball_loss = pinball.sum()
        else:
            raise ValueError

        # --- mse on median quantile ---
        median_pred = y_pred[..., self.median_idx]
        mse_loss = self.mse(median_pred, y_true)

        return pinball_loss + self.mse_weight * mse_loss        

class PinchPinballLoss(BaseLoss):
    def __init__(self, quantiles, mse_weight=1.0, pinch_weight=0.1, reduction='mean'):
        super().__init__()
        if not all(0 < q < 1 for q in quantiles):
            raise ValueError("Quantiles must be strictly between 0 and 1.")
        self.quantiles = quantiles
        self.register_buffer("quantile_levels", torch.tensor(quantiles, dtype=torch.float32))
        if 0.5 not in quantiles:
            raise ValueError("0.5 must be included to apply MSE on median.")
        self.median_idx = quantiles.index(0.5)
        self.mse_weight = mse_weight
        self.pinch_weight = pinch_weight
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, y_pred, y_true):
        targets_expanded = y_true.unsqueeze(-1).expand_as(y_pred)
        errors = targets_expanded - y_pred

        q = self.quantile_levels.to(errors.device)
        pinball = torch.where(errors >= 0, q * errors, (q - 1) * errors)

        if self.reduction == "mean":
            pinball_loss = pinball.mean()
        elif self.reduction == "sum":
            pinball_loss = pinball.sum()
        else:
            raise ValueError

        # median MSE
        median_pred = y_pred[..., self.median_idx]
        mse_loss = self.mse(median_pred, y_true)

        # --- new pinch term ---
        diffs = y_pred[..., 1:] - y_pred[..., :-1]
        pinch_loss = torch.mean(torch.relu(-diffs) + diffs ** 2)  # soft monotone + gap penalty

        return pinball_loss + self.mse_weight * mse_loss + self.pinch_weight * pinch_loss        