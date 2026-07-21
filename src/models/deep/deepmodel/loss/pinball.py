import torch
import torch.nn as nn
from typing import List, Tuple
from .baseloss import BaseLoss

class PinballLoss(BaseLoss):
    """ 
    Pinball (quantile) loss for probabilistic forecasting.

    Trains the model to predict multiple quantiles simultaneously by penalising
    over- and under-predictions asymmetrically. For a quantile level q, 
    underpredictions are penalised by q and overpredictions by (1 - q), 
    so the model learns to predict the q-th quantile of the target distribution.

    An additional MSE term on the median quantile (q=0.5) is included to 
    anchor the central prediction and stabilise training.
    """
    quantile_levels: torch.Tensor

    def __init__(self, quantiles: List[float], mse_weight: float = 1.0):
        super().__init__()

        if not all(0 < q < 1 for q in quantiles):
            raise ValueError("Quantiles must be strictly between 0 and 1.")
        if 0.5 not in quantiles:
            raise ValueError("0.5 must be included for median MSE.")

        self.register_buffer(
            "quantile_levels",
            torch.tensor(quantiles, dtype=torch.float32)
        )

        self.median_idx = quantiles.index(0.5)
        self.mse_weight = mse_weight
        self.mse = nn.MSELoss()

    def normalize(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return y_pred, y_true.unsqueeze(-1)

    def compute(self, y_pred, y_true):
        errors = y_true - y_pred

        q = self.quantile_levels.to(y_pred.device)  # ← move to same device

        pinball = torch.where(
            errors >= 0,
            q * errors,
            (q - 1) * errors
        )

        pinball_loss = pinball.mean()

        median_pred = y_pred[..., self.median_idx]
        mse_loss = self.mse(median_pred, y_true.squeeze(-1))

        return pinball_loss + self.mse_weight * mse_loss