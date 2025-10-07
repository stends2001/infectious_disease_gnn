import torch
import torch.nn as nn

class ExponentialDecayLoss(nn.Module):
    def __init__(self, gamma=0.9):
        super().__init__()
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        horizon = y_pred.size(1)
        weights = torch.tensor([self.gamma ** t for t in range(horizon)],
                               dtype=y_pred.dtype,
                               device=y_pred.device)
        se = (y_pred - y_true) ** 2
        weighted_se = se * weights
        return weighted_se.mean()