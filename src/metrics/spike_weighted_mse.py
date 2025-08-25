import torch
import torch.nn.functional as F

def spike_weighted_mse(y_pred, y_true):
    weights = 1 + torch.abs(y_true)  # weight more on higher target magnitude
    return torch.mean(weights * (y_pred - y_true)**2)

def mse(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

def spike_timing_weighted_mse(y_pred, y_true):
    # Base weight: scale by magnitude
    base_weight = 1 + torch.abs(y_true)
    
    # Emphasize time points where the true values sharply rise (first derivative)
    # Approximate temporal diff along batch dimension (assuming batch = time)
    dy = torch.abs(y_true[1:] - y_true[:-1])
    dy = F.pad(dy, (1, 0))  # pad to match y_true shape
    
    # Scale weights by the amount of change (larger weight where target changes a lot)
    weights = base_weight * (1 + 5 * dy)  # 5 is a hyperparam to tune
    
    return torch.mean(weights * (y_pred - y_true)**2)
