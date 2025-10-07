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



def temporal_smoothness_loss(y_pred, y_true, y_lag, beta=0.1):

    """

    Penalizes predictions that don't follow temporal patterns

    """
    # Standard MSE

    mse_loss = torch.mean((y_pred - y_true)**2)

    # Temporal smoothness: predictions should follow temporal trends

    # Penalize if prediction is too different from lagged value

    temporal_loss = torch.mean((y_pred - y_lag)**2)

    return mse_loss + beta * temporal_loss

def spike_detection_loss(y_pred, y_true, threshold=0.5):
    """
    Handles spike-aware loss with a fallback if no spikes exist.
    Shapes: y_pred and y_true can be [B, N] or [N]
    """
    # Flatten everything for safe indexing
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    spike_mask = y_true > threshold

    if spike_mask.sum() > 0:
        spike_loss = F.mse_loss(y_pred[spike_mask], y_true[spike_mask])
        normal_loss = F.mse_loss(y_pred[~spike_mask], y_true[~spike_mask])
        return 2.0 * spike_loss + normal_loss
    else:
        return F.mse_loss(y_pred, y_true)


def spatial_consistency_loss(y_pred, y_true, edge_index, alpha=0.1):
    """

    Penalizes predictions that don't respect spatial relationships

    """

    # Standard MSE

    mse_loss = torch.mean((y_pred - y_true)**2)

    # Spatial consistency: neighboring nodes should have similar predictions

    spatial_loss = 0

    if edge_index.shape[1] > 0:

    # Get predictions for connected nodes

        pred_i = y_pred[edge_index[0]] # Source nodes

        pred_j = y_pred[edge_index[1]] # Target nodes

        # Penalize large differences between neighbors

        spatial_loss = torch.mean((pred_i - pred_j)**2)

    return mse_loss + alpha * spatial_loss