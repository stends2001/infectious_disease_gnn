
from .exp_decay import ExponentialDecayLoss
from .standardmse import StandardMSELoss

LOSS_REGISTRY = {
    'exp_decay': ExponentialDecayLoss,
    'mse': StandardMSELoss,
}

def get_loss(name, **kwargs):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss {name}")
    return LOSS_REGISTRY[name](**kwargs)






