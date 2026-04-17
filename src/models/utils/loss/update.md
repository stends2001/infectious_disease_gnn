for the coding group, I'd like to work out the following in my loss classes:
LOSS_REGISTRY = {}

def register_loss(name: str):
    def decorator(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator

@register_loss('mse')
class MSELoss(BaseLoss):
    ...

@register_loss('pinball')
class PinballLoss(BaseLoss):
    ...