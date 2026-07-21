LOSS_REGISTRY = {}
def register_loss(name: str):
    def decorator(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator