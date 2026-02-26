from ..issues import Warning, Error

from typing import List

# === MODEL ERRORS ===== #

class ModelError(Error):
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)   

class ModelInitError(Error):
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)    

class ModelStatusError(Error):
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context) 

class FutureUpdateError(Error):
    """
    for something that I should implement at somepoint but haven't yet
    """
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)

class DeviceWarning(Warning):
    """
    warnings are to be printed! 
    """
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)  

# ===== DEEPMODEL FACTORY METHODS ====== #

class InvalidOPtimizerError(Error):
    def __init__(self, optimizer_name: str, supported_optimizers: List[str]):
        message = f'Invalid optimizer {optimizer_name}. Supported optimzers are {supported_optimizers}'
        super().__init__(message, code=None, context=None)        

class InvalidLossError(Error):
    def __init__(self, loss_name: str, supported_losses: List[str]):
        message = f'Invalid loss {loss_name}. Supported losses are {supported_losses}'
        super().__init__(message, code=None, context=None)         

class InvalidSchedulerError(Error):
    def __init__(self, scheduler_name: str, supported_schedulers: List[str]):
        message = f'Invalid scheduler {scheduler_name}. Supported schedulers are {supported_schedulers}'
        super().__init__(message, code=None, context=None)                             

# ===== DEEPMODEL FACTORY METHODS ====== #

class MissingPredictionsError(Error):
    """
    errors are to be raised!
    """    
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)

class InvalidPredictionsError(Error):
    """
    errors are to be raised!
    """    
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)
