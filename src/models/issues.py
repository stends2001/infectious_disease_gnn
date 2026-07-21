from typing import List

# === MODEL ERRORS ===== #

class ModelError(Exception):
    def __init__(self, message: str):
        super().__init__(message)   

class ModelInitError(Exception):
    def __init__(self, message: str):
        super().__init__(message)    

class ModelStatusError(Exception):
    def __init__(self, message: str):
        super().__init__(message) 

class FutureUpdateError(Exception):
    """
    for something that I should implement at somepoint but haven't yet
    """
    def __init__(self, message: str):
        super().__init__(message)

class DeviceWarning(Exception):
    """
    warnings are to be printed! 
    """
    def __init__(self, message: str):
        super().__init__(message)  

# ===== DEEPMODEL FACTORY METHODS ====== #

class InvalidOptimizerError(Exception):
    def __init__(self, optimizer_name: str, supported_optimizers: List[str]):
        message = f'Invalid optimizer {optimizer_name}. Supported optimzers are {supported_optimizers}'
        super().__init__(message)        

class InvalidLossError(Exception):
    def __init__(self, loss_name: str, supported_losses: List[str]):
        message = f'Invalid loss {loss_name}. Supported losses are {supported_losses}'
        super().__init__(message)         

class InvalidSchedulerError(Exception):
    def __init__(self, scheduler_name: str, supported_schedulers: List[str]):
        message = f'Invalid scheduler {scheduler_name}. Supported schedulers are {supported_schedulers}'
        super().__init__(message)                             

# ===== DEEPMODEL FACTORY METHODS ====== #

class MissingPredictionsError(Exception):
    """
    errors are to be raised!
    """    
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message)

class InvalidPredictionsError(Exception):
    """
    errors are to be raised!
    """    
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message)
