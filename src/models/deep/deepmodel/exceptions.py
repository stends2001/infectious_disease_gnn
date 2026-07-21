from typing import List 

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