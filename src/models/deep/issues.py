from torch import Tensor as Tensor
from typing import List

# ========== DATA SHAPE ISSUES =========== #

class UnexpectedDataShape(Exception):
    """
    errors are to be raised!
    """    
    def __init__(self, received_obj: str, expected_obj: str, context: str):
        message = f"context: {context} \nExcpected {expected_obj}, got {received_obj}"
        super().__init__(message)

class InconsistentDataShape(Exception):
    def __init__(self, message, code = None, context = None):
        super().__init__(message)    