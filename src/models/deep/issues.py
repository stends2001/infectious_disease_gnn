from ...issues import Warning, Error
from torch import Tensor as Tensor
from typing import List

# ========== DATA SHAPE ISSUES =========== #

class UnexpectedDataShape(Error):
    """
    errors are to be raised!
    """    
    def __init__(self, received_obj: str, expected_obj: str, context: str):
        message = f"context: {context} \nExcpected {expected_obj}, got {received_obj}"
        super().__init__(message, code=None, context=None)

class InconsistentDataShape(Error):
    def __init__(self, message, code = None, context = None):
        super().__init__(message, code=code, context=context)    