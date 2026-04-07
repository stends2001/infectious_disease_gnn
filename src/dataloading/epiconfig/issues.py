from typing import List, Type
from ...issues import Warning, Error

# ======= EPICONFIG ======= #
class EpiConfigWarning(Warning):
    """
    warnings are to be printed! 
    """
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)       

class EpiConfigValidationError(Error):
    """
    errors are to be raised!
    """    
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)

class EpiConfigLimitationError(Error):
    """
    """    
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)  

class InvalidCovariatePath(Error):
    """
    """    
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)      
