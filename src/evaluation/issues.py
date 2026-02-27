from typing import List, Type
from ..issues import Warning, Error

class MetricError(Error):
    """
    warnings are to be printed! 
    """
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)       
