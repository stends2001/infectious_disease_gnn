from ...issues import Warning, Error

class FutureUpdateError(Error):
    """
    for something that I should implement at somepoint but haven't yet
    """
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)

class ModelStatusError(Error):
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)    


class InvalidPredictionsError(Error):
    """
    errors are to be raised!
    """    
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)
