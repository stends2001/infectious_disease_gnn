from ..issues import Error

class DataLoaderManagerError(Error):
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)       

class ExperimentDirectoryNotFoundError(Error):
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)       

class ExperimentDirectoryInvalidError(Error):
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)      

class InvalidModelNameError(Error):
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)      

class MetricsException(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)