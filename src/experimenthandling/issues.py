class DataLoaderManagerError(Exception):
    def __init__(self, message: str):
        super().__init__(message)       

class ExperimentDirectoryNotFoundError(Exception):
    def __init__(self, message: str):
        super().__init__(message)       

class ExperimentDirectoryInvalidError(Exception):
    def __init__(self, message: str):
        super().__init__(message)      

class InvalidModelNameError(Exception):
    def __init__(self, message: str):
        super().__init__(message)      

class MetricsException(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)