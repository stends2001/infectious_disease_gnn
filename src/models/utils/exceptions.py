class ModelInitError(Exception):
    def __init__(self, message: str):
        super().__init__(message)    

class ModelStatusError(Exception):
    def __init__(self, message: str):
        super().__init__(message) 