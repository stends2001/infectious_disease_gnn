class MissingPredictionsError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class InvalidPredictionsError(Exception):   
    def __init__(self, message: str):
        super().__init__(message)