from typing import List, Type
from ....issues import Warning, Error

# ======== EPIDATAORCHESTRATION =========== #

class EpiDataOrchestrationError(Error):
    def __init__(self, message: str):
        statement = f"Data Orchestration couldn't be run; {message}"
        super().__init__(statement)    

class MissingEpiDataContainer(Error):
    def __init__(self, datastage: str, previous_method: str):
        super().__init__(f"No {datastage} attribute found for DataOrchestrator. Run {previous_method}() first")     

class NonExistentAttributeEpiDataContainer(Error):
    def __init__(self, class_name: str, attribute_name: str):
        message= f"Attribute {attribute_name} does not exist in {class_name}"
        super().__init__(message, code=None, context=None)               

# ======= TEMPORAL SUMMARY ==== #
class TemporalError(Error):
    def __init__(self, message: str):
        super().__init__(f'Invalid EpiDataTemporalSummary: {message}')