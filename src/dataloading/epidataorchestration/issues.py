from ...issues.warnings import Warning

# ======= EPICONFIG ======= #

class EpiConfigWarning(Warning):
    def __init__(self, statement: str):
        super().__init__(statement)

class EpiConfigError(Exception):
    def __init__(self, explanation: str):
        statement = "Epiconfig couldn't be loaded" + "\n" + explanation
        super().__init__(statement)

class CurrentEpiConfigError(Exception):
    def __init__(self, explanation: str):
        statement = "Epiconfig couldn't be loaded" + "\n" + explanation
        super().__init__(statement)

# ======== COLUMN_REGISTRY ====== #

class ColEntryError(Exception):
    """Base class for all column entry related errors."""
    def __init__(self, entryname: str, message: str):
        self.entryname  = entryname
        self.message    = message
        super().__init__(f"{self.__class__.__name__} - {self.message}")

class InvalidColEntryError(ColEntryError):
    """Raised when a column entry is invalid."""
    def __init__(self, entryname: str, explanation: str):
        message = f"Column Registration entry '{entryname}' is invalid. {explanation}"
        super().__init__(entryname, message)    

class ColEntryMissingError(ColEntryError):
    """Raised when a column entry is missing."""
    def __init__(self, entryname: str):
        message = f"Column Registration entry '{entryname}' is missing."
        super().__init__(entryname, message)

class ColEntryMissingTransformationError(ColEntryError):
    """Raised when a transformation is invalid."""
    def __init__(self, entryname: str):
        message = f"Column Registration entry '{entryname}' has transformation_group None (independent) but no transformation attribute was found."
        super().__init__(entryname, message)

class ColEntryMissingTransformationReferralError(ColEntryError):
    """Raised when a transformation referral is invalid."""
    def __init__(self, entryname: str, referral: str):
        message = f"Column Registration entry '{entryname}' has transformation_group {referral} for which no transformation attribute was found."
        super().__init__(entryname, message)

# ======== DATAORCHESTRATION =========== #

class DataOrchestrationError(Exception):
    def __init__(self, explanation: str):
        statement = "Data Orchestration couldn't be run" + "\n" + explanation
        super().__init__(statement)    

class DataOrchestrationContainerNotFound(Exception):
    def __init__(self, datastage: str, previous_method: str):
        super().__init__(f"No {datastage} attribute found for DataOrchestrator. Run {previous_method}() first")        

# ======= TEMPORAL SUMMARY ==== #
class TemporalError(Exception):
    def __init__(self, message: str):
        super().__init__(f'Invalid EpiDataTemporalSummary: {message}')