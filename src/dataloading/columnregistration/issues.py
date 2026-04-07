from ...issues.errors import Error

class ColumnRegistryError(Error):
    """
    Base class for all ColumnRegistry-related errors
    """
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)       

class InvalidColEntry(ColumnRegistryError):
    """
    Raised when a column entry is invalid
    """
    def __init__(self, entryname: str, explanation: str):
        message = f"Entry '{entryname}' is invalid. {explanation}"
        super().__init__(message)    

class MissingColEntry(ColumnRegistryError):
    """
    Raised when looking for a column entry that doesn't exist
    """
    def __init__(self, entryname: str):
        message = f"Entry '{entryname}' doesn't exist."
        super().__init__(message)        

class MissingTransformation(ColumnRegistryError):
    """
    Raised when a transformation is invalid
    """
    def __init__(self, entryname: str):
        message = f"Entry'{entryname}' has transformation_group 'self' (independent) but no transformation attribute was found."
        super().__init__(message)

class MissingTransformationReferral(ColumnRegistryError):
    """
    Raised when a transformation referral is invalid
    """
    def __init__(self, entryname: str, referral: str):
        message = f"Entry '{entryname}' has transformation_group {referral} for which no transformation attribute was found."
        super().__init__(message)