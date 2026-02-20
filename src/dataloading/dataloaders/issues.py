from ...issues import Warning, Error


class DataEntryError(Error):
    """
    Wrong data entry inside GraphDataList or DeepDataList
    """    
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)
