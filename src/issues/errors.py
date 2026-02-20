from .base import IssueBase

class Error(IssueBase):
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)  

class WissdatenMountingError(Error):
    def __init__(self, path: str):
        super().__init__(f"Wissdaten not mounted at path: {path}")