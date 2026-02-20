from .base import IssueBase

class Warning(IssueBase):
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        super().__init__(message, code=code, context=context)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}: {self.message}'
