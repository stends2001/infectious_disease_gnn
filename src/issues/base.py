class IssueBase(Exception):
    """
    Base class for all custom issues (errors and warnings).

    Attributes:
    -----------
    message: str
        Human-readable description.
    code: str | None
        Optional machine-readable code.
    context: str | None
        Optional context.
    """
    def __init__(self, message: str, *, code: str | None = None, context: str | None = None):
        self.message    = message
        self.code       = code
        self.context    = context
        
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.code:
            parts.append(f"(code: {self.code})")
        if self.context:
            parts.append(f"[context: {self.context}]")
        return " ".join(parts)
