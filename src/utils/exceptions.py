class InvalidExtension(Exception):
    def __init__(self, extension_expected: str, extension_got: str):
        msg = f"Expected' {extension_expected}', got '{extension_got}'."
        super().__init__(msg)
