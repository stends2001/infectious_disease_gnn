class InvalidExtension(Exception):
    def __init__(self, extension_expected: str, extension_got: str):
        msg = f"Expected' {extension_expected}', got '{extension_got}'."
        super().__init__(msg)

class AttributeNotFound(Exception):
    def __init__(self, missing_attr: str, class_name: str):
        msg = f"no attribute {missing_attr} in {class_name}"
        super().__init__(msg)
