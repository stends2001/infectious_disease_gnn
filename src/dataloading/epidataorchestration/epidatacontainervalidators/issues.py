from typing import List

from ....issues import Error

class UnexpectedAttributeTypeError(Error):
    
    def __init__(self, attribute_name: str, cls_name: str, found_type: str, allowed_types: List[str]):
        message = f"Attribute {attribute_name} has an unexpected type in {cls_name}. Got {found_type} but only {allowed_types} are allowed"
        super().__init__(message)

class EmptyAttributeTypeError(Error):
    
    def __init__(self, attribute_name: str, cls_name: str):
        message = f"Attribute {attribute_name} in {cls_name} is empty"
        super().__init__(message)

class MissingColumnError(Error):
    
    def __init__(self, attribute_name: str, column_name: str, cls_name: str):
        message = f"Attribute {attribute_name} in {cls_name} has no column {column_name}"
        super().__init__(message)        

class InvalidTokenizationError(Error):
    
    def __init__(self, attribute_name: str, cls_name: str, missing_tokens: List[int], leftover_tokens: List[int]):
        message = f"Attribute {attribute_name} in {cls_name} is not properly tokenized. Missing tokens: {missing_tokens}. Leftover tokens: {leftover_tokens}"
        super().__init__(message)        

class NaNsFoundError(Error):
    
    def __init__(self, attribute_name: str, cls_name: str, columns: List[str]):
        message = f"Attribute {attribute_name} in {cls_name} has NaNs in columns {columns}."
        super().__init__(message)        

class InvalidNormalizationError(Error):

    def __init__(self, attribute_name: str, cls_name: str, column: str, specifications: str):
        message = f"Attribute {attribute_name} in {cls_name} has an invalid normaliztaion for {column}.\n{specifications}"
        super().__init__(message)        