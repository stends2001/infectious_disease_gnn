from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import pandas as pd

from .issues import UnexpectedAttributeTypeError, EmptyAttributeTypeError
from ...epiconfig.epiconfig import EpiConfig

class EpiDataContainerValidator(ABC):
    """ 
    Parent class to all epidataorchestrator - containers
    validators.

    Subclasses may have any helper methods, but must
    have one centralized `.validate()` method that 
    orchestrates and runs the validation.

    Some methods are shared and therefore defined in this parent
    class. This includes:
    - `validate_type()`
    - `validate_length_nonzero()`
    """

    def __init__(self,
                epiconfig:              EpiConfig,
                dataclass_validated:    str):
        
        self.epiconfig          =  epiconfig
        self.dataclass_validated= dataclass_validated

        self.allowed_types   = (pd.DataFrame, Dict)

    @abstractmethod
    def validate(self):
        pass

    def _validate_type(self, attribute_name: str, stored_attribute: Any):
        """validates type of attribute. Must be one of `self.allowed_types`."""
        if not isinstance(stored_attribute, self.allowed_types):

            raise UnexpectedAttributeTypeError(attribute_name, 
                                               self.dataclass_validated, 
                                               str(type(stored_attribute)), 
                                               [str(obj) for obj in self.allowed_types])
        
    def _validate_length_nonzero(self, attribute_name: str, stored_attribute: Union[pd.DataFrame, Dict]):
        """validates that attribute is not emtpy"""
        if not len(stored_attribute) > 0:
            raise EmptyAttributeTypeError(attribute_name, self.dataclass_validated)
                    