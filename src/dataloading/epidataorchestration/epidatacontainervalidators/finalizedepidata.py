from typing import List
import pandas as pd 

from .base import EpiDataContainerValidator
from .issues import MissingColumnError, NaNsFoundError
from ..epidatacontainers import FinalizedEpiData
from ...epiconfig import EpiConfig
from ...columnregistration.column_registry import ColumnRegistration

class FinalizedValidator(EpiDataContainerValidator):
    """ 

    """

    def __init__(self,
                 epiconfig:  'EpiConfig',
                 column_registry: ColumnRegistration,
                 finalizedepidata: 'FinalizedEpiData'):

        super().__init__(epiconfig, 
                         dataclass_validated='FinalizedEpiData')

        self.finalizedepidata  = finalizedepidata
        self.column_registry = column_registry

    def validate(self):
        """
        """
        attrs           = self._get_expected_attributes()

        for attr_name in attrs:

            # retrieve attribute
            stored_attribute = getattr(self.finalizedepidata, attr_name)

            # validate the type
            self._validate_type(attr_name, stored_attribute)

            # validate the size 
            self._validate_length_nonzero(attr_name, stored_attribute)
                           
            if isinstance(stored_attribute, pd.DataFrame):
                
                # validate mandatory columns being present                
                self._validate_presence_columns(attr_name, stored_attribute)

                # validate absence of NaNs
                self._validate_nan(attr_name, stored_attribute)                
               
    def _validate_nan(self, attribute_name: str, stored_attribute: pd.DataFrame):
        nan_columns = stored_attribute.columns[stored_attribute.isna().any()].tolist()
        if nan_columns:     
            raise NaNsFoundError(attribute_name, self.dataclass_validated, nan_columns)

    def _validate_presence_columns(self, attribute_name: str, stored_attribute: pd.DataFrame):

        # still need to add the dynamically adjusted target columns
        all_cols = self.column_registry.context_columns + self.column_registry.feature_columns + ["pred"]

        for col in all_cols:
            if col not in stored_attribute:
                raise MissingColumnError(attribute_name, col, self.dataclass_validated)        
        
    def _get_expected_attributes(self) -> List[str]:
        """returns a list of strings with expected attributes"""
        # these are mandatory
        expected_attributes = ['data','data_denorm']       

        return expected_attributes