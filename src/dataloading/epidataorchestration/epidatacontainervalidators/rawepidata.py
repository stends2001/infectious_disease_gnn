from typing import List, Dict, Any
import pandas as pd 

from .base import EpiDataContainerValidator
from .issues import MissingColumnError
from ..epidatacontainers import RawEpiData
from ...epiconfig.epiconfig import EpiConfig

class RawValidator(EpiDataContainerValidator):
    """ 
    Validator of RawEpiData classes
    Validates the existent attributes and that they
    are not empty. Also validates the presence of 
    columns "level" and "key".

    Examples
    --------
    >>> RawValidator(orchestrator.config, orchestrator.data_raw).validate()

    See Also
    --------
    For more information, please see EpiDataContainerValidator
    """

    def __init__(self,
                 epiconfig:  'EpiConfig',
                 rawepidata: 'RawEpiData'):

        super().__init__(epiconfig, 
                         dataclass_validated='RawEpiData')

        self.rawepidata= rawepidata
        self.cols      = ['level','key']

    def validate(self):
        """
        We validate that all attributes have length (they're all a dataframe (pd or gpd) or a dictionary)
        Attempting to access a missing attribute will raise an error internally to the containers, so we
        dont need to deal with that here.

        We loop through all attributes
        """
        attrs           = self._get_expected_attributes()
        
        for attr_name in attrs:

            # retrieve attribute
            stored_attribute = getattr(self.rawepidata, attr_name)

            # validate the type
            self._validate_type(attr_name, stored_attribute)

            # validate the size 
            self._validate_length_nonzero(attr_name, stored_attribute)
        
            # validate mandatory columns being present
            if isinstance(stored_attribute, pd.DataFrame) and attr_name not in ['disease','region_harmonization']:
                 self._validate_presence_columns(attr_name, stored_attribute)
                
    def _validate_presence_columns(self, attribute_name: str, stored_attribute: pd.DataFrame):
        for col in self.cols:
            if col not in stored_attribute:
                raise MissingColumnError(attribute_name, col, self.dataclass_validated)        

    def _get_expected_attributes(self) -> List[str]:
        """returns a list of strings with expected attributes"""
        # these are mandatory
        expected_attributes = ['disease','population_size','shapedata','region_harmonization','tokenization_map']

        if self.epiconfig.feature_popdens:
            expected_attributes.append('population_density')
        
        if self.epiconfig.feature_popage:
            expected_attributes.append('population_age')            

        if self.epiconfig.feature_gisd:
            expected_attributes.append('gisd')                   

        if self.epiconfig.feature_kreise_classes:
            expected_attributes.append('kreise_classes')                 

        if self.epiconfig.feature_borders:
            expected_attributes.append('borders')          

        return expected_attributes