from typing import List, assert_never
import pandas as pd 

from .base import EpiDataContainerValidator
from .issues import MissingColumnError, NaNsFoundError, InvalidNormalizationError
from ..epidatacontainers import NormalizedEpiData
from ...columnregistration.column_registry import ColumnRegistration
from ...epiconfig import EpiConfig

class NormalizedValidator(EpiDataContainerValidator):
    """ 

    """

    def __init__(self,
                 epiconfig:  'EpiConfig',
                 column_registry: ColumnRegistration,                 
                 normalizedepidata: 'NormalizedEpiData'):

        super().__init__(epiconfig, 
                         dataclass_validated='NormalizedEpiData')

        self.normalizedepidata= normalizedepidata
        self.column_registry  = column_registry
        self.cols             = [c.column_name for c in column_registry.columns]

    def validate(self):
        """
        """
        attrs           = self._get_expected_attributes()

        for attr_name in attrs:

            # retrieve attribute
            stored_attribute = getattr(self.normalizedepidata, attr_name)

            # validate the type
            self._validate_type(attr_name, stored_attribute)

            # validate the size 
            self._validate_length_nonzero(attr_name, stored_attribute)
                           
            # validate mandatory columns being present                
            self._validate_presence_columns(attr_name, stored_attribute)

            # validate absence of NaNs
            self._validate_nan(attr_name, stored_attribute)

            # validate normalization 
            self._validate_normalization(attr_name, stored_attribute)

    def _validate_normalization(self, attribute_name: str, stored_attribute: pd.DataFrame):

        tolerance = 1e-6

        # normalization was done bsed on train df
        train_df = stored_attribute[stored_attribute['train']]

        for col_entry in self.column_registry.columns:

            match (col_entry.transformation, col_entry._transformation_group):
            
                # Skip columns that don't have normalization attribute
                case (False, _):
                    continue # continue with next col_entry
                  
                # Determine which normalization parameters to use
                # independent transformation first
                case (True, 'self'):
                
                    # example: {'zscore': {'mean': np.float64(0.12332682669392722), 'std': np.float64(0.31541569993631763)}}
                    normalization_dict  = col_entry._transformation_params['normalization']
                    
                    for normalization_funcname, params_dict in normalization_dict.items():
                        
                        if normalization_funcname == 'zscore' :
                            expected_mean = 0
                            expected_std  = 1.0

                            computed_mean = train_df[col_entry.column_name].mean()
                            computed_std  = train_df[col_entry.column_name].std()

                            if abs(computed_mean - expected_mean) > tolerance:
                                specs = f"Expected mean about 0 (tolerance {tolerance}). Got {computed_mean}"
                                raise InvalidNormalizationError(attribute_name, self.dataclass_validated, col_entry.column_name, specs)
                            
                            if abs(computed_std - expected_std) > tolerance:
                                specs = f"Expected std about 1.0 (tolerance {tolerance}). Got {computed_std}"
                                raise InvalidNormalizationError(attribute_name, self.dataclass_validated, col_entry.column_name, specs)                            

                        elif normalization_funcname == 'minmax':
                            expected_max = 1.0
                            expected_min = 0 

                            computed_max = train_df[col_entry.column_name].max()
                            computed_min  = train_df[col_entry.column_name].min()

                            if abs(computed_max - expected_max) > tolerance:
                                specs = f"Expected max about 1.0 (tolerance {tolerance}). Got {computed_max}"
                                raise InvalidNormalizationError(attribute_name, self.dataclass_validated, col_entry.column_name, specs)
                            
                            if abs(computed_min - expected_min) > tolerance:
                                specs = f"Expected min about 0 (tolerance {tolerance}). Got {computed_min}"
                                raise InvalidNormalizationError(attribute_name, self.dataclass_validated, col_entry.column_name, specs)                                         

                        else:
                            raise ValueError(f'no expected normalization - validation checks existent forn normalization method {normalization_funcname}')
            
                # dependent transformation: expect a referral
                case (True, str()):
                    pass

                case _:
                    assert_never(col_entry.transformation, col_entry._transformation_group)
            
               
    def _validate_presence_columns(self, attribute_name: str, stored_attribute: pd.DataFrame):
        for col in self.cols:
            if col not in stored_attribute.columns:
                raise MissingColumnError(attribute_name, col, self.dataclass_validated)        
        
    def _validate_nan(self, attribute_name: str, stored_attribute: pd.DataFrame):
        nan_columns = stored_attribute.columns[stored_attribute.isna().any()].tolist()
        if nan_columns:     
            NaNsFoundError(attribute_name, self.dataclass_validated, nan_columns)
        
    def _get_expected_attributes(self) -> List[str]:
        """returns a list of strings with expected attributes"""
        # these are mandatory
        expected_attributes = ['data']       

        return expected_attributes