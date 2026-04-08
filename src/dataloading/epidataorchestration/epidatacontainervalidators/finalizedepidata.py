from typing import List, assert_never, Union
import pandas as pd 

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pandas import Timestamp

from .base import EpiDataContainerValidator
from .issues import MissingColumnError, NaNsFoundError, IncorrectPeakTimeError
from ..epidatacontainers import RawEpiData, HarmonizedEpiData, FinalizedEpiData
from ...epiconfig import EpiConfig
from ...columnregistration.column_registry import ColumnRegistration

class FinalizedValidator(EpiDataContainerValidator):
    """ 

    """

    def __init__(self,
                 epiconfig:         'EpiConfig',
                 column_registry:   ColumnRegistration,
                 rawepidata:        'RawEpiData',
                 harmonizedepidata: 'HarmonizedEpiData',
                 finalizedepidata:  'FinalizedEpiData'):

        super().__init__(epiconfig, 
                         dataclass_validated='FinalizedEpiData')

        self.rawepidata         = rawepidata
        self.harmonizedepidata  = harmonizedepidata
        self.finalizedepidata   = finalizedepidata
        self.column_registry    = column_registry

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

        self._validate_peak_time(self.finalizedepidata.data_denorm)            
               
    def _validate_nan(self, attribute_name: str, stored_attribute: pd.DataFrame):
        nan_columns = stored_attribute.columns[stored_attribute.isna().any()].tolist()
        if nan_columns:     
            raise NaNsFoundError(attribute_name, self.dataclass_validated, nan_columns)

    def _validate_presence_columns(self, attribute_name: str, stored_attribute: pd.DataFrame):

        all_cols = self.column_registry.context_columns + self.column_registry.feature_columns+ self.column_registry.target_columns[1:]

        for col in all_cols:
            if col not in stored_attribute:
                raise MissingColumnError(attribute_name, col, self.dataclass_validated)        

    def _validate_peak_time(self, data_denorm: pd.DataFrame):

        final_data              = data_denorm.copy()
        target_column           = f'target_lead{self.epiconfig.horizon_leadtime}'
        final_peak_value        = final_data[target_column].max()

        final_peak_time         = final_data[final_data[target_column]== final_peak_value]['timestamp'].iloc[0]
        final_peak_node         = final_data[final_data[target_column]== final_peak_value][self.epiconfig.id_column].iloc[0]

        peak_key_code           = next(k for k, v in self.rawepidata.tokenization_map.items() if v == final_peak_node)
        computed_peak_time      = self._shift_date(final_peak_time, self.epiconfig.horizon_leadtime)

        if self.epiconfig.target_column == 'incidence':

            peak_population_size = self.harmonizedepidata.epidata[
                    (self.harmonizedepidata.epidata[self.epiconfig.id_column] == final_peak_node) &
                    (self.harmonizedepidata.epidata[self.epiconfig.temporal_column] == computed_peak_time)
                ]['population_size'].iloc[0]

            computed_casenumbers    = round(final_peak_value * peak_population_size / self.epiconfig.incidence_scalar)
        
        else:
            computed_casenumbers    = round(final_peak_value)
             
        true_casenumbers = self.harmonizedepidata.epidata[
                    (self.harmonizedepidata.epidata[self.epiconfig.id_column] == final_peak_node) &
                    (self.harmonizedepidata.epidata[self.epiconfig.temporal_column] == computed_peak_time)
                ]['cases'].iloc[0]
        
        if true_casenumbers == computed_casenumbers:
            print('correct peak time')

        else:
            
            specs = (
                f'Computed: node {final_peak_node} at timestamp {final_peak_time} has {computed_casenumbers} cases.\n'
                f'In reality at this point we have {true_casenumbers} cases.'
                     )

            raise IncorrectPeakTimeError(self.dataclass_validated, specs)

    def _get_expected_attributes(self) -> List[str]:
        """returns a list of strings with expected attributes"""
        # these are mandatory
        expected_attributes = ['data','data_denorm']       

        return expected_attributes
        
    def _shift_date(self, date: Union[datetime,Timestamp], steps: int) -> Union[datetime,Timestamp]:
        """Shift date by steps (positive=forward, negative=backward)"""

        match self.epiconfig.temporal_frequency:

            case 'd':
                return date + timedelta(days=steps)
            
            case 'w':
                return date + timedelta(weeks=steps)
            
            case 'm':
                return date + relativedelta(months=steps)

            case _:
                assert_never(self.epiconfig.temporal_frequency)