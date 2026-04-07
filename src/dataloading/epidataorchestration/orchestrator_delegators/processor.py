import time
from typing import TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from ...epiconfig import EpiConfig

from ....utils.textformatting import checkmark
from ..utils.temporal_summary import EpiDataTemporalSummary
from ..epidatacontainers.processedepidata import ProcessedEpiData
from ..epidatacontainers.harmonizedepidata import HarmonizedEpiData

class EpiDataProcessor:     
    """
    Processes the harmonzied data

    Parameters:
    -----------
    epiconfig: EpiConfig
    temporal_summary: EpiDataTemporalSummary

    Utility:
    -------
    the orchestrate method runs all required methods, based on EpiConfig and returns an
    instance of ProcessedEpiData    
    """
    def __init__(self, 
                 config:            'EpiConfig', 
                 temporal_summary:  EpiDataTemporalSummary):
        
        self.config             = config
        self.temporal_summary   = temporal_summary

    def _add_incidence_column(self, epipopdata: pd.DataFrame) -> pd.DataFrame:
        """adds incidence column"""
        epipopdata['incidence'] = epipopdata['cases'] / epipopdata['population_size'] * self.config.incidence_scalar     

        if self.config.verbose > 1:
            print(f'{checkmark} incidence column added')    

        return epipopdata
    
    def _drop_cases_column(self, epipopdata: pd.DataFrame) -> pd.DataFrame:
        """drops cases column -> only if target => incidence"""

        if self.config.verbose > 1:
            print(f'{checkmark} cases column removed')    

        return epipopdata.drop(columns=['cases']) 
       
    def _filter_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """using the timestamp column, filter on min/max date => using those determined in the temporal summary"""
        
        # Use extended min date from temporal summary
        dfc     = df.copy()
        mindate = self.temporal_summary.get_extended_dates()['min']
        maxdate = self.temporal_summary.get_extended_dates()['max']
             
        dfc = dfc.loc[dfc['timestamp'] <  maxdate].reset_index(drop=True)         
        dfc = dfc.loc[dfc['timestamp'] >= mindate].reset_index(drop=True)     

        if self.config.verbose > 1:
            print(f'{checkmark} filtered on dates')    

        return dfc
    
    def _filter_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """using the year column, filter on min/max date => using those determined in the temporal summary"""        
        dfc = df.copy() 
        minyear = self.temporal_summary.get_extended_dates()['min'].year

        # for max year its different in year => when max date '2021-02-01' 
        # we still need the data at '2021-01-01' so we can only include year < max_year+1
        maxyear = self.temporal_summary.get_extended_dates()['max'].year + 1

        dfc = dfc.loc[dfc['year'] <  maxyear].reset_index(drop=True)         
        dfc = dfc.loc[dfc['year'] >= minyear].reset_index(drop=True)             

        return dfc

    def orchestrate(self, harmonizeddata: 'HarmonizedEpiData') -> 'ProcessedEpiData':
        
        time_start = time.time()
        epidata = self._add_incidence_column(harmonizeddata.epidata.copy())
        epidata = epidata.drop(columns = 'population_size')

        if self.config.target_column == 'incidence':
            epidata = self._drop_cases_column(epidata)  
              
        epidata = self._filter_dates(epidata)

        # extra features; initiating them on None
        population_size         = None
        population_density_data = None    
        population_age          = None
        gisd_data               = None             
        kreise_classes          = None
        borders                 = None 
        vacmap                  = None

        if self.config.feature_popsize:
            population_size = self._filter_years(harmonizeddata.population_size)

        if self.config.feature_popdens:
            population_density_data = self._filter_years(harmonizeddata.population_density)
                   
        if self.config.feature_popage:
            population_age = self._filter_years(harmonizeddata.population_age)               

        if self.config.feature_gisd:
            gisd_data = self._filter_years(harmonizeddata.gisd)            

        if self.config.feature_kreise_classes:
            kreise_classes = harmonizeddata.kreise_classes             

        if self.config.feature_borders:
            borders        = harmonizeddata.borders
        
        processed_data = ProcessedEpiData(epidata = epidata,   
                                          _population_size   = population_size,
                                          _population_density= population_density_data,
                                          _population_age    = population_age,                                          
                                          _gisd              = gisd_data,
                                          _kreise_classes    = kreise_classes,
                                          _borders           = borders
                                          )
        time_end = time.time()
        if self.config.verbose > 2:
            print(f'Execution of EpiDataProcessor took {round(time_end - time_start,3)}s')    
        if self.config.verbose > 1:
            print("")

        return processed_data
