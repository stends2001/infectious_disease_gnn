from dataclasses import dataclass
from typing import Literal
from pathlib import Path
import pandas as pd

from ..dataorchestration.epiconfig import EpiConfig
from ..dataorchestration.dataorchestrator import DataOrchestrator

from ...utils.textformatting import checkmark

class AirpConfigError(Exception):
    def __init__(self, explanation: str):
        statement = "AirpConfig couldn't be loaded" + "\n" + explanation
        super().__init__(statement)

@dataclass
class AirpConfig:
    data_orchestrator: DataOrchestrator

    ######################
    min_date: str = '2012-01-01'
    max_date: str = '2023-12-31'

    ######################
    incidence_scalar: int = 100_000

    ######################
    normalization_method: Literal['zscore','minmax']            = 'zscore'
    normalization_group:  Literal['collectively','individually']= 'individually'

    def __post_init__(self):
        self.epiconfig = self.data_orchestrator.config   

        self.min_date_ts  = pd.to_datetime(self.min_date)
        self.max_date_ts  = pd.to_datetime(self.max_date)

        self.recommended_min_date_ts =  pd.to_datetime('2012-01-01')
        self.recommended_max_date_ts =  pd.to_datetime('2023-12-31')

        self._validate_compatibility_epiconfig()
        self._validate_current_limitations()

        print(f'AirpConfig loaded succesfully {checkmark}')

    def _validate_compatibility_epiconfig(self):
        if self.epiconfig.disease != 'measles':
            raise AirpConfigError(f"AirpConfig is only compatible with measles data! Please adjust data-orchestrator's disease in EpiConfig.")
        
        if self.epiconfig.max_date > self.recommended_max_date_ts:
            raise AirpConfigError(f'Global population data is currently limited until {self.recommended_min_date_ts.date}, while EpiConfig data goes until {self.epiconfig.max_date}')
        
        if self.epiconfig.min_date < self.recommended_min_date_ts:
            raise AirpConfigError(f'Global casedata is currently limited from {self.recommended_min_date_ts.date}, while EpiConfig data goes from {self.epiconfig.min_date}')        

    def _validate_current_limitations(self):
        if self.min_date_ts < self.recommended_min_date_ts:
            raise AirpConfigError(f'Global casedata is currently limited from {self.recommended_min_date_ts.date}, while input requests {self.min_date}')   
        if self.max_date_ts > self.recommended_max_date_ts:            
            raise AirpConfigError(f'Global population data is currently limited until {self.recommended_max_date_ts.date}, while input requests {self.max_date}')            

    def get_flights_path(self) -> Path:
        """Path to flights CSV file."""
        return self.epiconfig.data_path / 'processed/international/mobility/travel/country_german_airp/selection2.csv'      
    
    def get_worldharm_path(self) -> Path:
        """Path to world harmonization TSV file."""
        return self.epiconfig.data_path / 'processed/international/geospatial/harmonization/world_harmonization.tsv'
    
    def get_global_cases_path(self) -> Path:
        """Path to who - cases CSV file."""
        return self.epiconfig.data_path / 'processed/international/epidemiology/casedata/who_mv_monthly.csv'    
    
    def get_global_mcv1_path(self) -> Path:
        """Path to mcv1 CSV file."""
        return self.epiconfig.data_path / 'processed/international/epidemiology/vaxdata/who_mcv1.csv'

    def get_global_mcv2_path(self) -> Path:
        """Path to mcv2 CSV file."""
        return self.epiconfig.data_path / 'processed/international/epidemiology/vaxdata/who_mcv2.csv'
        
    def get_global_popsize_path(self) -> Path:
        """Path to global popsize CSV file."""
        return self.epiconfig.data_path / 'processed/international/sociodemography/population_size.csv'
    
    def get_airports_shapedata(self) -> Path:
        """Path to shapefile of IATA airports SHP file."""
        return self.epiconfig.data_path / 'processed/international/geospatial/shapefiles/iata_airports.shp'       
    
    def __repr__(self) -> str:
        representation = ("<AirpConfig(epiconfig, min_date, max_date, incidence_scalar, normalization_method, normalization_group)>")

        return representation                            