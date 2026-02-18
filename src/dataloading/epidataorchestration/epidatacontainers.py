import pandas as pd 
import geopandas as gpd
from dataclasses import dataclass
from typing import Literal, Dict, Optional, TYPE_CHECKING

from .issues import DataOrchestrationError
from ...utils.textformatting import checkmark

if TYPE_CHECKING:
    from .temporal_summary import EpiDataTemporalSummary

@dataclass
class RawEpiData:
    """
    Datacontainer for raw dataframes. Just loaded, unchanged dfs.

    Parameters:
    ----------
    disease: pd.DataFrame
        casedata like survstat
    population: pd.DataFrame
        data on population_size per Kreis in Germany
    shapedata_node: gpd.GeoDataFrame
        shapedata for specified nuts level
    nuts_harm: pd.DataFrame
        lookup table in which each nuts3 is related to nuts2 and nuts1

    # optionally
    population_berlin: Optional[pd.DataFrame]
        population size in 2024 for the districts in Berlin.
        necessary when epiconfig.split_berlin == True
    population_density: Optional[pd.DataFrame]
        population density per node in Germany
        necessary when epiconfig.feature_popdens == True        
    gisd: Optional[pd.DataFrame]
        gisd data per node in Germany
        necessary when epiconfig.feature_gisd == True        
    population_age: Optional[pd.DataFrame]
        percentual population ages per node in Germany
        necessary when epiconfig.feature_popage == True
    """

    disease:                pd.DataFrame
    population:             pd.DataFrame
    shapedata_node:         gpd.GeoDataFrame
    shapedata_collection:   Dict[str, gpd.GeoDataFrame]
    nuts_harm:              pd.DataFrame    
    
    _population_berlin:      Optional[pd.DataFrame] = None
    _population_density:     Optional[pd.DataFrame] = None
    _gisd:                   Optional[pd.DataFrame] = None
    _population_age:         Optional[pd.DataFrame] = None
    
    @property
    def population_berlin(self) -> pd.DataFrame:
        df = self._population_berlin
        if df is None: 
            raise DataOrchestrationError(f'Attempted to access population_berlin at {self.__class__.__name__} but no such data found')
        return df

    @property
    def population_density(self) -> pd.DataFrame:
        df = self._population_density
        if df is None: 
            raise DataOrchestrationError(f'Attempted to access population_density at {self.__class__.__name__} but no such data found')
        return df   

    @property
    def gisd(self) -> pd.DataFrame:
        df = self._gisd
        if df is None: 
            raise DataOrchestrationError(f'Attempted to access gisd at {self.__class__.__name__} but no such data found')
        return df

    @property
    def population_age(self) -> pd.DataFrame:
        df = self._population_age
        if df is None: 
            raise DataOrchestrationError(f'Attempted to access population_age at {self.__class__.__name__} but no such data found')
        return df         

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(disease {checkmark}, "
                f"population {checkmark}, "
                f"shapedata_node {checkmark}, "
                f"shapedata_collection {checkmark}, "                
                f"nuts_harm {checkmark}")
        
        if self._population_berlin is not None:
            representation += f", population_berlin {checkmark}"
        
        if self._population_density is not None:
            representation += f", population_density {checkmark}"

        if self._gisd is not None:
            representation += f", gisd {checkmark}"

        if self._population_age is not None:
            representation += f", population_age {checkmark}"

        return representation +")>"

@dataclass 
class ContextData:
    """
    Datacontainer for context-data

    Parameters:
    ----------
    nuts_level: Literal['nuts1', 'nuts2', 'nuts3']
    shapedata_node: gpd.GeoDataFrame
        shapedata for specified nuts level: with tokenized id column
    shapedata_nuts0: gpd.GeoDataFrame
        shapedata for raw, unchanged German border (single row)
    shapedata_nuts1: gpd.GeoDataFrame
        shapedata for raw, unchanged German federal states    
    shapedata_nuts2: gpd.GeoDataFrame
        shapedata for raw, unchanged German Regierungsbezirke    
    shapedata_nuts3: gpd.GeoDataFrame  
        shapedata for raw, unchanged German Kreisen (including Berlin split and merged)   
    nuts_harm:  pd.DataFrame
        table as follows:
        ___________________________________________________
        | f"{nuts_level}_name" | f"{epiconfig.id_column}" |

    tokenization_map: Dict[str, Dict[(int | str), (int | str)]]
        the tokenization maps:
        - f"{epiconfig.id_column} - idx"    => Kennziffern: token_id
        - f"idx - {epiconfig.id_column}"    => token_id:    Kennziffern
    temporal_summary: EpiDataTemporalSummary
        temporal_summary created based on EpiConfig
    """    
    nuts_level:             Literal['nuts1', 'nuts2', 'nuts3']
    shapedata_node:         gpd.GeoDataFrame
    shapedata_nuts0:        gpd.GeoDataFrame
    shapedata_nuts1:        gpd.GeoDataFrame
    shapedata_nuts2:        gpd.GeoDataFrame
    shapedata_nuts3:        gpd.GeoDataFrame            
    nuts_harm:              pd.DataFrame
    tokenization_map:       Dict[str, Dict[(int | str), (int | str)]]
    temporal_summary:       'EpiDataTemporalSummary'
        
    @property 
    def num_nodes(self):
        return len(self.tokenization_map['nuts_node-idx'])    

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(nuts_level = {self.nuts_level}, "
                f"num_nodes = {self.num_nodes}, "
                f"shapedata_node {checkmark}, "
                f"shapedata_nuts0 {checkmark}, "
                f"shapedata_nuts1 {checkmark}, "
                f"shapedata_nuts2 {checkmark}, "
                f"shapedata_nuts3 {checkmark}, "                                                                
                f"nuts_harm {checkmark}, "
                f"tokenization_map {checkmark}, "
                f"temporal_summary {checkmark}"            
                )
        representation += ")>"
        return representation

@dataclass
class HarmonizedData:
    """
    Datacontainer for harmonized-data, both in space and time (that is to say: resampled when necessary)

    Parameters:
    ----------
    epidata: pd.DataFrame
        epidemiological data harmonized in nuts and time
        ___________________________________________________________________________________________________
        | f'{epiconfig.temporal_column}' | 'cases' | 'year' | 'population_size' | f'{epiconfig.id_column}' |
        
    # optional
    population_density: pd.DataFrame
    gisd: pd.DataFrame
    population_age: pd.DataFrame
    """        
    epidata:     pd.DataFrame

    _population_density: Optional[pd.DataFrame] = None
    _gisd:               Optional[pd.DataFrame] = None    
    _population_age:     Optional[pd.DataFrame] = None    

    @property
    def population_density(self) -> pd.DataFrame:
        df = self._population_density
        if df is None: 
            raise DataOrchestrationError(f'Attempted to access population_density at {self.__class__.__name__} but no such data found')
        return df         

    @property
    def gisd(self) -> pd.DataFrame:
        df = self._gisd
        if df is None: 
            raise DataOrchestrationError(f'Attempted to access gisd at {self.__class__.__name__} but no such data found')
        return df         

    @property
    def population_age(self) -> pd.DataFrame:
        df = self._population_age
        if df is None: 
            raise DataOrchestrationError(f'Attempted to access population_age at {self.__class__.__name__} but no such data found')
        return df                     

    def __repr__(self):
        representation = f"<{self.__class__.__name__}(epidata {checkmark}"

        if self._population_density is not None:
            representation += f", population_density {checkmark}"

        if self._gisd is not None:
            representation += f", gisd {checkmark}"      

        if self._population_age is not None:
            representation += f", population_age {checkmark}"                  
                          
        representation += ")>"
        return representation

@dataclass
class ProcessedEpiData:
    """
    Datacontainer for processed-data
    filtered on dates, added incidence column if necessary

    Parameters:
    ----------
    epidata: pd.DataFrame
        
    # optional
    population_size: pd.DataFrame
    population_density: pd.DataFrame
    gisd: pd.DataFrame
    population_age: pd.DataFrame
    """           
    epidata:     pd.DataFrame

    _population_size:    Optional[pd.DataFrame] = None
    _population_density: Optional[pd.DataFrame] = None
    _gisd:               Optional[pd.DataFrame] = None      
    _population_age:     Optional[pd.DataFrame] = None      
    
    @property
    def population_size(self) -> pd.DataFrame:
        df = self._population_size
        if df is None: 
            raise DataOrchestrationError(f'Attempted to access population_size at {self.__class__.__name__} but no such data found')
        return df      
    
    @property
    def population_density(self) -> pd.DataFrame:
        df = self._population_density
        if df is None: 
            raise DataOrchestrationError(f'Attempted to access population_density at {self.__class__.__name__} but no such data found')
        return df      
    
    @property
    def gisd(self) -> pd.DataFrame:
        df = self._gisd
        if df is None: 
            raise DataOrchestrationError(f'Attempted to access gisd at {self.__class__.__name__} but no such data found')
        return df      
    
    @property
    def population_age(self) -> pd.DataFrame:
        df = self._population_age
        if df is None: 
            raise DataOrchestrationError(f'Attempted to access population_age at {self.__class__.__name__} but no such data found')
        return df      

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(epidata {checkmark}")

        if self._population_size is not None:
            representation += f", population_size {checkmark}"

        if self._population_density is not None:
            representation += f", population_density {checkmark}"

        if self._gisd is not None:
            representation += f", gisd {checkmark}"      

        if self._population_age is not None:
            representation += f", population_age {checkmark}"            

        representation += ")>"
        return representation

@dataclass
class FeatureEpiData:
    """
    Datacontainer for feature-engineered-data, and shifted target

    Parameters:
    ----------
    epidata: pd.DataFrame
    """        
    epidata:               pd.DataFrame

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(epidata {checkmark}")
        representation += ")>"
        return representation  

@dataclass
class NormalizedEpiData:
    """
    Datacontainer for normalized data

    Parameters:
    ----------
    epidata: pd.DataFrame
    """     
    epidata:     pd.DataFrame  

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(epidata {checkmark}")
        representation += ")>"
        return representation

@dataclass
class FinalizedEpiData:
    """
    Datacontainer for finalized data, both normalized and denormalized

    Parameters:
    ----------
    epidata: pd.DataFrame
    """     
    data:        pd.DataFrame
    data_denorm: pd.DataFrame

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(data {checkmark},"
                f"data_denorm {checkmark}"
                )
        representation += ")>"
        return representation   