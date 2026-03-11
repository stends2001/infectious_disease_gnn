import pandas as pd 
import geopandas as gpd
from dataclasses import dataclass
from typing import Literal, Dict, Optional, TYPE_CHECKING

from .issues import EpiDataOrchestrationError, NonExistentAttributeEpiDataContainer
from ...utils.textformatting import checkmark

if TYPE_CHECKING:
    from src.dataloading.epidataorchestration.temporal_summary import EpiDataTemporalSummary

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
    population_size:        pd.DataFrame
    shapedata:              gpd.GeoDataFrame
    nuts_harm:              pd.DataFrame    
    
    _population_density:     Optional[pd.DataFrame] = None
    _population_age:         Optional[pd.DataFrame] = None
    _gisd:                   Optional[pd.DataFrame] = None    
    _kreise_classes:         Optional[pd.DataFrame] = None 
    _borders:                Optional[pd.DataFrame] = None 
    _vacmap:                 Optional[pd.DataFrame] = None
    
    @property
    def population_density(self) -> pd.DataFrame:
        df = self._population_density
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'population_density')
        return df   

    @property
    def population_age(self) -> pd.DataFrame:
        df = self._population_age
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'population_age')
        return df           

    @property
    def gisd(self) -> pd.DataFrame:
        df = self._gisd
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'gisd')
        return df

    @property
    def kreise_classes(self) -> pd.DataFrame:
        df = self._kreise_classes
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'kreise_classes')
        return df    
         
    @property
    def borders(self) -> pd.DataFrame:
        df = self._borders
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'borders')
        return df    

    @property
    def vacmap(self) -> pd.DataFrame:
        df = self._vacmap
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'vacmap')
        return df          

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(disease {checkmark}, "
                f"population_size {checkmark}, "
                f"shapedata {checkmark}, "               
                f"nuts_harm {checkmark}")
        
        if self._population_density is not None:
            representation += f", population_berlin {checkmark}"
        
        if self._population_age is not None:
            representation += f", population_age {checkmark}"

        if self._gisd is not None:
            representation += f", gisd {checkmark}"

        if self._kreise_classes is not None:
            representation += f", kreise_classes {checkmark}"

        if self._borders is not None:
            representation += f", borders {checkmark}"            

        if self._vacmap is not None:
            representation += f", vacmap {checkmark}"                  

        return representation +")>"

@dataclass 
class ContextEpiData:
    """
    Datacontainer for context-epidata

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
    population_size: pd.DataFrame        
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
    shapedata:              gpd.GeoDataFrame   
    nuts_shapedata:         gpd.GeoDataFrame     
    population_size:        pd.DataFrame
    nuts_harm:              pd.DataFrame
    tokenization_map:       Dict[str, Dict[(int | str), (int | str)]]
    temporal_summary:       'EpiDataTemporalSummary'
        
    @property 
    def num_nodes(self):
        return len(self.tokenization_map['nuts_node-idx'])    

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(nuts_level = {self.nuts_level}, "
                f"num_nodes = {self.num_nodes}, "
                f"shapedata {checkmark}, "             
                f"nuts_shapedata {checkmark}, "                  
                f"population_size {checkmark}, "                                                                       
                f"nuts_harm {checkmark}, "
                f"tokenization_map {checkmark}, "
                f"temporal_summary {checkmark}"            
                )
        representation += ")>"
        return representation

@dataclass
class HarmonizedEpiData:
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

    _population_size:    Optional[pd.DataFrame] = None
    _population_density: Optional[pd.DataFrame] = None 
    _population_age:     Optional[pd.DataFrame] = None    
    _gisd:               Optional[pd.DataFrame] = None  
    _kreise_classes:     Optional[pd.DataFrame] = None
    _borders:            Optional[pd.DataFrame] = None
    _vacmap:             Optional[pd.DataFrame] = None

    @property
    def population_size(self) -> pd.DataFrame:
        df = self._population_size
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'population_size')
        return df 

    @property
    def population_density(self) -> pd.DataFrame:
        df = self._population_density
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'population_density')
        return df         

    @property
    def population_age(self) -> pd.DataFrame:
        df = self._population_age
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'population_age')
        return df             

    @property
    def gisd(self) -> pd.DataFrame:
        df = self._gisd
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'gisd')
        return df         

    @property
    def kreise_classes(self) -> pd.DataFrame:
        df = self._kreise_classes
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'kreise_classes')
        return df     

    @property
    def borders(self) -> pd.DataFrame:
        df = self._borders
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'borders')
        return df   

    @property
    def vacmap(self) -> pd.DataFrame:
        df = self._vacmap
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'vacmap')
        return df                        

    def __repr__(self):
        representation = f"<{self.__class__.__name__}(epidata {checkmark}"

        if self._population_size is not None:
            representation += f", _population_size {checkmark}"

        if self._population_density is not None:
            representation += f", population_density {checkmark}"
        
        if self._population_age is not None:
            representation += f", population_age {checkmark}"

        if self._gisd is not None:
            representation += f", gisd {checkmark}"

        if self._kreise_classes is not None:
            representation += f", kreise_classes {checkmark}"

        if self._borders is not None:
            representation += f", borders {checkmark}"            

        if self._vacmap is not None:
            representation += f", vacmap {checkmark}"    

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
    _population_age:     Optional[pd.DataFrame] = None         
    _gisd:               Optional[pd.DataFrame] = None       
    _kreise_classes:     Optional[pd.DataFrame] = None
    _borders:            Optional[pd.DataFrame] = None
    _vacmap:             Optional[pd.DataFrame] = None    
    
    @property
    def population_size(self) -> pd.DataFrame:
        df = self._population_size
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'population_size')
        return df      
    
    @property
    def population_density(self) -> pd.DataFrame:
        df = self._population_density
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'population_density')
        return df      
        
    @property
    def population_age(self) -> pd.DataFrame:
        df = self._population_age
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'population_age')
        return df 
    
    @property
    def gisd(self) -> pd.DataFrame:
        df = self._gisd
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'gisd')
        return df           
    
    @property
    def kreise_classes(self) -> pd.DataFrame:
        df = self._kreise_classes
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'kreise_classes')
        return df       

    @property
    def borders(self) -> pd.DataFrame:
        df = self._borders
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'borders')
        return df   

    @property
    def vacmap(self) -> pd.DataFrame:
        df = self._vacmap
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'vacmap')
        return df          

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(epidata {checkmark}")

        if self._population_size is not None:
            representation += f", population_size {checkmark}"

        if self._population_density is not None:
            representation += f", population_density {checkmark}"

        if self._population_age is not None:
            representation += f", population_age {checkmark}"      

        if self._gisd is not None:
            representation += f", gisd {checkmark}"      

        if self._kreise_classes is not None:
            representation += f", kreise_classes {checkmark}"       

        if self._borders is not None:
            representation += f", borders {checkmark}"       

        if self._vacmap is not None:
            representation += f", vacmap {checkmark}"       


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
    data:               pd.DataFrame

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(data {checkmark}")
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
    data:     pd.DataFrame  

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(data {checkmark}")
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