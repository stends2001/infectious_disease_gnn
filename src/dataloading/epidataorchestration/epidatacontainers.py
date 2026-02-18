import pandas as pd 
import geopandas as gpd
from dataclasses import dataclass
from typing import Literal, Dict, Union, Optional, List
from typing import TYPE_CHECKING
from ...utils.textformatting import checkmark

if TYPE_CHECKING:
    from .epiconfig import EpiConfig
    from .temporal_summary import EpiDataTemporalSummary

@dataclass
class RawEpiData:

    disease:                pd.DataFrame
    population:             pd.DataFrame
    shapedata_node:         gpd.GeoDataFrame
    shapedata_collection:   Dict[str, gpd.GeoDataFrame]
    nuts_harm:              pd.DataFrame    
    population_berlin:      Optional[pd.DataFrame] = None

    population_density:     Optional[pd.DataFrame] = None
    gisd:                   Optional[pd.DataFrame] = None
    population_age:         Optional[pd.DataFrame] = None
    
    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(disease {checkmark}, "
                f"population {checkmark}, "
                f"shapedata_node {checkmark}, "
                f"shapedata_collection {checkmark}, "                
                f"nuts_harm {checkmark}")
        
        if self.population_berlin is not None:
            representation += f", population_berlin {checkmark}"
        
        if self.population_density is not None:
            representation += f", population_density {checkmark}"

        if self.gisd is not None:
            representation += f", gisd {checkmark}"

        if self.population_age is not None:
            representation += f", population_age {checkmark}"

        return representation +")>"

@dataclass 
class ContextData:
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
        representation = (f"<{self.__class__.__name__}(nuts_level = {self.nuts_level},"
                f"num_nodes={self.num_nodes}"
                f"shapedata_node {checkmark}, "
                f"shapedata_nuts0 {checkmark}, "
                f"shapedata_nuts1 {checkmark}, "
                f"shapedata_nuts2 {checkmark}, "
                f"shapedata_nuts3 {checkmark}, "                                                                
                f"nuts_harm {checkmark}, "
                f"tokenization_map {checkmark},"
                f"temporal_summary {checkmark}"            
                )
        representation += ")>"
        return representation

@dataclass
class HarmonizedData:
    epidata:     pd.DataFrame

    population_density: Optional[pd.DataFrame] = None
    gisd:               Optional[pd.DataFrame] = None    
    population_age:     Optional[pd.DataFrame] = None    

    def __repr__(self):
        representation = f"<{self.__class__.__name__}(epidata {checkmark}"

        if self.population_density is not None:
            representation += f", population_density {checkmark}, "

        if self.gisd is not None:
            representation += f", gisd {checkmark}"      

        if self.population_age is not None:
            representation += f", population_age {checkmark}"                  
                          
        representation += ")>"
        return representation

@dataclass
class ProcessedEpiData:
    epidata:     pd.DataFrame

    population_density: Optional[pd.DataFrame] = None
    gisd:               Optional[pd.DataFrame] = None      
    population_age:     Optional[pd.DataFrame] = None      
    
    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(epidata {checkmark}")

        if self.population_density is not None:
            representation += f", population_density {checkmark}, "

        if self.gisd is not None:
            representation += f", gisd {checkmark}"      

        if self.population_age is not None:
            representation += f", population_age {checkmark}"            

        representation += ")>"
        return representation

@dataclass
class FeatureEpiData:
    epidata:               pd.DataFrame

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(epidata {checkmark}")
        representation += ")>"
        return representation  

@dataclass
class NormalizedEpiData:
    epidata:     pd.DataFrame  

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(epidata {checkmark}")
        representation += ")>"
        return representation

@dataclass
class FinalizedEpiData:
    data:        pd.DataFrame
    data_denorm: pd.DataFrame

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(data {checkmark},"
                f"data_denorm {checkmark}"
                )
        representation += ")>"
        return representation   