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

    disease:            pd.DataFrame
    population:         pd.DataFrame
    shapedata:          gpd.GeoDataFrame
    nuts_harm:          pd.DataFrame    
    population_berlin:  Optional[pd.DataFrame] = None
    
    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(disease {checkmark}, "
                f"population {checkmark}, "
                f"shapedata {checkmark}, "
                f"nuts_harm {checkmark}")
        
        if self.population_berlin is not None:
            representation += f", population_berlin {checkmark}"

        return representation +")>"

@dataclass 
class ContextData:
    nuts_level:         Literal['nuts1', 'nuts2', 'nuts3']
    shapedata:          gpd.GeoDataFrame
    nuts_harm:          pd.DataFrame
    tokenization_map:   Dict[str, Dict[(int | str), (int | str)]]
    temporal_summary:   'EpiDataTemporalSummary'
        
    @property 
    def num_nodes(self):
        return len(self.tokenization_map['nuts_node-idx'])
    
    @property     
    def num_timestamps(self):
        return len(self.tokenization_map['timestamp_idx'])        

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(nuts_level = {self.nuts_level},"
                f"num_nodes={self.num_nodes}"
                f"shapedata {checkmark}, "
                f"nuts_harm {checkmark}, "
                f"tokenization_map {checkmark},"
                f"temporal_summary {checkmark}"            
                )
        representation += ")>"
        return representation

@dataclass
class HarmonizedData:
    data:     pd.DataFrame

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(data {checkmark}")
        representation += ")>"
        return representation

@dataclass
class ProcessedEpiData:
    data:     pd.DataFrame
    
    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(data {checkmark}")
        representation += ")>"
        return representation

@dataclass
class FeatureEpiData:
    data:               pd.DataFrame

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(data {checkmark}")
        representation += ")>"
        return representation  

@dataclass
class NormalizedEpiData:
    data:     pd.DataFrame  

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(data {checkmark}")
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