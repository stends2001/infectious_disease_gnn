import pandas as pd 
import geopandas as gpd
from dataclasses import dataclass
from typing import Literal, Dict

from ..utils.temporal_summary import EpiDataTemporalSummary
from ....utils.textformatting import checkmark


@dataclass 
class ContextEpiData:
    """

    """    
    country:                Literal['germany','netherlands']
    level:                  Literal['nuts1', 'nuts2', 'nuts3','ggd','lau']
    global_shapedata:       gpd.GeoDataFrame   
    local_shapedata:        gpd.GeoDataFrame     
    population_size:        pd.DataFrame
    nodenames:              pd.DataFrame
    region_harmonization:   pd.DataFrame
    tokenization_map:       Dict[str, int]
    temporal_summary:       'EpiDataTemporalSummary'

    @property
    def num_nodes(self) -> int:
        return len(self.local_shapedata)

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(country = {self.country}, "
                f"level = {self.level}, "
                f"global_shapedata {checkmark}, "             
                f"local_shapedata {checkmark}, "                  
                f"population_size {checkmark}, "                   
                f"region_harmonization {checkmark}, "                                                                      
                f"nodenames {checkmark}, "
                f"tokenization_map {checkmark}, "
                f"temporal_summary {checkmark}"            
                )
        representation += ")>"
        return representation