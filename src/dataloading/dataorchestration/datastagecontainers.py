import pandas as pd 
import geopandas as gpd
from dataclasses import dataclass
from typing import Literal, Dict, Union, Optional

from ...utils.textformatting import checkmark

# ============= DATA CONTAINERS =============

@dataclass
class RawEpiData:
    """
    Container for raw data loaded from files. 

    Parameters:
    ----------
    disease: pd.DataFrame
        survstat data
    population: pd.Dataframe
        population size per kreise (TODO)
    shapedata: gpd.GeoDataFrame
        shape of specific nutslevel. for nuts3 there's the option to split berlin,
        explaining the 412 rows in that file (400 kreisen (incl. Berlin) + 12 Berlin districts)
    harmonization: pd.DataFrame
        pd.DataFrame of all nutscodes (nuts3 | nuts2 | nuts1)
    nuts_names: pd.DataFrame
        pd.DataFrame of all nutscodes and nutsnames (nuts3key | nuts3name | nuts2key | nuts2name | nuts1key | nuts1name)
    population_berlin: Optional[pd.DataFrame] = None
        optional population data for Berlin districts (used same proportions for each year) only when
        epiconfig.split_berlin = False
    """
    disease:        pd.DataFrame
    population:     pd.DataFrame
    shapedata:      gpd.GeoDataFrame
    harmonization:  pd.DataFrame
    nuts_names:     pd.DataFrame    
    population_berlin : Optional[pd.DataFrame] = None

    
    def __repr__(self):
        representation = (f"<RawEpiData(disease {checkmark}, "
                f"population {checkmark}, "
                f"shapedata {checkmark}, "
                f"harmonization {checkmark}, "
                f"nuts_names {checkmark}")
        
        if self.population_berlin is not None:
            representation += f", population_berlin {checkmark}"

        return representation +")>"

@dataclass
class ContextEpiData:
    """Container for (geographically) harmonized data."""
    nuts_level:     Literal['nuts1', 'nuts2', 'nuts3']
    num_nodes:      int
    epipopdata:     pd.DataFrame
    shapedata:      gpd.GeoDataFrame
    nuts_names:     pd.DataFrame
    tokenization_map: Dict[str, Dict[Union[int,str],Union[int,str]]]
    
    def __repr__(self):
        return (f"<ContextData(nuts_level={self.nuts_level}, num_nodes={self.num_nodes}, epipopdata, shapedata, nuts_names, tokenization_map)>")

@dataclass
class ProcessedEpiData:
    epipopdata:     pd.DataFrame
    
    def __repr__(self):
        return ('<ProcessedEpiData(epipopdata)>')        

@dataclass
class FeatureEpiData:
    data:               pd.DataFrame

    def __repr__(self):
        return ('<FeatureEpiData(data)>')   

@dataclass
class NormalizedEpiData:
    data:     pd.DataFrame  

    def __repr__(self):
        return ('<NormalizedEpiData(data)>')    

@dataclass
class FinalizedEpiData:
    data:        pd.DataFrame
    config:      dict 
    groundtruth: pd.DataFrame

    def __repr__(self):
        return (f"FinalizedEpiData(data, config, groudntruth)>")          