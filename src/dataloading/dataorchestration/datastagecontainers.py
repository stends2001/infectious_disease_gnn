import pandas as pd 
import geopandas as gpd
from dataclasses import dataclass
from typing import Literal, Dict, Union, Optional, List
from typing import TYPE_CHECKING
from ...utils.textformatting import checkmark

if TYPE_CHECKING:
    from .epiconfig import EpiConfig
    from .temporalsummary import EpiDataTemporalSummary

# ============= DATA CONTAINERS =============

@dataclass
class RawEpiData:
    """
    Container for raw data loaded from files. 

    Parameters
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
    nuts_names:     pd.DataFrame    
    population_berlin : Optional[pd.DataFrame] = None
    gisd:           Optional[pd.DataFrame] = None

    
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
class ContextData:
    """Container for context-data (not used in the pipeline)"""
    nuts_level:     Literal['nuts1', 'nuts2', 'nuts3']
    num_nodes:      int
    shapedata:      gpd.GeoDataFrame
    nuts_names:     pd.DataFrame
    tokenization_map: Dict[str, Dict[Union[int,str],Union[int,str]]]    
    temporal_summary: 'EpiDataTemporalSummary'
    
    def __repr__(self):
        return (f"<ContextData(nuts_level={self.nuts_level}, num_nodes={self.num_nodes}, shapedata, nuts_names, tokenization_map, temporal_summary)>")    

@dataclass
class HarmonizedData:
    """Container for nuts-harmonized data."""
    data:     pd.DataFrame
    gisd:     Optional[pd.DataFrame] = None

    def __repr__(self):
        repr_str = "<HarmonizedData(data"
        if self.gisd is not None:
            repr_str += ", gisd"
        return repr_str + ")>"

@dataclass
class ProcessedEpiData:
    data:     pd.DataFrame
    
    def __repr__(self):
        return ('<ProcessedEpiData(data)>')        

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
    data_denorm: pd.DataFrame
    config:      'EpiConfig' 
    groundtruth: pd.DataFrame

    def __repr__(self):
        return (f"FinalizedEpiData(data, data_denorm, config, groundtruth)>")          