from dataclasses import dataclass
from typing import Tuple, Dict
import geopandas as gpd
import pandas as pd

from ...utils.textformatting import checkmark

@dataclass
class RawAirpData:
    flights:        pd.DataFrame
    worldharm:      pd.DataFrame
    airportharm:    gpd.GeoDataFrame
    popsize:        pd.DataFrame
    mv_cases:       pd.DataFrame
    mcv1:           pd.DataFrame  
    mcv2:           pd.DataFrame 

    
    def __repr__(self):
        representation = (f"<RawAirpData(flights {checkmark}, "
                f"worldharm {checkmark}, "
                f"airportharm {checkmark}, "                
                f"popsize {checkmark}, "
                f"mv_cases {checkmark}, "
                f"mcv1 {checkmark}, "
                f"mcv2 {checkmark}"
        )                

        return representation +")>"
    
@dataclass
class ProcessedAirpData:
    flightsdata:    pd.DataFrame
    epidata:        pd.DataFrame
    
    def __repr__(self):
        representation = (f"<ProcessedAirpData(flightsdata {checkmark}, "
                f"epidata {checkmark}"
        )                

        return representation +")>"    
    
@dataclass 
class ContextAirpData:
    world_harm:                         pd.DataFrame 
    airp_harm:                          pd.DataFrame 
    tokenization_map_airports:          Tuple[Dict[str,int], Dict[int,str]]          
    tokenization_map_countries:         Tuple[Dict[str,int], Dict[int,str]]
    num_airports:                       int

    def __repr__(self):
        representation = (f"<ContextAirpData(world_harm {checkmark}, "
                f"airp_harm {checkmark}"
                f"tokenization_map_airports {checkmark}"
                f"tokenization_map_countries {checkmark}"                                
        )                

        return representation +")>"     

@dataclass
class FeatureAirpData:
    data:                               pd.DataFrame

    def __repr__(self):
        representation = (f"<FeatureAirpData(data {checkmark}"                               
        )                

        return representation +")>"     
    
@dataclass
class NormalizedAirpData:
    data:                               pd.DataFrame
    normalization_parameters:           dict

    def __repr__(self):
        representation = (f"<NormalizedAirpData(data {checkmark}, "
                f"normalization_parameters {checkmark}"                          
        )                

        return representation +")>"         