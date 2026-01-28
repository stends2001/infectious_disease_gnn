from dataclasses import dataclass
from typing import Tuple, Dict
import geopandas as gpd
import pandas as pd

from ...utils.textformatting import checkmark

@dataclass
class RawAirpData:
    flights:        pd.DataFrame
    worldharm:      pd.DataFrame
    worldshape:     gpd.GeoDataFrame
    airportharm:    gpd.GeoDataFrame
    popsize:        pd.DataFrame
    mv_cases:       pd.DataFrame
    mcv1:           pd.DataFrame  
    mcv2:           pd.DataFrame 

    
    def __repr__(self):
        representation = (f"<RawAirpData(flights {checkmark}, "
                f"worldharm {checkmark}, "
                f"worldshape {checkmark}, "                
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
    # World (HL1)
    world_harm:                         pd.DataFrame 
    world_num_nodes:                    int
    world_shapefile:                    pd.DataFrame
    world_tokenization_map:             Tuple[Dict[str,int], Dict[int,str]]
    
    # Airports (HL2)
    airport_harm:                       pd.DataFrame
    airport_num_nodes:                  int
    airport_shapefile:                  pd.DataFrame 
    airport_tokenization_map:           Tuple[Dict[str,int], Dict[int,str]]

    # Nuts (HL3)
    nuts_harm:                          pd.DataFrame
    nuts_num_nodes:                     int
    nuts_shapefile:                     pd.DataFrame
    nuts_tokenization_map:              Tuple[Dict[str,int], Dict[int,str]]

    def __post_init__(self):
        self._validate_num_nodes()
        

    def _validate_num_nodes(self):
        # HL1
        if not (
                self.world_num_nodes
                == len(self.world_harm)
                == len(self.world_shapefile)
                == len(self.world_tokenization_map[0])
                == len(self.world_tokenization_map[1])
                ):
            raise ValueError('Inconsistent number of nodes accross HL1: world in ContextAirpData')
        
        # HL2
        if not (
                self.airport_num_nodes
                == len(self.airport_harm)
                == len(self.airport_shapefile)
                == len(self.airport_tokenization_map[0])
                == len(self.airport_tokenization_map[1])
                ):
            raise ValueError('Inconsistent number of nodes accross HL2: airports in ContextAirpData')

        # HL3
        if not (
                self.nuts_num_nodes
                == len(self.nuts_harm)
                == len(self.nuts_shapefile)
                == len(self.nuts_tokenization_map[0])
                == len(self.nuts_tokenization_map[1])
                ):
            raise ValueError('Inconsistent number of nodes accross HL3: nuts in ContextAirpData')                

    def __repr__(self) -> str:
        return (f"<ContextAirpData(HL_harm, HL_num_nodes, HL_shapefile, HL_tokenization_map)> for HL in ['world','airport','nuts]")                
    
    def __str__(self) -> str:
        representation = ("<ContextAirpData("
                f"world_harm, world_num_nodes, world_shapefile, world_tokenization_map, "                       
                f"airport_harm, airport_num_nodes, airport_shapefile, airport_tokenization_map, "
                f"nuts_harm, nuts_num_nodes, nuts_shapefile, nuts_tokenization_map"                  
        )                
        return representation+")>" 

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