import geopandas as gpd 
import pandas as pd
from typing import Dict, Tuple
from pathlib import Path

from ...utils import compare_sets
from ..exceptions import MissingColumnError


class GraphContextDataProcessor:
    """ 
    TODO:
        popsize filter on 2020 -> dynamic

    call `process()`    
    """
    def __init__(self,
                 level:             str,
                 id_col:            str,
                 token_col:         str,
                 country_path:      Path
                 ):
        
        self.id_col         = id_col 
        self.token_col      = token_col  
        self.level          = level  
        self.country_path   = country_path

        self.shp_raw, self.pop_raw = self._load()
        self._validate()

    def process(self) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, Dict[str, int]]:
        shp_f, pop_f = self._filter(self.shp_raw, self.pop_raw)
        shp_t, pop_t, map_t = self._tokenize(shp_f, pop_f)

        return shp_t, pop_t, map_t

    def _load(self) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
        shp = gpd.read_file(self.country_path / 'geospatial' / 'level_shapes.shp')
        pop = pd.read_csv(self.country_path  / 'sociodemography' / 'population_size.csv', dtype = {'key': 'str'})        
        return shp, pop

    def _validate(self):
        """internally called validation - method. Checks for id-columns' presence among the datasets."""
        if self.id_col not in self.shp_raw.columns:
            raise MissingColumnError(self.id_col, 'shape data')
        
        if self.id_col not in self.pop_raw.columns:
            raise MissingColumnError(self.id_col, 'population data')

    def _filter(self,
                 shape_data: gpd.GeoDataFrame, 
                 population_data: pd.DataFrame)  -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """filters data on level and year"""
        shp_f = shape_data[shape_data['level'] == self.level].reset_index(drop=True)
        pop_f = population_data[population_data['level'] == self.level].reset_index(drop=True)
        pop_f = pop_f[pop_f['year'] == 2020].reset_index(drop = True)
            
        return shp_f, pop_f

    def _tokenize(self, 
                 shape_data: gpd.GeoDataFrame, 
                 population_data: pd.DataFrame) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, Dict[str, int]]:
        """sets a tokenization_map and tokenizes all DataFrames"""
        
        shape_data              = shape_data.copy()
        population_data         = population_data.copy()    

        unique_shape_data_ids   = set(shape_data[self.id_col])
        tokenization_map        = {str(key): value for value, key in enumerate(sorted(unique_shape_data_ids))}        

        if population_data is not None:
            unique_population_data_ids = set(population_data[self.id_col])

            compare_sets(unique_shape_data_ids,unique_population_data_ids)

            population_data[self.token_col] = (
                population_data[self.id_col]
                .map(tokenization_map)
                .astype(int)
            )

            if not population_data[self.token_col].notna().all():
                raise ValueError(f'Unmapped keys found in population data:  {population_data[self.id_col][population_data[self.token_col].isna()].tolist()}')

            population_data.drop(columns = self.id_col, inplace = True)

        shape_data[self.token_col] = (
            shape_data[self.id_col]
                .map(tokenization_map)
                .astype(int)
        )    

        if not shape_data[self.token_col].notna().all():
            raise ValueError(f'Unmapped keys found in shape data:  {shape_data[self.id_col][shape_data[self.token_col].isna()].tolist()}')            

        shape_data.drop(columns = self.id_col, inplace = True)            

        return shape_data, population_data, tokenization_map