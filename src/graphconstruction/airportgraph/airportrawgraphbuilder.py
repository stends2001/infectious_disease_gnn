import geopandas as gpd 
import pandas as pd
import numpy as np 
from typing import Literal, List
from dataclasses import dataclass, field

class GraphConnectionsDataFrameError(Exception):
    def __init__(self, explanation: str):
        statement = "GraphConnectionsDataFrame couldnt be initialized" + "\n" + explanation
        super().__init__(statement)

@dataclass
class GraphConnectionsDataFrame:
    """ 
    each AirportRawGraphCreator returns an instance of this class
    """
    df: pd.DataFrame
    required_columns: List[str] = field(default_factory=lambda: ['node_layer2', 'node_layer3', 'weight'])

    def __post_init__(self):
        self._validate_columns()
        self._validate_positive_raw_weights()
        self._validate_dtypes()
        self._validate_node_combinations()

    def _validate_dtypes(self):
        if not pd.api.types.is_integer_dtype(self.df['node_layer2']):
            raise GraphConnectionsDataFrameError("node_layer2 must be integers")
        if not pd.api.types.is_integer_dtype(self.df['node_layer3']):
            raise GraphConnectionsDataFrameError("node_layer3 must be integers")
        if not pd.api.types.is_numeric_dtype(self.df['weight']):
            raise GraphConnectionsDataFrameError("weight must be numeric")

    def _validate_node_combinations(self):
        node2_unique = self.df['node_layer2'].unique()
        node3_unique = self.df['node_layer3'].unique()
        all_combinations = pd.MultiIndex.from_product([node2_unique, node3_unique])
        df_combinations  = pd.MultiIndex.from_frame(self.df[['node_layer2', 'node_layer3']])
        missing          = all_combinations.difference(df_combinations)
        if not missing.empty:
            raise GraphConnectionsDataFrameError(f"Missing node combinations: {list(missing)}")        

    def _validate_columns(self):
        missing = [col for col in self.required_columns if col not in self.df.columns]
        if missing:
            raise GraphConnectionsDataFrameError(f"Missing required columns: {missing}")        

    def _validate_positive_raw_weights(self):
        if (self.df['weight'] < 0).any():
            raise GraphConnectionsDataFrameError("Graph contains non-positive weights")
    
    @property
    def df_copy(self) -> pd.DataFrame:
        return self.df.copy()
  
class AirportRawGraphBuilder():
    """
    each method returns an instance of GraphConnectionsDataFrame
    which looks like:
    ______________________________________
    | node_layer2 | node_layer3 | weight |
    ______________________________________

    these weight-values are not normalized!
    """

    def __init__(self, nuts_shapefile: gpd.GeoDataFrame, airport_shapefile: gpd.GeoDataFrame):

        self.expected_input_crs     = 'epsg:4326' 
        self.germany_crs            = 25832      

        # epsg 4326 is in degrees. Need to convert it do distance based
        self.projected_nuts_shapefile    = nuts_shapefile.to_crs(self.germany_crs)
        self.projected_airport_shapefile = airport_shapefile.to_crs(self.germany_crs)        

    def distance(self, radius_km: float = 100.0, decay: Literal['boolean','linear','exponential'] = 'boolean') -> 'GraphConnectionsDataFrame':
        # 1. Make a Cartesian join between airports and nuts
        layer2 = self.projected_airport_shapefile[['node_layer2', 'geometry']].copy()
        layer3 = self.projected_nuts_shapefile[['node_layer3', 'geometry']].copy()
        
        # Add a temporary key for cross join
        layer2['_tmp']      = 1
        layer3['_tmp']      = 1
        df_pairs            = layer2.merge(layer3, on='_tmp', suffixes=('_airport', '_nuts'))
        df_pairs.drop(columns=['_tmp'], inplace=True)

        # 2. Compute distances (in meters, then convert to km)
        df_pairs['distance_km'] = df_pairs.apply(
            lambda row: row['geometry_airport'].distance(row['geometry_nuts']) / 1000, axis=1)

        # 3. Assign weight based on radius
        #   1. boolean
        if decay == 'boolean':
            df_pairs['weight'] = (df_pairs['distance_km'] <= radius_km).astype(int)
        
        #   2. linear
        elif decay == 'linear':
            # Weight decreases linearly from 1 at distance=0 to 0 at distance=radius_km
            df_pairs['weight'] = np.clip(1 - df_pairs['distance_km'] / radius_km, 0, 1)
        
        #   3. exponential
        elif decay == 'exponential':
            # Weight decreases exponentially: exp(-distance / radius)
            df_pairs['weight'] = np.exp(-df_pairs['distance_km'] / radius_km)

        else:
            raise ValueError(f"Unsupported decay method: {decay}")

        # 4. Keep only necessary columns
        df_result = df_pairs[['node_layer2', 'node_layer3', 'weight']]

        return GraphConnectionsDataFrame(df_result)
   