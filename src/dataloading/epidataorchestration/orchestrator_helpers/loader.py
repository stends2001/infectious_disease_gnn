import time
import json
from typing import assert_never, TYPE_CHECKING, Dict
import pandas as pd
import geopandas as gpd
from ....utils.textformatting import checkmark

if TYPE_CHECKING:
    from ...epiconfig import EpiConfig

from ..containers import RawEpiData

# ============= DATA IMPORTATION CLASS =============
class EpiDataReader:
    """
    """
    
    def __init__(self, epiconfig: EpiConfig):
        self.epiconfig = epiconfig

    def orchestrate(self) -> RawEpiData:
        time_start = time.time()
        
        rawdata = RawEpiData(
            disease             = self._load_disease_data(),
            population_size     = self._load_population_size_data(),
            shapedata           = self._load_shapedata(),
            region_harmonization= self._load_regional_harm(),
            tokenization_map    = self._load_tokenization_map(),
        
            # optional data
            _population_density  = self._load_population_density()               if self.epiconfig.feature_popdens          else None,
            _population_age      = self._load_population_age()                   if self.epiconfig.feature_popage           else None,
            _kreise_classes      = self._load_kreise_classes()                   if self.epiconfig.feature_kreise_classes   else None,
            _gisd                = self._load_gisd()                             if self.epiconfig.feature_gisd             else None,
            _borders             = self._load_borders_data()                     if self.epiconfig.feature_borders          else None               
        )

        time_end = time.time()
        if self.epiconfig.verbose > 2:
            print(f'Execution of EpiDataReader took {round(time_end - time_start,3)}s')

        if self.epiconfig.verbose > 1:
            print("")
        return rawdata
  
    # ======= MANDATORY DATA ======= #

    def _load_disease_data(self) -> pd.DataFrame:
        """
        loads disease data cleaned from survstat

        German df looks like:
        __________________________________________________________
        | 'week' | 'nuts3_key' | 'cases' | 'year ' | 'timestamp' |

        Dutch df looks like:
        _______________________________________
        | 'timestamp' | 'cases' | 'lau_key ' | 
        """        
        filepath    = self.epiconfig.path_manager.get('cases')

        match self.epiconfig.country:
            case 'germany':
                initial_key  = 'kz_kreis'
                renamed_key  = 'nuts3_key' 
        
            case 'netherlands':
                initial_key  = 'lau_key'
                renamed_key  = 'lau_key'
            
            case _:
                assert_never(self.epiconfig.country)

                
        df = pd.read_csv(
            filepath,
            parse_dates = ['timestamp'],
            dtype       = {initial_key:  str, 
                           'cases':      int}
        ).rename(columns={initial_key: renamed_key})
        
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded raw disease data")
        
        return df
    
    def _load_population_size_data(self) -> pd.DataFrame:
        """
        loads population data

        df looks like:
        _______________________________________________
        | 'level' | 'key' | 'year' | 'population_size |       
        """         
        filepath = self.epiconfig.path_manager.get('population_size')
        
        df = pd.read_csv(
            filepath,
            dtype = {'key' : str}
        )
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded raw population data")
        
        return df

    def _load_shapedata(self) -> gpd.GeoDataFrame:
        """
        loads shapedata for the specified nuts level

        gdf looks like:
        __________________________________________
        | 'level' | 'key' | 'geometry' |

        """          
        filepath = self.epiconfig.path_manager.get('shapefile')
        
        gdf             = gpd.read_file(filepath)
        gdf['key']      = gdf['key'].astype(str)
        
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded raw shapedata ({len(gdf)} regions)")
        
        return gdf
      
    def _load_regional_harm(self) -> pd.DataFrame:
        """
        loads harmonization data for nuts divisions in Germany

        df looks like for Germany:
        _________________________________________________________________________________________
        | 'nuts3_key' | 'nuts2_key' | 'nuts1_key' | 'nuts3_name' | 'nuts2_name' | 'nuts1_name' |

        df looks like this for NL:
        ______________________________________________________________________________________________________________
        | 'lau_key' | 'ggd_key' | 'nuts2_key' | 'nuts1_key' | 'lau_name' | 'ggd_name' | 'nuts2_name' | 'nuts1_name' |              
        """  

        filepath = self.epiconfig.path_manager.get('region_harmonization')   
        
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded raw NUTS names")
        
        return df

    def _load_tokenization_map(self) -> Dict[str, int]:
        
        filepath = self.epiconfig.path_manager.get('tokenization_map')       

        with open(filepath, "r") as f:
            tokenization_map = json.load(f)

        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded tokenization map")         

        return tokenization_map   

    # ======= OPTIONAL DATA =======#
  
    # optionally relevant for NL and Germany
    def _load_population_density(self) -> pd.DataFrame:
        """
        loads population density data for the specified nuts level

        df looks like:
        __________________________________________________________________
        | 'level' | 'key' | 'year' | 'population_density' |

        """          
        filepath = self.epiconfig.path_manager.get('population_density')
        
        df = pd.read_csv(
            filepath,
            dtype   = {'key': str},
        )
        
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded population density")
        
        return df

    # only for Germany
    def _load_population_age(self) -> pd.DataFrame:
        """
        loads population density data for the specified nuts level

        df looks like:
        _______________________________________________________________________________
        | 'level' | 'key' | '{age_group0}' | ... | '{age_group16}' | 'year' |

        where age_groups have certain names, namely: 'under_3_years' for example.
        """          
        filepath = self.epiconfig.path_manager.get('population_age')
        
        df = pd.read_csv(
            filepath,
            dtype   = {'key': str},
        )
        
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded population age")
        
        return df        
    
    # only for Germany
    def _load_gisd(self) -> pd.DataFrame:
        """
        loads population density data for the specified nuts level

        df looks like:
        __________________________________________________________
        | 'level' | 'key' | 'gisd_score' | 'year' |

        NOTE
        ----
        GISD data is only available for nuts2/3 of Germany, and with a select timeframe.
        """          
        filepath = self.epiconfig.path_manager.get('gisd')
        
        df = pd.read_csv(
            filepath,
            dtype   = {'key': str}
        )
        
                
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded gisd")
        
        return df
    
    # only for Germany
    def _load_kreise_classes(self) -> pd.DataFrame:
        """
      ['nuts_level', 'nuts_key', 'settlement_location_central',
       'settlement_location_peripheral', 'settlement_location_very_central',
       'settlement_location_very_peripheral',
       'settlement_type_large_independent_city',
       'settlement_type_small_independent_city',
       'settlement_type_sparsely_populated_rural', 'settlement_type_urban',
       'settlement_type_urbanizing_rural', 'east', 'west', 'kreis_type_kreis',
       'kreis_type_kreisfreie_stadt', 'kreis_type_landkreis',
       'kreis_type_regionalverband', 'kreis_type_stadtkreis']
        """          
        filepath = self.epiconfig.path_manager.get('kreise_classes')
        
        df = pd.read_csv(
            filepath,
            dtype= {'key' : str}
            )
                
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded kreise classes")
        
        return df
    
    # only for Germany
    def _load_borders_data(self) -> pd.DataFrame:
        """
        ['nuts_level', 'nuts_key', 'Austria', 'Belgium', 'Czech', 'Denmark',
       'France', 'Luxembourg', 'Netherlands', 'Poland', 'Switzerland', 'none']
        """          
        filepath = self.epiconfig.path_manager.get('border_regions')
        
        df = pd.read_csv(
            filepath,
            dtype   = {'key': str},
            )
                
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded border data")
        
        return df
