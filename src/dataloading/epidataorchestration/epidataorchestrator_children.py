import pandas as pd
import geopandas as gpd
from typing import Tuple, Dict, Literal, TYPE_CHECKING
import numpy as np
import time

from ...utils.constants import berlin_district_ids, berlin_id
from ...utils.textformatting import checkmark

from .temporal_summary import EpiDataTemporalSummary
from .column_registry import ColumnRegistration
from .epidatacontainers import RawEpiData, HarmonizedEpiData, ContextEpiData, ProcessedEpiData, FeatureEpiData, NormalizedEpiData, FinalizedEpiData
from .normalization import apply_minmax_scaling, apply_zscore_scaling, pipeline_minmax_normalization, pipeline_zscore_normalization
from .normalization import reverse_log, reverse_minmax_scaling, reverse_zscore_scaling
from .issues import EpiDataOrchestrationError, MissingEpiDataContainer

if TYPE_CHECKING:
    from .epiconfig import EpiConfig

# ============= DATA IMPORTATION CLASS =============
class EpiDataReader:
    """
    Class to load all required dataframes

    Parameters:
    ----------
    config: EpiConfig

    Utility:
    -------
    the orchestrate method runs all required methods, based on EpiConfig and returns an
    instance of RawEpiData
    """
    
    def __init__(self, epiconfig: 'EpiConfig'):
        self.epiconfig = epiconfig
    
    def _load_disease_data(self) -> pd.DataFrame:
        """
        loads disease data cleaned from survstat

        df looks like:
        __________________________________________________________
        | 'week' | 'nuts3_key' | 'cases' | 'year ' | 'timestamp' |
        """        
        filepath = self.epiconfig.get_disease_path()
                
        df = pd.read_csv(
            filepath,
            parse_dates = ['timestamp'],
            dtype       = {'kz_kreis':  str, 
                           'cases':     int}
        ).rename(columns={'kz_kreis': 'nuts3_key'})
        
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded raw disease data")
        
        return df
    
    def _load_population_data(self) -> pd.DataFrame:
        """
        loads population data

        df looks like:
        _____________________________________________
        | 'nuts3_key' | 'year ' | 'population_size' |
        """         
        filepath = self.epiconfig.get_population_path()
        
        df = pd.read_csv(
            filepath,
            dtype   = {'nuts3': str}
        ).rename(columns = {'nuts3':'nuts3_key'})
        
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded raw population data")
        
        return df
    
    def _load_population_data_berlin_districts(self) -> pd.DataFrame:
        """
        loads population data (I created myself) for districts in Berlin

        df looks like:
        ___________________________________
        | 'nuts3_key' | 'population_size' |
        """          
        filepath = self.epiconfig.get_population_berlin_districts_path()
        
        df = pd.read_csv(
            filepath,
            dtype={'nuts3': str}
            ).rename(columns = {'nuts3':'nuts3_key'})
        
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded raw berlin districts population data")
        
        return df        

    def _load_node_shapedata(self) -> gpd.GeoDataFrame:
        """
        loads shapedata for the specified nuts level

        gdf looks like:
        _____________________________________________________
        | 'nuts{int}_key' | 'nuts{int}_name' | 'geometry ' |

        where 'nuts{int}' is variable and depends on the input in epiconfig.
        """          
        filepath = self.epiconfig.get_nuts_shapefile_path()
        
        gdf = gpd.read_file(filepath).drop(columns=['level'], errors='ignore')
        
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded raw shapedata ({len(gdf)} regions)")
        
        return gdf
    
    def _load_shapedata_collection(self) -> Dict[str, gpd.GeoDataFrame]:
        """
        Loads all background shapedata: shapedata for all nuts levels, that won't be changed depending on nuts resolution in config
        Returned in a dictionary with keys ['nuts0' - 'nuts3'] and the gdfs as values
        """
        filepaths = self.epiconfig.get_shapefile_paths()

        shapefiles= {key:  gpd.read_file(filepaths[key]).drop(columns=['level'], errors='ignore') for key in list(filepaths.keys())}

        return shapefiles
        
    def _load_nuts_harm(self) -> pd.DataFrame:
        """
        loads harmonization data for nuts divisions in Germany

        df looks like:
        _________________________________________________________________________________________
        | 'nuts3_key' | 'nuts2_key' | 'nuts1_key' | 'nuts3_name' | 'nuts2_name' | 'nuts1_name' |
        """  
        # main file, additions-file
        filepath = self.epiconfig.get_nuts_harmonization_path()   
        
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded raw NUTS names")
        
        return df

    def _load_population_density(self) -> pd.DataFrame:
        """
        loads population density data for the specified nuts level

        df looks like:
        _____________________________________________________
        | '{nuts_level}_key' | 'year' | 'population_density ' |

        where '{nuts_level}' is self.nuts_level in epiconfig.
        """          
        filepath = self.epiconfig.get_population_density_path()
        
        df = pd.read_csv(
            filepath,
            dtype={f'{self.epiconfig.nuts_level}_key': str}
            )
        
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded population density")
        
        return df

    def _load_population_age(self) -> pd.DataFrame:
        """
        loads population density data for the specified nuts level

        df looks like:
        _____________________________________________________________________________________________
        | '{nuts_level}_key' | 'year' | 'age_group0' | ... | 'age_group16' | 'population_size' |

        where '{nuts_level}' is self.nuts_level in epiconfig.
        """          
        filepath = self.epiconfig.get_population_age_path()
        
        df = pd.read_csv(
            filepath,
            dtype={f'{self.epiconfig.nuts_level}': str},
            parse_dates=['timestamp']
            )
        
        df = df.rename(columns = {self.epiconfig.nuts_level : f'{self.epiconfig.nuts_level}_key'})
        timestamp: pd.Series[pd.Timestamp]  = df['timestamp']    
        df['year'] = timestamp.dt.year
        df.drop(columns = 'timestamp', inplace = True)

        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded population age")
        
        return df        

    def _load_gisd(self) -> pd.DataFrame:
        """
        loads population density data for the specified nuts level

        df looks like:
        _____________________________________________________
        | '{nuts_level}_key' | 'year' | 'gisd ' |

        where '{nuts_level}' is self.nuts_level in epiconfig.
        """          
        filepath = self.epiconfig.get_gisd_path()
        
        df = pd.read_csv(
            filepath,
            dtype={f'{self.epiconfig.nuts_level}_key': str}
            )
                
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} Loaded gisd")
        
        return df

    def orchestrate(self) -> RawEpiData:
        time_start = time.time()

        rawdata = RawEpiData(
            disease             = self._load_disease_data(),
            population          = self._load_population_data(),
            shapedata_node      = self._load_node_shapedata(),
            shapedata_collection= self._load_shapedata_collection(),
            nuts_harm           = self._load_nuts_harm(),

            # optional data
            _population_berlin   = self._load_population_data_berlin_districts() if self.epiconfig.split_berlin     else None,   
            _population_density  = self._load_population_density()               if self.epiconfig.feature_popdens  else None,
            _gisd                = self._load_gisd()                             if self.epiconfig.feature_gisd     else None,
            _population_age      = self._load_population_age()                   if self.epiconfig.feature_popage   else None
        )

        time_end = time.time()
        if self.epiconfig.verbose > 2:
            print(f'Execution of EpiDataReader took {round(time_end - time_start,3)}s')

        if self.epiconfig.verbose > 1:
            print("")
        return rawdata
            
# ============= HARMONIZATION CLASS =============
class EpiDataHarmonizer:
    """
    Harmonizes the raw data in space and time

    Parameters:
    -----------
    epiconfig: EpiConfig

    Utility:
    -------
    the orchestrate method runs all required methods, based on EpiConfig and returns an
    instance of HarmonizedEpiData and one of ContextEpiData    
    """
    def __init__(self, epiconfig: 'EpiConfig'):
        self.epiconfig = epiconfig    

    def _add_berlin_districts(self, population_nuts3: pd.DataFrame, population_berlin_districts: pd.DataFrame) -> pd.DataFrame:
        """
        when berlin to be split -> add population data by district
        based on population data for nuts3 and for berlin districts, concatenate into one df with 412 nuts3 values.
        """
        # get all berlin - population and find the percentages per district 
        total_population_berlin                                     = sum(population_berlin_districts['population_size'])
        relative_population_size_berlin_districts                   = population_berlin_districts.copy()
        relative_population_size_berlin_districts['population_size']= relative_population_size_berlin_districts['population_size'] / total_population_berlin

        # for every year, multiply these percentages with the total population in Berlin for the districts and concatenate to df
        df_11000 = population_nuts3[population_nuts3['nuts3'] == berlin_id][['year', 'population_size']].set_index('year')
            
        yearly_dfs = []  # Collect dataframes to concatenate later

        for year, base_pop in df_11000['population_size'].items():
            yearly_rows = relative_population_size_berlin_districts.copy()
            yearly_rows['year'] = int(year) # type: ignore[assignment]
            yearly_rows['population_size'] = yearly_rows['population_size'] * base_pop
            yearly_rows['population_size'] = yearly_rows['population_size'].astype(int)
            
            yearly_dfs.append(yearly_rows)

        result_df = pd.concat(yearly_dfs, ignore_index=True)
        combined  = pd.concat([population_nuts3, result_df], ignore_index=True)

        if self.epiconfig.verbose > 1:
            print(f'{checkmark} berlin districts - population data included')          
        return combined

    def _mutate_berlin_district_ids(self, epidemiology_df: pd.DataFrame) -> pd.DataFrame:
        """
        When berlin not to be split -> mutate all nuts3 values of the districts into
        berlin ones (11000) for the subsequent aggregation onto nuts3/nuts2/nuts1 levels.
        """
        epidemiology_df.loc[epidemiology_df['nuts3_key'].isin(berlin_district_ids), 'nuts3_key'] = berlin_id
        
        if self.epiconfig.verbose > 1:
            print(f'{checkmark} berlin districts renamed into berlin city')  

        return epidemiology_df

    def _aggregate_by_nuts(self, epidemiology_df: pd.DataFrame, population_df: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
        """aggregates epidemiology and population data per nuts level"""
        epi_df_aggr = epidemiology_df.groupby(['timestamp', f'{self.epiconfig.nuts_level}_key']).aggregate({'cases':'sum'}).reset_index()     
        pop_df_aggr = population_df.groupby(['year', f'{self.epiconfig.nuts_level}_key']).aggregate({'population_size':'sum'}).reset_index() 

        if self.epiconfig.verbose > 1:
            print(f'{checkmark} epidemiology and population data aggregated on nuts')    

        return epi_df_aggr, pop_df_aggr

    def _add_nuts_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """adds nuts-level column"""
        if self.epiconfig.nuts_level == "nuts1":
            df['nuts1_key']= df['nuts3_key'].str[:2]     

        elif self.epiconfig.nuts_level =='nuts2':
            df['nuts2_key']= df['nuts3_key'].str[:3]

        if self.epiconfig.verbose > 1:
            print(f'{checkmark} nuts column added')  
        return df

    def _merge_epipopdata(self, epidemiology_df: pd.DataFrame, population_df: pd.DataFrame) -> pd.DataFrame:
        """
        merge epidemiology with population data, by extracting the year from epidemiology data. 
        The year column is dropped
        """
        if 'year' not in epidemiology_df.columns:
            timestamp: pd.Series[pd.Timestamp]  = epidemiology_df['timestamp']
            epidemiology_df['year']             = timestamp.dt.year      #typing: ignore

        if self.epiconfig.verbose > 1:
            print(f'{checkmark} epidemiological- and population data merged')  

        return pd.merge(epidemiology_df, population_df, on = [f'{self.epiconfig.nuts_level}_key','year'])

    def _get_nuts_data(self, raw_nuts_names: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the nuts data with only the unique values.
        Initially the raw_nuts_names is a df of all nuts3/nuts2/nuts1 levels.
        So if nuts_level is 2 or 1, many entries can be dropped.
        """
        columns     = [f'{self.epiconfig.nuts_level}_key',f'{self.epiconfig.nuts_level}_name']
        unique_nuts = raw_nuts_names[columns].drop_duplicates().reset_index(drop=True)

        if self.epiconfig.verbose > 1:
            print(f'{checkmark} nuts levels extracted')  

        return unique_nuts
    
    def _return_tokenization_map(self, df: pd.DataFrame, token_col: str) -> Tuple[Dict[str,int], Dict[int,str]]:
        """
        from input dataframe returns two dictionaries of the tokenization:
        - {token_col-value  : token-value}      => [str, int]
        - {token-value      : token_col-value}  => [int, str]

        tokens are associated alphabetically
        """
        unique_col_values    = sorted(list(df[token_col].unique()))
        colvalue_token       = {}
        token_colvalue       = {}

        for idx, cc in enumerate(unique_col_values):
            colvalue_token[cc] = idx 
            token_colvalue[idx]= cc 

        return colvalue_token, token_colvalue
    
    def _apply_tokenization(self, df: pd.DataFrame, tokenization_map: Dict[str, int], col_to_tokenize: str, token_colname):
        """
        applies tokeninization found in tokenization map, through:

        dfc[token_colname]   = dfc[col_to_tokenize].map(tokenization_map).astype('Int64')
        """
        dfc                  = df.copy()
        dfc[token_colname]   = dfc[col_to_tokenize].map(tokenization_map).astype('Int64')
        dfc.reset_index(drop = True, inplace = True)   
        dfc.drop(columns = col_to_tokenize, inplace=True)

        return dfc  

    def _resample(self, epi_df: pd.DataFrame, temporal_freq: Literal['m','w','d']) -> pd.DataFrame:
        """ 
        resamples timestamp to requested temporal frequency
        """
        # the only disease with temp freq below w is covid_daily
        if temporal_freq == 'd':
            if self.epiconfig.disease == 'covid_daily':
                resampled_df = epi_df
            else:
                raise EpiDataOrchestrationError(f'temporal_freq == "d" is only valid for disease "covid_daily" not for {self.epiconfig.disease}')
                
        elif temporal_freq == 'w':
            if self.epiconfig.disease == 'covid_daily':
                resampled_df = (
                    epi_df.set_index(self.epiconfig.temporal_column)
                    .groupby(f'{self.epiconfig.nuts_level}_key')
                    .resample('W-MON')                                  # survstat system reports weekly data on mondays!
                    .agg({
                        'cases':            'sum',
                        'population_size':  'mean',
                    })
                    .reset_index(drop = False)
                )
            else:
                resampled_df = epi_df

        # else temporal freq  == m. has been established in epiconfig validation methods
        else:
            resampled_df = (
                    epi_df.set_index(self.epiconfig.temporal_column)
                    .groupby(f'{self.epiconfig.nuts_level}_key')
                    .resample('MS')
                    .agg({
                        'cases':            'sum',
                        'population_size':  'mean',
                    })
                    .reset_index(drop = False)
                )
            
        timestamps: pd.Series[pd.Timestamp] = resampled_df['timestamp']
        resampled_df['year']                = timestamps.dt.year
        resampled_df                        = resampled_df[[self.epiconfig.temporal_column,f'{self.epiconfig.nuts_level}_key','cases','year','population_size']]
        return resampled_df

    def _return_temporal_summary(self) -> 'EpiDataTemporalSummary':
        """returns an instance of EpiDataTemporalSummary, based on EpiConfig"""
        return EpiDataTemporalSummary(self.epiconfig.temporal_frequency,
                                    str(self.epiconfig.min_date), 
                                    str(self.epiconfig.max_date),
                                    self.epiconfig.split_trainval,
                                    self.epiconfig.split_valtest,
                                    self.epiconfig.horizon_size,
                                    self.epiconfig.horizon_leadtime,
                                    self.epiconfig.lag_num,
                                    self.epiconfig.sequence_length)        

    def orchestrate(self, rawdata: 'RawEpiData') -> Tuple['HarmonizedEpiData', 'ContextEpiData']:
        """
        The function that orchestrates all others
        """
        time_start = time.time()

        if self.epiconfig.split_berlin:
            if rawdata.population_berlin is None:
                raise EpiDataOrchestrationError("'population_berlin' attribute is not found in rawdata")
            
            population_data = self._add_berlin_districts(rawdata.population, rawdata.population_berlin)
            raw_epidata     = rawdata.disease
        else:
            raw_epidata     = self._mutate_berlin_district_ids(rawdata.disease)
            population_data = rawdata.population

        epidemiology_data   = self._add_nuts_column(raw_epidata)
        population_data     = self._add_nuts_column(population_data)
        aggregated_dfs      = self._aggregate_by_nuts(epidemiology_data, population_data)
        epipopdata          = self._merge_epipopdata(aggregated_dfs[0], aggregated_dfs[1])
        nutsnames           = self._get_nuts_data(rawdata.nuts_harm)

        epipopdata          = self._resample(epipopdata, self.epiconfig.temporal_frequency)

        tokenization_map_id = self._return_tokenization_map(epipopdata, f'{self.epiconfig.nuts_level}_key')

        tokenization_map    = {
            'nuts_node-idx' : tokenization_map_id[0],    
            'idx-nuts_node' : tokenization_map_id[1],    
        }

        # apply tokens
        epipopdata = self._apply_tokenization(epipopdata, 
                                              tokenization_map['nuts_node-idx'],
                                              f'{self.epiconfig.nuts_level}_key',
                                              self.epiconfig.id_column)
        shapedata  = self._apply_tokenization(rawdata.shapedata_node, 
                                              tokenization_map['nuts_node-idx'],
                                              f'{self.epiconfig.nuts_level}_key', 
                                              self.epiconfig.id_column)
        nutsnames  = self._apply_tokenization(nutsnames, 
                                              tokenization_map['nuts_node-idx'],
                                              f'{self.epiconfig.nuts_level}_key',         
                                              self.epiconfig.id_column)
        population_data = self._apply_tokenization(population_data,
                                                   tokenization_map['nuts_node-idx'],
                                                   f'{self.epiconfig.nuts_level}_key',
                                                   self.epiconfig.id_column)

        # extra features
        population_density_data = None
        gisd_data               = None
        population_age          = None   


        if self.epiconfig.feature_popdens:
            population_density_data = self._apply_tokenization(rawdata.population_density, 
                                                               tokenization_map['nuts_node-idx'],
                                                               f'{self.epiconfig.nuts_level}_key', 
                                                               self.epiconfig.id_column)
            
        if self.epiconfig.feature_gisd:
            gisd_data = self._apply_tokenization(rawdata.gisd, 
                                                 tokenization_map['nuts_node-idx'],
                                                 f'{self.epiconfig.nuts_level}_key', 
                                                 self.epiconfig.id_column)                        

        if self.epiconfig.feature_popage:
            population_age = self._apply_tokenization(rawdata.population_age, 
                                                      tokenization_map['nuts_node-idx'],
                                                      f'{self.epiconfig.nuts_level}_key', 
                                                      self.epiconfig.id_column)                  

        if isinstance(shapedata, pd.DataFrame):
            shapedata = gpd.GeoDataFrame(shapedata)

        harmdata = HarmonizedEpiData(
            epidata             = epipopdata,

            _population_density  = population_density_data,
            _gisd                = gisd_data,
            _population_age      = population_age
        )
        
        ctxdata = ContextEpiData(
            nuts_level          = self.epiconfig.nuts_level,
            shapedata_node      = shapedata,
            shapedata_nuts0     = rawdata.shapedata_collection['nuts0'],
            shapedata_nuts1     = rawdata.shapedata_collection['nuts1'],
            shapedata_nuts2     = rawdata.shapedata_collection['nuts2'],
            shapedata_nuts3     = rawdata.shapedata_collection['nuts3'],
            population_size     = population_data,
            nuts_harm           = nutsnames,
            tokenization_map    = tokenization_map,
            temporal_summary    = self._return_temporal_summary()            
        )

        time_end = time.time()
        if self.epiconfig.verbose > 2:
            print(f'Execution of Harmonizer took {round(time_end - time_start,3)}s')  
        if self.epiconfig.verbose > 1:
            print("")              

        return harmdata, ctxdata

# ============= PREPROCESSING CLASS =============
class EpiDataProcessor:     
    """
    Processes the harmonzied data

    Parameters:
    -----------
    epiconfig: EpiConfig
    temporal_summary: EpiDataTemporalSummary

    Utility:
    -------
    the orchestrate method runs all required methods, based on EpiConfig and returns an
    instance of ProcessedEpiData    
    """
    def __init__(self, 
                 config: 'EpiConfig', 
                 temporal_summary: EpiDataTemporalSummary):
        
        self.config             = config
        self.temporal_summary   = temporal_summary

    def _add_incidence_column(self, epipopdata: pd.DataFrame) -> pd.DataFrame:
        """adds incidence column"""
        epipopdata['incidence'] = epipopdata['cases'] / epipopdata['population_size'] * self.config.incidence_scalar     

        if self.config.verbose > 1:
            print(f'{checkmark} incidence column added')    

        return epipopdata
    
    def _drop_cases_column(self, epipopdata: pd.DataFrame) -> pd.DataFrame:
        """drops cases column -> only if target => incidence"""

        if self.config.verbose > 1:
            print(f'{checkmark} cases column removed')    

        return epipopdata.drop(columns=['cases']) 
       
    def _filter_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """using the timestamp column, filter on min/max date => using those determined in the temporal summary"""
        
        # Use extended min date from temporal summary
        dfc     = df.copy()
        mindate = self.temporal_summary.get_extended_dates()['min']
        maxdate = self.temporal_summary.get_extended_dates()['max']
             
        dfc = dfc.loc[dfc['timestamp'] <  maxdate].reset_index(drop=True)         
        dfc = dfc.loc[dfc['timestamp'] >= mindate].reset_index(drop=True)     

        if self.config.verbose > 1:
            print(f'{checkmark} filtered on dates')    

        return dfc
    
    def _filter_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """using the year column, filter on min/max date => using those determined in the temporal summary"""        
        dfc = df.copy() 
        minyear = self.temporal_summary.get_extended_dates()['min'].year

        # for max year its different in year => when max date '2021-02-01' 
        # we still need the data at '2021-01-01' so we can only include year < max_year+1
        maxyear = self.temporal_summary.get_extended_dates()['max'].year + 1

        dfc = dfc.loc[dfc['year'] <  maxyear].reset_index(drop=True)         
        dfc = dfc.loc[dfc['year'] >= minyear].reset_index(drop=True)             

        return dfc

    def orchestrate(self, harmonizeddata: 'HarmonizedEpiData') -> 'ProcessedEpiData':
        
        time_start = time.time()
        epipopdata = self._add_incidence_column(harmonizeddata.epidata.copy())

        if self.config.target_column != 'cases':
            epipopdata = self._drop_cases_column(epipopdata)        

        epipopdata = self._filter_dates(epipopdata)

        # extra features; initiating them on None
        population_density_data = None
        gisd_data               = None      
        population_age          = None   

        if self.config.feature_popdens:
            population_density_data = self._filter_years(harmonizeddata.population_density)

        if self.config.feature_gisd:
            gisd_data = self._filter_years(harmonizeddata.gisd)
                   
        if self.config.feature_popage:
            population_age = self._filter_years(harmonizeddata.population_age)                       

        processed_data = ProcessedEpiData(epidata            = epipopdata,
                                          
                                          _population_density= population_density_data,
                                          _gisd              = gisd_data,
                                          _population_age    = population_age
                                          )
        time_end = time.time()
        if self.config.verbose > 2:
            print(f'Execution of EpiDataProcessor took {round(time_end - time_start,3)}s')    
        if self.config.verbose > 1:
            print("")

        return processed_data

# ============= FEATURE CLASS =============            
class EpiFeatureBuilder:
    """
    Builds the features based on the processed data

    Parameters:
    -----------
    epiconfig: EpiConfig
    column_registration: ColumnRegistration,
    temporal_summary: EpiDataTemporalSummary
    
    Utility:
    -------
    the orchestrate method runs all required methods, based on EpiConfig and returns an
    instance of FeatureEpiData    
    """    
    def __init__(self, 
                 epiconfig: 'EpiConfig', 
                 column_registration: ColumnRegistration,
                 temporal_summary: EpiDataTemporalSummary):
        
        self.epiconfig          = epiconfig
        self.column_registration= column_registration
        self.temporal_summary   = temporal_summary

    def _add_time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        adds all time indices depending on epiconfig: 
        can be any combination of time_index_d/w/m 
        """
        
        dfc                                 = df.copy()
        timestamps: pd.Series[pd.Timestamp] = dfc[self.epiconfig.temporal_column]
        iso_calendar                        = timestamps.dt.isocalendar()

        years           = iso_calendar['year']
        weeks           = iso_calendar['week']   
        months          = timestamps.dt.month              
        days            = iso_calendar['day']  # 1=Monday, 7=Sunday        

        sin_col_basis = f'tt_sin'
        cos_col_basis = f'tt_cos'

        # ============ day in week ===========
        if self.epiconfig.time_index_d: 
            if self.temporal_summary.temporal_frequency != 'd':
                raise EpiDataOrchestrationError(f"can't put temporal index for day in week for data that has no daily temporal frequency")
            days_in_week = 7

            sin_col_d = sin_col_basis+"_d"
            cos_col_d = cos_col_basis+"_d"

            dfc[sin_col_d] = np.sin(2 * np.pi * days / days_in_week)
            dfc[cos_col_d] = np.cos(2 * np.pi * days / days_in_week)    

            self.column_registration.add_column(sin_col_d, 'feature', needs_normalization=False, transformation_group=None)    
            self.column_registration.add_column(cos_col_d, 'feature', needs_normalization=False, transformation_group=None)     

        # ============ week in year ===========
        if self.epiconfig.time_index_w:         
            if self.temporal_summary.temporal_frequency not in ['d','w']:
                raise EpiDataOrchestrationError(f"can't put temporal index for week in year data that has no daily or weekly temporal frequency")

            def _year_has_53_weeks(year: int) -> bool:
                dec_28 = pd.Timestamp(year=year, month=12, day=28)
                return dec_28.isocalendar()[1] == 53
            
            unique_years    = years.unique()
            year_week_count = {year: 53 if _year_has_53_weeks(year) else 52 for year in unique_years}
            weeks_in_year   = years.map(year_week_count)       

            sin_col_w = sin_col_basis+"_w"
            cos_col_w = cos_col_basis+"_w"

            dfc[sin_col_w] = np.sin(2 * np.pi * weeks / weeks_in_year)
            dfc[cos_col_w] = np.cos(2 * np.pi * weeks / weeks_in_year)

            self.column_registration.add_column(sin_col_w, 'feature', needs_normalization=False, transformation_group=None)    
            self.column_registration.add_column(cos_col_w, 'feature', needs_normalization=False, transformation_group=None)     

        # ============ month in year ===========
        if self.epiconfig.time_index_m:

            months_in_year = 12

            sin_col_m = sin_col_basis+"_m"
            cos_col_m = cos_col_basis+"_m"

            dfc[sin_col_m] = np.sin(2 * np.pi * months / months_in_year)
            dfc[cos_col_m] = np.cos(2 * np.pi * months / months_in_year)

            self.column_registration.add_column(sin_col_m, 'feature', needs_normalization=False, transformation_group=None)    
            self.column_registration.add_column(cos_col_m, 'feature', needs_normalization=False, transformation_group=None)                 

        return dfc 
    
    def _lag_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        lag variable according to epiconfig
        """
        dfc = df.copy()
        
        for lag in range(0, self.epiconfig.lag_num):
            feature     = f'{self.epiconfig.lag_column}_lag{lag}'
            dfc[feature]= dfc.groupby(self.epiconfig.id_column)[self.epiconfig.lag_column].shift(lag)
            
            if self.epiconfig.lag_column == self.epiconfig.target_column:
                reference_normalization = 'target'
            elif lag == 0:
                reference_normalization = 'self'
            else:
                reference_normalization = f'{self.epiconfig.lag_column}_lag0'

            self.column_registration.add_column(
                feature, 
                'feature',
                needs_normalization  =True,
                transformation_group =reference_normalization
            )
            
        if self.epiconfig.verbose > 1:
            print(f'{checkmark} lags added') 

        return dfc.dropna().reset_index(drop = True)
  
    def _shift_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """shift target by horizon_leadtime (but into the future => negative)"""
        dfc          = df.copy()
        dfc['target']= dfc.groupby(self.epiconfig.id_column)[self.epiconfig.target_column].shift(-(self.epiconfig.horizon_leadtime))
        return dfc

    def _rename_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename target column to 'target'"""
        if self.epiconfig.verbose > 1:
            print(f'{checkmark} target column renamed as such') 
        return df.rename(columns={f'{self.epiconfig.target_column}_future': 'target'})

    def _reorder_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rearrange columns in predefined order"""
        dfc          = df.copy()

        if self.epiconfig.verbose > 1:
            print(f'{checkmark} columns reordered') 

        return dfc[self.column_registration.registered_columns]

    def _add_delta_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute first difference of target column, store t-1 anchor for reversal."""
        col = self.epiconfig.target_column
        df[f'{col}_anchor'] = df.groupby(self.epiconfig.id_column)[col].shift(1)
        df[col] = df.groupby(self.epiconfig.id_column)[col].diff()

        return df.dropna().reset_index(drop=True)

    def orchestrate(self, processed_data: 'ProcessedEpiData') -> 'FeatureEpiData':
        time_start   = time.time()
        feature_data = processed_data.epidata.copy()

        feature_data = feature_data.drop(columns=['population_size'], errors='ignore')
        if self.epiconfig.verbose > 1:
            print(f'{checkmark} population size removed')

        if self.epiconfig.feature_popdens:
            self.column_registration.add_column(
                'population_density',
                'feature',
                needs_normalization=True,
                transformation_group='self'
            )            
            feature_data = pd.merge(feature_data, processed_data.population_density, on = [self.epiconfig.id_column, 'year'])

        if self.epiconfig.feature_gisd:
            self.column_registration.add_column(
                f'gisd',
                'feature',
                needs_normalization=False
            )               
            feature_data = pd.merge(feature_data, processed_data.gisd, on = [self.epiconfig.id_column, 'year'])           

        if self.epiconfig.feature_popage:
            processed_feature_popage = processed_data.population_age

            for cc in processed_feature_popage.columns:
                if cc not in ['year',self.epiconfig.id_column,'population_size']:
                    self.column_registration.add_column(
                        cc,
                        'feature',
                        needs_normalization=False
                    )         
                elif cc == 'population_size':
                    if self.epiconfig.feature_popsize:
                        self.column_registration.add_column(
                            cc,
                            'feature',
                            needs_normalization=True,
                            transformation_group='self'
                        )  
                    else:
                        processed_feature_popage.drop(columns = ['population_size'], inplace = True)
                                          
            feature_data = pd.merge(feature_data, processed_feature_popage, on = [self.epiconfig.id_column, 'year'])               

        # Delta transform: must happen before lags and target shift,
        # so that lag features and the forecast target are all in delta-space
        if self.epiconfig.predict_difference:
            feature_data = self._add_delta_column(feature_data)
            self.column_registration.update_transformation(
                'target',
                {'delta': {'anchor_col': f'{self.epiconfig.target_column}_anchor'}}
            )
            self.column_registration.add_column(
                f'{self.epiconfig.target_column}_anchor',
                'context',
                needs_normalization=False
            )
            if self.epiconfig.verbose > 1:
                print(f'{checkmark} delta transform applied')

        feature_data = self._add_time_index(feature_data)
        feature_data = self._lag_variable(feature_data)
        feature_data = self._shift_target(feature_data)
        feature_data = self._rename_target(feature_data)
        feature_data = self._reorder_df(feature_data)

        time_end = time.time()
        if self.epiconfig.verbose > 2:
            print(f'Execution of EpiFeatureBuilder took {round(time_end - time_start,3)}s')

        return FeatureEpiData(epidata=feature_data)

# ============= NORMALIZER CLASS ============= 
class EpiNormalizer:
    """
    Normalizes the FeatureData

    Parameters:
    -----------
    epiconfig: EpiConfig
    column_registration: ColumnRegistration,
    temporal_summary: EpiDataTemporalSummary
    
    Utility:
    -------
    the orchestrate method runs all required methods, based on EpiConfig and returns an
    instance of NormalizedEpiData    
    """      
    def __init__(self, 
                 epiconfig: 'EpiConfig', 
                 column_registration: ColumnRegistration, 
                 temporal_summary: EpiDataTemporalSummary):
        
        self.epiconfig              = epiconfig 
        self.temporal_summary       = temporal_summary
        self.column_registration    = column_registration
        self.normalization_functions= {
            'pipeline': {'minmax': pipeline_minmax_normalization,   'zscore': pipeline_zscore_normalization},
            'apply':    {'minmax': apply_minmax_scaling,            'zscore': apply_zscore_scaling}
        }

    def _set_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """set split columns based on temporal_summary"""
        splits = self.temporal_summary.get_target_splits()
        
        df['train'] = df[self.epiconfig.temporal_column] < splits['trainval']
        df['val']   = (df[self.epiconfig.temporal_column] >= splits['trainval']) & (df[self.epiconfig.temporal_column] < splits['valtest'])
        df['test']  = df[self.epiconfig.temporal_column] >= splits['valtest']
        
        for split_col in ['train', 'val', 'test']:
            self.column_registration.add_column(split_col, 'split')
        
        if self.epiconfig.verbose > 1:
            print(f'{checkmark} split columns added (input timeline)')
        
        return df

    def _apply_log_transform(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """log_transform the columns specified"""
        df_transformed                              = df.copy()
           
        if col not in df_transformed.columns:
            raise EpiDataOrchestrationError(f"{col} not found in df. Couldn't be log-transformed")

        df_transformed[col]                         = np.log(df_transformed[col] + self.epiconfig.log_shift)
      
        if self.epiconfig.verbose > 1:
            print(f'{checkmark} {col} logged')     

        return df_transformed

    def _update_colregistry_postlog(self, col:str) -> None:
            """register the logging in colregistry"""
            self.column_registration.update_transformation(
                col, 
                {'log': self.epiconfig.log_shift}
            )           

    def _orchestrate_logging(self, split_data: pd.DataFrame) -> pd.DataFrame:
        """if nothing to log, then return origional df"""
        if self.epiconfig.log_transform:
            cols_to_log     = []
            for col in self.epiconfig.log_transform:

                # if col == future target then register log for target
                if col == self.epiconfig.target_column:
                    cols_to_log += ['target']
                    self._update_colregistry_postlog('target')

                # if col == lag column then do the log for all lagged columns, but only register in transformation for lag0
                # not in combination with previous condition. If target == lag, the lag column follows the normalization of target
                elif col == self.epiconfig.lag_column:
                    cols_to_log += [f'{self.epiconfig.lag_column}_lag{lag}' for lag in range(0, self.epiconfig.lag_num)]
                    self._update_colregistry_postlog(f'{self.epiconfig.lag_column}_lag0')

                # else register log column-specifically
                else:
                    cols_to_log += [col]
                    self._update_colregistry_postlog(col)

            for col in cols_to_log:
                split_data = self._apply_log_transform(split_data, col)    

        return split_data   

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        normalizes data and stores information in column_registration
        NOTE: not only the target is normalized based on the training data: that goes for all features!
        I haven't figured out if this is an issue, but I imagine it is not.
        For what it's worth, it's an easy fix.     
        """
        train_df        = df[df['train']]
        normalized_df   = df.copy()

        # if normalization_method is excplicity set to None then return df not-normalized
        if not self.epiconfig.normalization_method:
            return normalized_df

        elif self.epiconfig.normalization_method not in self.normalization_functions['apply']:
            raise EpiDataOrchestrationError(f'No normalization function {self.epiconfig.normalization_method} found.')

        ### First pass ###
        # get all transformation parameters per group (.transformation_group = 'self')
        for col_entry in self.column_registration.columns:
            
            # Skip columns that don't have normalization attribute
            if not col_entry.transformation:
                continue # continue with next col_entry
            
            # Only calculate params for columns with independent normalization (normalization_group == 'self')
            if col_entry.transformation_group == 'self':
                _, norm_parameters = self.normalization_functions['pipeline'][self.epiconfig.normalization_method](
                    train_df, 
                    [col_entry.column_name]
                )
                
                # Update transformation with normalization parameters
                self.column_registration.update_transformation(
                    col_entry.column_name,
                    {'normalization': norm_parameters[col_entry.column_name]}
                )

        if self.epiconfig.verbose > 1:
            print(f'{checkmark} normalization parameters retrieved and stored')     
            
        ### Second pass ###
        # apply those parameters to all columns with the relevant transformation group - reference
        for col_entry in self.column_registration.columns:
            
            # Skip columns that don't have normalization attribute
            if not col_entry.transformation:
                continue # continue with next col_entry
                  
            # Determine which normalization parameters to use
            # independent transformation first
            if col_entry.transformation_group == 'self':
                
                params = {col_entry.column_name: col_entry.transformation_params['normalization']}
                if self.epiconfig.verbose > 2:
                    print(f"{col_entry.column_name} normalized independently")
            
            # dependent transformation: expect a referral
            else:
                # Use reference column's normalization
                ref_col_entry = self.column_registration.get_by_name(col_entry.transformation_group)
                
                params = {col_entry.column_name: ref_col_entry.transformation_params['normalization']}
                if self.epiconfig.verbose > 2:
                    print(f"{col_entry.column_name} normalized based on {ref_col_entry.column_name}")
            
            # Apply normalization
            normalized_df = self.normalization_functions['apply'][self.epiconfig.normalization_method](
                normalized_df, 
                [col_entry.column_name], 
                params
            )

        if self.epiconfig.verbose > 1:
            print(f'{checkmark} normalization applied')     

        return normalized_df

    def orchestrate(self, feature_data: 'FeatureEpiData') -> 'NormalizedEpiData':
        time_start      = time.time()
        split_data      = self._set_splits(feature_data.epidata.copy())

        # depending on config, logging may not be done
        split_data      = self._orchestrate_logging(split_data)
        normalized_data = self._normalize(split_data)

        time_end = time.time()
        if self.epiconfig.verbose > 2:
            print(f'Execution of EpiNormalizer took {round(time_end - time_start,3)}s')        
        if self.epiconfig.verbose > 1:
            print("")
        return NormalizedEpiData(epidata=normalized_data)

# ============= FINALIZER CLASS ============= 
class EpiDataFinalizer:
    """
    Finalizes the data

    Parameters:
    -----------
    epiconfig: EpiConfig
    column_registration: ColumnRegistration
    
    Utility:
    -------
    the orchestrate method runs all required methods, based on EpiConfig and returns an
    instance of FinalizedEpiData    
    """   
    def __init__(self, 
                 epiconfig: 'EpiConfig', 
                 column_registration: ColumnRegistration):
        self.epiconfig = epiconfig 
        self.column_registration = column_registration

    def _create_pred_col_entry(self) -> None:
        """
        while pred doesn't exist in the data, models will end up with these columns.
        if prediction_quantiles are inputted in EpiConfig, these will be created here.
        """

        needs_normalization  = False if self.epiconfig.target_column == 'cases' else True
        transformation_group = 'target'if self.epiconfig.target_column != 'cases' else None

        self.column_registration.add_column(
            'pred',
            'pred',
            needs_normalization=needs_normalization,
            transformation_group=transformation_group
        )      
        quantiles = self.epiconfig.quantiles

        if quantiles:
            for quantile_idx, quantile in enumerate(quantiles):
                self.column_registration.add_column(
                    f'pred_q{quantile_idx}',
                    'pred',
                    needs_normalization=needs_normalization,
                    transformation_group=transformation_group
                )                     

    def _add_horizons(self, df: pd.DataFrame) -> pd.DataFrame:
        """adds target columns when horizon_size>1"""
        base_lead = self.epiconfig.horizon_leadtime
        for additional_steps in range(0, self.epiconfig.horizon_size):
            steps_ahead = base_lead + additional_steps
            target_col = f'target_lead{steps_ahead}'
            
            # Shift from the base target
            df[target_col] = df.groupby(self.epiconfig.id_column)[f'target'].shift(-additional_steps)
            
            needs_normalization  = False if self.epiconfig.target_column == 'cases' else True
            transformation_group = 'target'if self.epiconfig.target_column != 'cases' else None

            # Register in column registry
            self.column_registration.add_column(
                target_col,
                'target',
                needs_normalization=needs_normalization,
                transformation_group=transformation_group
            )
        
        if self.epiconfig.verbose > 1:
            print(f'{checkmark} targets for all horizons added')              
        return df.drop(columns = ['target'])

    def _drop_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """drop nans"""
        if self.epiconfig.verbose > 1:
            print(f"{checkmark} nans dropped")
        return df.dropna()

    def _set_target_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        There's two special cases for which we adjust the type of target
        - target == 'cases' & prediction_mode == 'regression'       =>  set target type to int
        - target == 'cases' & prediction_mode == 'classification'   =>  set target type to int(0,1)       
        """
        if self.epiconfig.target_column == 'cases':
            if self.epiconfig.prediction_mode == 'regression':
                for col in self.column_registration.target_columns:
                    if col in df.columns.tolist():
                        df[col] = df[col].astype(int)     

                if self.epiconfig.verbose > 1:
                    print(f"{checkmark} target column(s) set as integer")    

            elif self.epiconfig.prediction_mode == 'classification':
                for col in self.column_registration.target_columns:
                    if col in df.columns.tolist():                  
                        df.loc[df[col] > 0, col] = 1  
                        df[col] = df[col].astype(int)

                if self.epiconfig.verbose > 1:
                    print(f"{checkmark} target column(s) set as class")        

        return df   

    def _denormalize(self, normalized_df: pd.DataFrame) -> pd.DataFrame:
            """after shifting around and renaming columns, perform a denormalization for baseline models"""
            dfc = normalized_df.copy()
            
            # Get normalization method
            norm_method = self.epiconfig.normalization_method
            
            if not norm_method:
                return dfc
            
            # Reverse transformations for each column
            for col_entry in self.column_registration.columns:

                if col_entry.transformation:
                
                    # skip registrered 'pred' columns
                    if col_entry.column_name not in dfc.columns:
                        continue
            
                    # if self-normalization
                    if col_entry.transformation_group == 'self':
                        params = col_entry.transformation_params
                
                    # if referral-based normalization
                    else:
                        reference_entry = self.column_registration.get_by_name(col_entry.transformation_group)
                        print(f'for col {col_entry.column_name}: reference: {reference_entry.column_name}')
                        params = reference_entry.transformation_params
                
                    # Reverse normalization
                    if 'normalization' in params:
                        if norm_method == 'minmax':
                            dfc = reverse_minmax_scaling(dfc, params['normalization'], column=col_entry.column_name)
                        elif norm_method == 'zscore':
                            dfc = reverse_zscore_scaling(dfc, params['normalization'], column=col_entry.column_name)
                    
                    # Reverse log transform
                    if 'log' in params:
                        dfc = reverse_log(dfc, params['log'], column=col_entry.column_name)               

            return dfc

    def orchestrate(self, normalized_data: NormalizedEpiData) -> 'FinalizedEpiData':
        time_start = time.time()
        dfc         = normalized_data.epidata

        dfc = self._add_horizons(dfc)
        self._create_pred_col_entry()
    
        dfc_normalized_nanfree      = self._drop_nans(dfc)
        dfc_denormalized_nanfree    = self._denormalize(dfc_normalized_nanfree)

        # If target == 'cases' => target columns should be integers
        dfc_normalized_nanfree      = self._set_target_type(dfc_normalized_nanfree) 
        dfc_denormalized_nanfree    = self._set_target_type(dfc_denormalized_nanfree)        

        time_end = time.time()
        if self.epiconfig.verbose > 2:
            print(f'Execution of EpiFinalizer took {round(time_end - time_start,3)}s')  
        if self.epiconfig.verbose:
            print(f'{checkmark}{checkmark} All data finalized with correct timestamp alignment') 
        if self.epiconfig.verbose > 1:
            print("")

        return FinalizedEpiData(
            data        = dfc_normalized_nanfree,
            data_denorm = dfc_denormalized_nanfree,
        )