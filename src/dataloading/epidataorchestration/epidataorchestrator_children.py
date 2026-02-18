import pandas as pd
import geopandas as gpd
from typing import Tuple, Dict, Union, Literal, TYPE_CHECKING
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from ...utils.constants import berlin_district_ids, berlin_id
from ...utils.textformatting import warning_emoji, checkmark

from .temporal_summary import EpiDataTemporalSummary
from .column_registry import ColumnRegistration, ColEntryMissingTransformationReferralError, ColEntryMissingTransformationError
from .epidatacontainers import RawEpiData, HarmonizedData, ContextData, ProcessedEpiData, FeatureEpiData, NormalizedEpiData, FinalizedEpiData
from .normalization import apply_minmax_scaling, apply_zscore_scaling, pipeline_minmax_normalization, pipeline_zscore_normalization
from .normalization import reverse_log, reverse_minmax_scaling, reverse_zscore_scaling

if TYPE_CHECKING:
    from .epiconfig import EpiConfig

class DataOrchestrationError(Exception):
    def __init__(self, explanation: str):
        statement = "Data Orchestration couldn't be run" + "\n" + explanation
        super().__init__(statement)    

# ============= DATA IMPORTATION CLASS =============
class EpiDataReader:
    """
    """
    
    def __init__(self, config: 'EpiConfig'):
        self.config = config
    
    def _load_disease_data(self) -> pd.DataFrame:
        """
        loads disease data cleaned from survstat

        df looks like:
        __________________________________________________________
        | 'week' | 'nuts3_key' | 'cases' | 'year ' | 'timestamp' |
        """        
        filepath = self.config.get_disease_path()
                
        df = pd.read_csv(
            filepath,
            parse_dates = ['timestamp'],
            dtype       = {'kz_kreis':  str, 
                           'cases':     int}
        ).rename(columns={'kz_kreis': 'nuts3_key'})
        
        if self.config.verbose > 1:
            print(f"{checkmark} Loaded raw disease data")
        
        return df
    
    def _load_population_data(self) -> pd.DataFrame:
        """
        loads population data

        df looks like:
        _____________________________________________
        | 'nuts3_key' | 'year ' | 'population_size' |
        """         
        filepath = self.config.get_population_path()
        
        df = pd.read_csv(
            filepath,
            dtype   = {'nuts3': str}
        ).rename(columns = {'nuts3':'nuts3_key'})
        
        if self.config.verbose > 1:
            print(f"{checkmark} Loaded raw population data")
        
        return df
    
    def _load_population_data_berlin_districts(self) -> pd.DataFrame:
        """
        loads population data (I created myself) for districts in Berlin

        df looks like:
        ___________________________________
        | 'nuts3_key' | 'population_size' |
        """          
        filepath = self.config.get_population_berlin_districts_path()
        
        df = pd.read_csv(
            filepath,
            dtype={'nuts3': str}
            ).rename(columns = {'nuts3':'nuts3_key'})
        
        if self.config.verbose > 1:
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
        filepath = self.config.get_nuts_shapefile_path()
        
        gdf = gpd.read_file(filepath).drop(columns=['level'], errors='ignore')
        
        if self.config.verbose > 1:
            print(f"{checkmark} Loaded raw shapedata ({len(gdf)} regions)")
        
        return gdf
    
    def _load_shapedata_collection(self) -> Dict[str, gpd.GeoDataFrame]:

        filepaths = self.config.get_shapefile_paths()

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
        filepath = self.config.get_nuts_harmonization_path()   
        
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        
        if self.config.verbose > 1:
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
        filepath = self.config.get_population_density_path()
        
        df = pd.read_csv(
            filepath,
            dtype={f'{self.config.nuts_level}_key': str}
            )
        
        if self.config.verbose > 1:
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
        filepath = self.config.get_population_age_path()
        
        df = pd.read_csv(
            filepath,
            dtype={f'{self.config.nuts_level}': str},
            parse_dates=['timestamp']
            )
        
        df = df.rename(columns = {self.config.nuts_level : f'{self.config.nuts_level}_key'})
        df['year'] = df['timestamp'].dt.year
        df.drop(columns = 'timestamp', inplace = True)

        if self.config.verbose > 1:
            print(f"{checkmark} Loaded population density")
        
        return df        

    def _load_gisd(self) -> pd.DataFrame:
        """
        loads population density data for the specified nuts level

        df looks like:
        _____________________________________________________
        | '{nuts_level}_key' | 'year' | 'gisd ' |

        where '{nuts_level}' is self.nuts_level in epiconfig.
        """          
        filepath = self.config.get_gisd_path()
        
        df = pd.read_csv(
            filepath,
            dtype={f'{self.config.nuts_level}_key': str}
            )
                
        if self.config.verbose > 1:
            print(f"{checkmark} Loaded population density")
        
        return df

    def orchestrate(self) -> RawEpiData:
        time_start = time.time()

        rawdata = RawEpiData(
            disease             = self._load_disease_data(),
            population          = self._load_population_data(),
            shapedata_node      = self._load_node_shapedata(),
            shapedata_collection= self._load_shapedata_collection(),
            nuts_harm           = self._load_nuts_harm(),

            population_berlin   = self._load_population_data_berlin_districts() if self.config.split_berlin     else None,   
            population_density  = self._load_population_density()               if self.config.feature_popdens  else None,
            gisd                = self._load_gisd()                             if self.config.feature_gisd     else None,
            population_age      = self._load_population_age()                   if self.config.feature_popage   else None
        )

        time_end = time.time()
        if self.config.verbose > 2:
            print(f'Execution of EpiDataReader took {round(time_end - time_start,3)}s')

        if self.config.verbose > 1:
            print("")
        return rawdata
            
# ============= HARMONIZATION CLASS =============
class Harmonizer:
    def __init__(self, config: 'EpiConfig'):
        self.config = config    

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
            yearly_rows['year'] = year
            yearly_rows['population_size'] = yearly_rows['population_size'] * base_pop
            yearly_rows['population_size'] = yearly_rows['population_size'].astype(int)
            
            yearly_dfs.append(yearly_rows)

        result_df = pd.concat(yearly_dfs, ignore_index=True)
        combined  = pd.concat([population_nuts3, result_df], ignore_index=True)

        if self.config.verbose > 1:
            print(f'{checkmark} berlin districts - population data included')          
        return combined

    def _mutate_berlin_district_ids(self, epidemiology_df: pd.DataFrame) -> pd.DataFrame:
        """
        When berlin not to be split -> mutate all nuts3 values of the districts into
        berlin ones (11000) for the subsequent aggregation onto nuts3/nuts2/nuts1 levels.
        """
        epidemiology_df.loc[epidemiology_df['nuts3_key'].isin(berlin_district_ids), 'nuts3_key'] = '11000'
        
        if self.config.verbose > 1:
            print(f'{checkmark} berlin districts renamed into berlin city')  

        return epidemiology_df

    def _aggregate_by_nuts(self, epidemiology_df: pd.DataFrame, population_df: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
        """aggregates epidemiology and population data per nuts level"""
        epi_df_aggr = epidemiology_df.groupby(['timestamp', f'{self.config.nuts_level}_key']).aggregate({'cases':'sum'}).reset_index()     
        pop_df_aggr = population_df.groupby(['year', f'{self.config.nuts_level}_key']).aggregate({'population_size':'sum'}).reset_index() 

        if self.config.verbose > 1:
            print(f'{checkmark} epidemiology and population data aggregated on nuts')    

        return epi_df_aggr, pop_df_aggr

    def _add_nuts_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """adds nuts-level column"""
        if self.config.nuts_level == "nuts1":
            df['nuts1_key']= df['nuts3_key'].str[:2]     

        elif self.config.nuts_level =='nuts2':
            df['nuts2_key']= df['nuts3_key'].str[:3]

        if self.config.verbose > 1:
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

        if self.config.verbose > 1:
            print(f'{checkmark} epidemiological- and population data merged')  

        return pd.merge(epidemiology_df, population_df, on = [f'{self.config.nuts_level}_key','year'])

    def _get_nuts_data(self, raw_nuts_names: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the nuts data with only the unique values.
        Initially the raw_nuts_names is a df of all nuts3/nuts2/nuts1 levels.
        So if nuts_level is 2 or 1, many entries can be dropped.
        """
        columns     = [f'{self.config.nuts_level}_key',f'{self.config.nuts_level}_name']
        unique_nuts = raw_nuts_names[columns].drop_duplicates().reset_index(drop=True)

        if self.config.verbose > 1:
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
        dfc                  = df.copy()
        dfc[token_colname]   = dfc[col_to_tokenize].map(tokenization_map).astype('Int64')
        dfc.reset_index(drop = True, inplace = True)   
        dfc.drop(columns = col_to_tokenize, inplace=True)

        return dfc  

    def _resample(self, epi_df: pd.DataFrame, temporal_freq: Literal['m','w','d']) -> pd.DataFrame:

        if temporal_freq == 'd':
            if self.config.disease == 'covid_daily':
                resampled_df = epi_df
            else:
                raise DataOrchestrationError(f'temporal_freq == "d" is only valid for disease "covid_daily" not for {self.config.disease}')
        
        # the only disease with temp freq below w is covid_daily
        elif temporal_freq == 'w':
            if self.config.disease == 'covid_daily':
                resampled_df = (
                    epi_df.set_index(self.config.temporal_column)
                    .groupby(f'{self.config.nuts_level}_key')
                    .resample('W-MON')
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
                    epi_df.set_index(self.config.temporal_column)
                    .groupby(f'{self.config.nuts_level}_key')
                    .resample('MS')
                    .agg({
                        'cases':            'sum',
                        'population_size':  'mean',
                    })
                    .reset_index(drop = False)
                )
            
        timestamps: pd.Series[pd.Timestamp] = resampled_df['timestamp']
        resampled_df['year']                = timestamps.dt.year
        resampled_df                        = resampled_df[[self.config.temporal_column,f'{self.config.nuts_level}_key','cases','year','population_size']]
        return resampled_df

    def _return_temporal_summary(self) -> 'EpiDataTemporalSummary':
        return EpiDataTemporalSummary(self.config.temporal_frequency,
                                    str(self.config.min_date), 
                                    str(self.config.max_date),
                                    self.config.split_trainval,
                                    self.config.split_valtest,
                                    self.config.horizon_size,
                                    self.config.horizon_leadtime,
                                    self.config.lag_num,
                                    self.config.sequence_length)        

    def orchestrate(self, rawdata: 'RawEpiData') -> Tuple['HarmonizedData', 'ContextData']:
        """
        The function that orchestrates all others
        """
        time_start = time.time()

        if self.config.split_berlin:
            if rawdata.population_berlin is None:
                raise DataOrchestrationError("'population_berlin' attribute is not found in rawdata")
            
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

        epipopdata          = self._resample(epipopdata, self.config.temporal_frequency)

        tokenization_map_id = self._return_tokenization_map(epipopdata, f'{self.config.nuts_level}_key')

        tokenization_map    = {
            'nuts_node-idx' : tokenization_map_id[0],    
            'idx-nuts_node' : tokenization_map_id[1],    
        }

        epipopdata = self._apply_tokenization(epipopdata, tokenization_map['nuts_node-idx'],f'{self.config.nuts_level}_key',        self.config.id_column)
        shapedata  = self._apply_tokenization(rawdata.shapedata_node, tokenization_map['nuts_node-idx'],f'{self.config.nuts_level}_key', self.config.id_column)
        nutsnames  = self._apply_tokenization(nutsnames, tokenization_map['nuts_node-idx'],f'{self.config.nuts_level}_key',         self.config.id_column)

        # extra features
        if self.config.feature_popdens:
            if rawdata.population_density is None: 
                raise DataOrchestrationError('no population density found in rawdata')
            population_density_data = self._apply_tokenization(rawdata.population_density, tokenization_map['nuts_node-idx'],f'{self.config.nuts_level}_key', self.config.id_column)
        else:
            population_density_data = None

        if self.config.feature_gisd:
            if rawdata.gisd is None: 
                raise DataOrchestrationError('no gisd found in rawdata')
            gisd_data = self._apply_tokenization(rawdata.gisd, tokenization_map['nuts_node-idx'],f'{self.config.nuts_level}_key', self.config.id_column)            
        else:
            gisd_data = None

        if self.config.feature_popage:
            if rawdata.population_age is None: 
                raise DataOrchestrationError('no population age found in rawdata')
            population_age = self._apply_tokenization(rawdata.population_age, tokenization_map['nuts_node-idx'],f'{self.config.nuts_level}_key', self.config.id_column)            
        else:
            population_age = None            

        if isinstance(shapedata, pd.DataFrame):
            shapedata = gpd.GeoDataFrame(shapedata)

        # temporal summary
        temporal_summary = self._return_temporal_summary()

        harmdata = HarmonizedData(
            epidata             = epipopdata,

            population_density  = population_density_data,
            gisd                = gisd_data,
            population_age      = population_age
        )
        
        ctxdata = ContextData(
            nuts_level          = self.config.nuts_level,
            shapedata_node      = shapedata,
            shapedata_nuts0     = rawdata.shapedata_collection['nuts0'],
            shapedata_nuts1     = rawdata.shapedata_collection['nuts1'],
            shapedata_nuts2     = rawdata.shapedata_collection['nuts2'],
            shapedata_nuts3     = rawdata.shapedata_collection['nuts3'],
            nuts_harm           = nutsnames,
            tokenization_map    = tokenization_map,
            temporal_summary    = temporal_summary            
        )

        time_end = time.time()
        if self.config.verbose > 2:
            print(f'Execution of Harmonizer took {round(time_end - time_start,3)}s')  
        if self.config.verbose > 1:
            print("")              

        return harmdata, ctxdata

# ============= PREPROCESSING CLASS =============
class EpiDataProcessor:     
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
       
    def _filter_dates(self, df) -> pd.DataFrame:
        # Use extended min date from temporal summary
        dfc     = df.copy()
        mindate = self.temporal_summary.get_extended_dates()['min']
        maxdate = self.temporal_summary.get_extended_dates()['max']
             
        
        dfc = dfc.loc[dfc['timestamp'] <  maxdate].reset_index(drop=True)         
        dfc = dfc.loc[dfc['timestamp'] >= mindate].reset_index(drop=True)     

        if self.config.verbose > 1:
            print(f'{checkmark} filtered on dates')    

        return dfc
    
    def _filter_years(self, df) -> pd.DataFrame:
        dfc = df.copy() 
        minyear = self.temporal_summary.get_extended_dates()['min'].year
        maxyear = self.temporal_summary.get_extended_dates()['max'].year + 1

        dfc = dfc.loc[dfc['year'] <  maxyear].reset_index(drop=True)         
        dfc = dfc.loc[dfc['year'] >= minyear].reset_index(drop=True)             

        return dfc

    def orchestrate(self, harmonizeddata: 'HarmonizedData') -> 'ProcessedEpiData':
        
        time_start = time.time()
        epipopdata = self._add_incidence_column(harmonizeddata.epidata.copy())

        if self.config.target_column != 'cases':
            epipopdata = self._drop_cases_column(epipopdata)        

        epipopdata = self._filter_dates(epipopdata)

        # extra features
        if self.config.feature_popdens:
            if harmonizeddata.population_density is None: 
                raise DataOrchestrationError('no population density found in rawdata')
            population_density_data = self._filter_years(harmonizeddata.population_density)
        else:
            population_density_data = None

        if self.config.feature_gisd:
            if harmonizeddata.gisd is None: 
                raise DataOrchestrationError('no gisd found in rawdata')
            gisd_data = self._filter_years(harmonizeddata.gisd)
        else:
            gisd_data = None            

        if self.config.feature_popage:
            if harmonizeddata.population_age is None: 
                raise DataOrchestrationError('no population age found in rawdata')
            population_age = self._filter_years(harmonizeddata.population_age)
        else:
            population_age = None                 

        processed_data = ProcessedEpiData(epidata           = epipopdata,
                                          population_density= population_density_data,
                                          gisd              = gisd_data,
                                          population_age    = population_age
                                          )
        time_end = time.time()
        if self.config.verbose > 2:
            print(f'Execution of EpiDataProcessor took {round(time_end - time_start,3)}s')    
        if self.config.verbose > 1:
            print("")

        return processed_data

# ============= FEATURE CLASS =============            
class EpiFeatureBuilder:
    def __init__(self, 
                 config: 'EpiConfig', 
                 column_registration: ColumnRegistration,
                 temporal_summary: EpiDataTemporalSummary):
        
        self.config             = config
        self.column_registration= column_registration
        self.temporal_summary   = temporal_summary

    def _add_time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        
        dfc                                 = df.copy()
        timestamps: pd.Series[pd.Timestamp] = dfc[self.config.temporal_column]
        iso_calendar                        = timestamps.dt.isocalendar()

        years           = iso_calendar['year']
        weeks           = iso_calendar['week']   
        months          = timestamps.dt.month              
        days            = iso_calendar['day']  # 1=Monday, 7=Sunday        

        sin_col_basis = f'tt_sin'
        cos_col_basis = f'tt_cos'

        # ============ day in week ===========
        if self.config.time_index_d: 
            if self.temporal_summary.temporal_frequency != 'd':
                raise DataOrchestrationError(f"can't put temporal index for day in week for data that has no daily temporal frequency")
            days_in_week = 7

            sin_col_d = sin_col_basis+"_d"
            cos_col_d = cos_col_basis+"_d"

            dfc[sin_col_d] = np.sin(2 * np.pi * days / days_in_week)
            dfc[cos_col_d] = np.cos(2 * np.pi * days / days_in_week)    

            self.column_registration.add_column(sin_col_d, 'feature', needs_normalization=False, transformation_group=None)    
            self.column_registration.add_column(cos_col_d, 'feature', needs_normalization=False, transformation_group=None)     

        # ============ week in year ===========
        if self.config.time_index_w:         
            if self.temporal_summary.temporal_frequency not in ['d','w']:
                raise DataOrchestrationError(f"can't put temporal index for week in year data that has no daily or weekly temporal frequency")

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
        if self.config.time_index_m:

            months_in_year = 12

            sin_col_m = sin_col_basis+"_m"
            cos_col_m = cos_col_basis+"_m"

            dfc[sin_col_m] = np.sin(2 * np.pi * months / months_in_year)
            dfc[cos_col_m] = np.cos(2 * np.pi * months / months_in_year)

            self.column_registration.add_column(sin_col_m, 'feature', needs_normalization=False, transformation_group=None)    
            self.column_registration.add_column(cos_col_m, 'feature', needs_normalization=False, transformation_group=None)                 

        return dfc 
    
    def _lag_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        dfc = df.copy()
        
        if self.config.lag_column != self.config.target_column:
            reference_normalization = None
        else:
            reference_normalization = 'target'

        for lag in range(0, self.config.lag_num):
            feature     = f'{self.config.lag_column}_lag{lag}'
            dfc[feature]= dfc.groupby(self.config.id_column)[self.config.lag_column].shift(lag)
            
            self.column_registration.add_column(
                feature, 
                'feature',
                needs_normalization  =True,
                transformation_group =reference_normalization
            )
            
        if self.config.verbose > 1:
            print(f'{checkmark} lags added') 

        return dfc.dropna().reset_index(drop = True)
  
    def _shift_target(self, df: pd.DataFrame) -> pd.DataFrame:
        dfc          = df.copy()
        dfc['target']= dfc.groupby(self.config.id_column)[self.config.target_column].shift(-(self.config.horizon_leadtime))
        return dfc

    def _rename_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename future target to 'target'"""
        if self.config.verbose > 1:
            print(f'{checkmark} target column renamed as such') 
        return df.rename(columns={f'{self.config.target_column}_future': 'target'})

    def _reorder_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rearrange columns in predefined order"""
        dfc          = df.copy()

        if self.config.verbose > 1:
            print(f'{checkmark} columns reordered') 

        return dfc[self.column_registration.registered_columns]

    def _add_delta_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute first difference of target column, store t-1 anchor for reversal."""
        col = self.config.target_column
        df[f'{col}_anchor'] = df.groupby(self.config.id_column)[col].shift(1)
        df[col] = df.groupby(self.config.id_column)[col].diff()

        return df.dropna().reset_index(drop=True)

    def orchestrate(self, processed_data: 'ProcessedEpiData') -> 'FeatureEpiData':
        time_start   = time.time()
        feature_data = processed_data.epidata.copy()

        # Feature: population_size
        # if self.config.feature_population_size:
        #     # raise DataOrchestrationError('currently only support the exclusion of population_size')
        #     pass
        # else:
        feature_data = feature_data.drop(columns=['population_size'], errors='ignore')
        if self.config.verbose > 1:
            print(f'{checkmark} population size removed')

        if self.config.feature_popdens:
            self.column_registration.add_column(
                'population_density',
                'feature',
                needs_normalization=True,
                transformation_group=None
            )            
            if processed_data.population_density is None:
                raise DataOrchestrationError('no population density found in processor')
            feature_data = pd.merge(feature_data, processed_data.population_density, on = [self.config.id_column, 'year'])

        if self.config.feature_gisd:
            self.column_registration.add_column(
                f'gisd',
                'feature',
                needs_normalization=False
            )               
            if processed_data.gisd is None:
                raise DataOrchestrationError('no population density found in processor')
            feature_data = pd.merge(feature_data, processed_data.gisd, on = [self.config.id_column, 'year'])           

        if self.config.feature_popage:
            processed_feature_popage = processed_data.population_age

            if processed_feature_popage is None:
                raise DataOrchestrationError('no population age found in processor')

            for cc in processed_feature_popage.columns:
                if cc not in ['year',self.config.id_column,'population_size']:
                    self.column_registration.add_column(
                        cc,
                        'feature',
                        needs_normalization=False
                    )         
                elif cc == 'population_size':
                    if self.config.feature_population_size:
                        self.column_registration.add_column(
                            cc,
                            'feature',
                            needs_normalization=True
                        )  
                    else:
                        processed_feature_popage.drop(columns = ['population_size'], inplace = True)
                                          
            feature_data = pd.merge(feature_data, processed_feature_popage, on = [self.config.id_column, 'year'])               

        # Delta transform: must happen before lags and target shift,
        # so that lag features and the forecast target are all in delta-space
        if self.config.predict_difference:
            feature_data = self._add_delta_column(feature_data)
            self.column_registration.update_transformation(
                'target',
                {'delta': {'anchor_col': f'{self.config.target_column}_anchor'}}
            )
            self.column_registration.add_column(
                f'{self.config.target_column}_anchor',
                'context',
                needs_normalization=False
            )
            if self.config.verbose > 1:
                print(f'{checkmark} delta transform applied')

        feature_data = self._add_time_index(feature_data)
        feature_data = self._lag_variable(feature_data)
        feature_data = self._shift_target(feature_data)
        feature_data = self._rename_target(feature_data)
        feature_data = self._reorder_df(feature_data)

        time_end = time.time()
        if self.config.verbose > 2:
            print(f'Execution of EpiFeatureBuilder took {round(time_end - time_start,3)}s')

        return FeatureEpiData(epidata=feature_data)

# ============= NORMALIZER CLASS ============= 
class EpiNormalizer:
    def __init__(self, config: 'EpiConfig', column_registration: ColumnRegistration, temporal_summary: EpiDataTemporalSummary):
        self.config                 = config 
        self.temporal_summary       = temporal_summary
        self.column_registration    = column_registration
        self.normalization_functions= {
            'pipeline': {'minmax': pipeline_minmax_normalization,   'zscore': pipeline_zscore_normalization},
            'apply':    {'minmax': apply_minmax_scaling,            'zscore': apply_zscore_scaling}
        }

    def _set_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create split columns using INPUT splits from temporal summary"""
        splits = self.temporal_summary.get_target_splits()
        
        df['train'] = df[self.config.temporal_column] < splits['trainval']
        df['val']   = (df[self.config.temporal_column] >= splits['trainval']) & (df[self.config.temporal_column] < splits['valtest'])
        df['test']  = df[self.config.temporal_column] >= splits['valtest']
        
        for split_col in ['train', 'val', 'test']:
            self.column_registration.add_column(split_col, 'split')
        
        if self.config.verbose > 1:
            print(f'{checkmark} split columns added (input timeline)')
        
        return df

    def _log_transform(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """log_transform columns specified"""
        df_transformed                              = df.copy()
           
        if col not in df_transformed.columns:
            raise DataOrchestrationError(f"{col} not found in df. Couldn't be log-transformed")

        df_transformed[col]                         = np.log(df_transformed[col] + self.config.log_shift)
      
        if self.config.verbose > 1:
            print(f'{checkmark} {col} logged')     

        return df_transformed

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """normalizes data and stores information in column_registration"""
        train_df        = df[df['train']]
        normalized_df   = df.copy()

        if self.config.normalization_method == 'none':
            return normalized_df

        elif self.config.normalization_method not in self.normalization_functions['apply']:
            raise DataOrchestrationError(f'No normalization function {self.config.normalization_method} found.')

        ### First pass ###
        # Calculate normalization parameters for columns that need independent normalization,
        # that is, for col-registration entries where transformation_group = None
        for col_entry in self.column_registration.columns:
            
            # Skip columns that don't have normalization attribute
            if col_entry.transformation_group == 'NA':
                continue
            
            # Only calculate params for columns with independent normalization (normalization_group is None)
            if col_entry.transformation_group is None:
                _, norm_parameters = self.normalization_functions['pipeline'][self.config.normalization_method](
                    train_df, 
                    [col_entry.column_name]
                )
                
                # Update transformation with normalization parameters
                self.column_registration.update_transformation(
                    col_entry.column_name,
                    {'normalization': norm_parameters[col_entry.column_name]}
                )

        if self.config.verbose > 1:
            print(f'{checkmark} normalization parameters retrieved and stored')     
            
        ### Second pass ###
        # Apply normalization to all columns based on parameters stored
        # or based on the reference -> for example, in the col-registry entry
        # for 'incidence_lag3' the attribute transformation_group = 'target'
        # meaning the variable is normalized using the normalization parameters
        # of 'target'
        for col_entry in self.column_registration.columns:
            
            # Skip columns that don't have normalization attribute
            if col_entry.transformation_group == 'NA':
                continue
                  
            # Determine which normalization parameters to use
            # independent transformation
            if col_entry.transformation_group is None:

                if col_entry.transformation:
                    # Use own normalization
                    params = {col_entry.column_name: col_entry.transformation['normalization']}
                    if self.config.verbose > 2:
                        print(f"{col_entry.column_name} normalized independently")

                else:
                    raise ColEntryMissingTransformationError(col_entry.column_name)
            
            # dependent transformation: expect a referral
            else:
                # Use reference column's normalization
                ref_col_entry = self.column_registration.get_by_name(col_entry.transformation_group)
                
                if ref_col_entry.transformation is None:
                    raise ColEntryMissingTransformationReferralError(col_entry.column_name, ref_col_entry.column_name)
                
                params = {col_entry.column_name: ref_col_entry.transformation['normalization']}
                if self.config.verbose > 2:
                    print(f"{col_entry.column_name} normalized based on {ref_col_entry.column_name}")
            
            # Apply normalization
            normalized_df = self.normalization_functions['apply'][self.config.normalization_method](
                normalized_df, 
                [col_entry.column_name], 
                params
            )

        
        if self.config.verbose > 1:
            print(f'{checkmark} normalization applied')     

        return normalized_df

    def _update_colregistry_postlog(self, col:str):
            self.column_registration.update_transformation(
                col, 
                {'log': self.config.log_shift}
            )           

    def orchestrate(self, feature_data: 'FeatureEpiData') -> 'NormalizedEpiData':
        time_start = time.time()
        split_data      = self._set_splits(feature_data.epidata.copy())
        cols_to_log     = []

        if self.config.log_transform:
            for col in self.config.log_transform:

                if col == self.config.target_column:
                    cols_to_log += ['target']
                    self._update_colregistry_postlog('target')

                if col == self.config.lag_column:
                    cols_to_log += [f'{self.config.lag_column}_lag{lag}' for lag in range(0, self.config.lag_num)]
                    if self.column_registration.get_by_name(f'{self.config.lag_column}_lag0').transformation_group is None:
                        self._update_colregistry_postlog(f'{self.config.lag_column}_lag0')

                else:
                    cols_to_log += [col]
                    self._update_colregistry_postlog(col)

        for col in cols_to_log:
            split_data = self._log_transform(split_data, col)
        
        normalized_data = self._normalize(split_data)

        time_end = time.time()
        if self.config.verbose > 2:
            print(f'Execution of EpiNormalizer took {round(time_end - time_start,3)}s')        
        if self.config.verbose > 1:
            print("")
        return NormalizedEpiData(epidata=normalized_data)

# ============= Finalize CLASS ============= 
class EpiDataFinalizer:
 
    def __init__(self, config: 'EpiConfig', column_registration: ColumnRegistration):
        self.config = config 
        self.column_registration = column_registration

    def _add_horizons(self, df: pd.DataFrame) -> pd.DataFrame:
        base_lead = self.config.horizon_leadtime
        for additional_steps in range(1, self.config.horizon_size):
            steps_ahead = base_lead + additional_steps
            target_col = f'target_lead{steps_ahead}'
            
            # Shift from the base target
            df[target_col] = df.groupby(self.config.id_column)[f'target_lead{base_lead}'].shift(-additional_steps)
            
            # Register in column registry
            self.column_registration.add_column(
                target_col,
                'target',
                transformation_group='target',
                needs_normalization=True
            )
        
        if self.config.verbose > 1:
            print(f'{checkmark} targets for all horizons added')              
        return df

    def _drop_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.verbose > 1:
            print(f"{checkmark} nans dropped")
        return df.dropna()

    def _set_targettype_integer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return df with columns containing 'target' as int
        Required when predicting casenumbers instead of incidence rates
        """
        for col in df.columns:
            if 'target' in col and 'timestamp' not in col:
                df[col] = df[col].astype(int)
       
        if self.config.verbose > 1:
            print(f"{checkmark} target columns set as integer")

        return df

    def _set_targettype_class(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return df with columns containing 'target' as binary
        Required when target is casenumber and prediction_mode is classification
        """
        for col in df.columns:
            if 'target' in col and 'timestamp' not in col:
                df.loc[df[col] > 0, col] = 1
       
        if self.config.verbose > 1:
            print(f"{checkmark} target columns set as class")

        return df

    # TODO: CLEAN UP
    def _denormalize(self, normalized_df: pd.DataFrame) -> pd.DataFrame:
            dfc = normalized_df.copy()
            
            # Get normalization method
            norm_method = self.config.normalization_method
            
            if norm_method == 'none':
                return dfc
            
            # Reverse transformations for each column
            for col_entry in self.column_registration.columns:
                if col_entry.column_name not in dfc.columns:
                    continue
                    
                # Skip context columns
                if col_entry.transformation_group == 'NA':
                    continue
                
                # Get transformation parameters
                if col_entry.transformation_group is None:
                    # Independent normalization
                    if col_entry.transformation:
                        params = col_entry.transformation
                    else:
                        continue           
                
                else:
                    # Use reference column's parameters
                    ref_entry = self.column_registration.get_by_name(col_entry.transformation_group)
                    if ref_entry.transformation:
                        params = ref_entry.transformation
                    else:
                        continue
                
                # Reverse normalization
                if 'normalization' in params:
                    if norm_method == 'minmax':
                        dfc = reverse_minmax_scaling(dfc, params['normalization'], column=col_entry.column_name)
                    elif norm_method == 'zscore':
                        dfc = reverse_zscore_scaling(dfc, params['normalization'], column=col_entry.column_name)
                
                # Reverse log transform
                if 'log' in params:
                    dfc = reverse_log(dfc, params['log'], column=col_entry.column_name)

            # now to target
            target_columns = [f'target_lead{steps_ahead+self.config.horizon_leadtime}' for steps_ahead in range(self.config.horizon_size)]   
            params          = self.column_registration.get_by_name('target').transformation      

            for colname in target_columns:
                # Reverse normalization
                if 'normalization' in params:
                    if norm_method == 'minmax':
                        dfc = reverse_minmax_scaling(dfc, params['normalization'], column=colname)
                    elif norm_method == 'zscore':
                        dfc = reverse_zscore_scaling(dfc, params['normalization'], column=colname)
                
                # Reverse log transform
                if 'log' in params:
                    dfc = reverse_log(dfc, params['log'], column=colname)                 


            return dfc

    def orchestrate(self, normalized_data: NormalizedEpiData) -> 'FinalizedEpiData':
        time_start = time.time()
        dfc         = normalized_data.epidata
        base_lead   = self.config.horizon_leadtime
        dfc         = dfc.rename(columns={'target': f'target_lead{base_lead}'})

        # Create additional horizons if needed
        if self.config.horizon_size > 1:
            dfc = self._add_horizons(dfc)
            
        # For predictions (will be generated during training)
        self.column_registration.add_column(
            'pred',
            'pred',
            transformation_group='target',
            needs_normalization=True
        )      

        dfc_nanfree         = self._drop_nans(dfc)
        dfc_denormalized    = self._denormalize(dfc_nanfree)


        # If target == 'cases' => target columns should be integers
        if self.config.target_column == 'cases':
            if self.config.prediction_mode == 'regression':
                dfc_nanfree = self._set_targettype_integer(dfc_nanfree)
            elif self.config.prediction_mode == 'classification':
                dfc_nanfree = self._set_targettype_class(dfc_nanfree)                
        time_end = time.time()
        if self.config.verbose > 2:
            print(f'Execution of EpiFinalizer took {round(time_end - time_start,3)}s')  
        if self.config.verbose:
            print(f'{checkmark}{checkmark} All data finalized with correct timestamp alignment') 
        if self.config.verbose > 1:
            print("")


        return FinalizedEpiData(
            data        = dfc_nanfree,
            data_denorm = dfc_denormalized,
        )