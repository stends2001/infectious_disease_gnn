import pandas as pd
import geopandas as gpd
from typing import Tuple, Dict, Union, Literal
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ...utils.textformatting import warning_emoji, checkmark
from ...utils.constants import berlin_district_ids, berlin_id
from .normalization import apply_minmax_scaling, apply_zscore_scaling, pipeline_minmax_normalization, pipeline_zscore_normalization
from .column_registry import ColumnRegistration, ColEntryMissingError, ColEntryMissingTransformationError, ColEntryMissingTransformationReferralError
from .epiconfig import EpiConfig
from .datastagecontainers import RawEpiData, ContextData, HarmonizedData, ProcessedEpiData, FeatureEpiData, NormalizedEpiData, FinalizedEpiData, ProcessedEpiData

# ============= READER CLASS =============
class EpiDataReader:
    """
    Only dataloading
    Separate functions with which datafiles are read and stored 

    Parameters
    ----------
    config: 'EpiConfig'

    Examples
    --------
    >>> rawdata = EpiDataReader(config).orchestrate()

    Returns
    -------
    .orchestrate() --> RawEpiData

    Downstream
    ----------
    output 'RawEpiData' is input for NUTSHarmonizer.orchestrate()
    """
    
    def __init__(self, config: 'EpiConfig'):
        self.config = config
    
    def _load_disease_data(self) -> pd.DataFrame:
        """
        Load raw disease case data downloaded from SurvStat.
        
        Returns
        --------
        pd.DataFrame with columns: [timestamp, nuts3, cases, week, year]
        """
        filepath = self.config.get_disease_path()
                
        df = pd.read_csv(
            filepath,
            parse_dates=['timestamp'],
            dtype={'kz_kreis': str, 'cases':int}
        ).rename(columns={'kz_kreis': 'nuts3_key'})
        
        if self.config.verbose > 1:
            print(f"{checkmark} Loaded raw disease data")
        
        return df
    
    def _load_population_data(self) -> pd.DataFrame:
        """
        Load population size data
        
        Returns
        --------
        pd.DataFrame with columns: [nuts3, year, population_size]
        """
        filepath = self.config.get_population_path()
        
        df = pd.read_csv(
            filepath,
            dtype={'nuts3': str}
        ).rename(columns = {'nuts3':'nuts3_key'})
        
        if self.config.verbose > 1:
            print(f"{checkmark} Loaded raw population data")
        
        return df
    
    def _load_population_data_berlin_districts(self) -> pd.DataFrame:
        """
        Load population size data for districts in Berlin
        
        Returns
        --------
        pd.DataFrame with columns: [nuts3, population_size]
        """        
        filepath = self.config.get_population_berlin_districts_path()
        
        df = pd.read_csv(
            filepath,
            dtype={'nuts3': str}).rename(columns = {'nuts3':'nuts3_key'})
        
        if self.config.verbose > 1:
            print(f"{checkmark} Loaded raw berlin districts population data")
        
        return df        

    def _load_shapedata(self) -> gpd.GeoDataFrame:
        """
        Load geographic shapefile
        
        Returns
        --------
        GeoDataFrame with geometry and nuts code
        """
        filepath = self.config.get_shapefile_path()
        
        gdf = gpd.read_file(filepath).drop(columns=['level'], errors='ignore')
        
        if self.config.verbose > 1:
            print(f"{checkmark} Loaded raw shapedata ({len(gdf)} regions)")
        
        return gdf
    
    def _load_nuts_names(self) -> pd.DataFrame:
        """
        ...
        """
        # main file, additions-file
        filepath = self.config.get_nuts_names_path()   
        
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        
        if self.config.verbose > 1:
            print(f"{checkmark} Loaded raw NUTS names")
        
        return df
    
    def _load_gisd_data(self) -> pd.DataFrame:
        """
        loads and returns gisd data for set nuts level
        """

        if self.config.nuts_level == 'nuts1':
            raise ValueError('currently no gisd data for nuts1 exists')
        if self.config.nuts_level == 'nuts3' and self.config.split_berlin:
            raise ValueError('no gisd data for berlin districts exists. please remove gisd data or merge berlin')

        # main file, additions-file
        filepath = self.config.get_gisd_path()   
        
        df = pd.read_csv(filepath,sep="," , dtype={f'{self.config.nuts_level}_key':str})
        
        if self.config.verbose > 1:
            print(f"{checkmark} Loaded raw gisd data")
        
        return df

    def orchestrate(self) -> RawEpiData:
        """
        Load all required data files
        
        Returns
        --------
        RawEpiData : Container with all raw dataframes
        """
        rawdata = RawEpiData(
            disease             = self._load_disease_data(),
            population          = self._load_population_data(),
            shapedata           = self._load_shapedata(),
            nuts_names          = self._load_nuts_names(),
            population_berlin   = self._load_population_data_berlin_districts() if self.config.split_berlin else None,
            gisd                = self._load_gisd_data() if self.config.include_gisd else None,            
        )

        if self.config.verbose:
            print(f'{checkmark}{checkmark} All raw data loaded')
        if self.config.verbose > 1:
            print("")

        return rawdata
    
# ============= HARMONIZATION CLASS =============
class NUTSHarmonizer:
    """
    Preprocesses raw data into harminized NUTS - regions

    Parameters
    ----------
    config: 'EpiConfig'

    Examples
    --------
    >>> harmeddata = NUTSHarmonizer(config).orchestrate(rawdata)

    Returns
    -------
    .orchestrate() --> Tuple['HarmonizedData', 'ContextData']

    Downstream
    ----------
    output 'HarmonizedData' is input for EpiDataProcessor.orchestrate()    
    output 'ContextData' is stored in orchestrator object for later use of inferring node-information
    """    
    def __init__(self, config: 'EpiConfig'):
        self.config = config    

    def _add_berlin_districts(self, population_nuts3: pd.DataFrame, population_berlin_districts: pd.DataFrame) -> pd.DataFrame:
        """
        when berlin to be split -> add population data by district
        based on population data for nuts3 and for berlin districts, concatenate into one df with 412 nuts3 values.
        """
        total_population_berlin                         = sum(population_berlin_districts['population_size'])
        relative_population_size_berlin_districts       = population_berlin_districts.copy()
        relative_population_size_berlin_districts['population_size']= relative_population_size_berlin_districts['population_size'] / total_population_berlin

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
        epidemiology_df_aggr = epidemiology_df.groupby(['timestamp', f'{self.config.nuts_level}_key']).aggregate({'cases':'sum'}).reset_index()     
        population_df_aggr   = population_df.groupby(['year', f'{self.config.nuts_level}_key']).aggregate({'population_size':'sum'}).reset_index() 

        if self.config.verbose > 1:
            print(f'{checkmark} epidemiology and population data aggregated on nuts')          
        return epidemiology_df_aggr, population_df_aggr

    def _add_nuts_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """adds nuts-level column"""
        if self.config.nuts_level == "nuts1":
            df['nuts1']= df['nuts3'].str[:2]     

        elif self.config.nuts_level =='nuts2':
            df['nuts2']= df['nuts3'].str[:3]

        if self.config.verbose > 1:
            print(f'{checkmark} nuts column added')  
        return df

    def _merge_epipopdata(self, epidemiology_df: pd.DataFrame, population_df: pd.DataFrame) -> pd.DataFrame:
        """
        merge epidemiology with population data, by extracting the year from epidemiology data. 
        The year column is dropped
        """
        if 'year' not in epidemiology_df.columns:
            epidemiology_df['year'] = epidemiology_df['timestamp'].dt.year 

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

    def _get_tokenization_map(self, df: pd.DataFrame) -> Dict[str,Dict[str, str]]:
        """df should be epipopdata"""
        unique_ids = sorted(df[f'{self.config.nuts_level}_key'].unique())
        id_idx     = {} # nuts  : int
        idx_id     = {} # int   : nuts      

        for idx, id in enumerate(unique_ids):
            id_idx[id] = idx                  # id (nuts1, nuts2 or nuts3) : node_id (zero based)
            idx_id[idx] = str(id)             # node_id (zero based): id (nuts1, nuts2 or nuts3)          

        tokenization_map = {"id_idx": id_idx, "idx_id": idx_id} 
        if self.config.verbose > 1:
            print(f'{checkmark} tokenization_map developed')    
        return tokenization_map

    def _tokenize_df(self, df: pd.DataFrame, tokenization_map, drop_key: bool = True) -> pd.DataFrame:
        df['node'] = df[f'{self.config.nuts_level}_key']
        df['node'] = df[f'{self.config.nuts_level}_key'].map(tokenization_map['id_idx'])  

        # get nuts-levels for rows to be dropped
        rows_to_drop        = df[df['node'].isna()]
        dropped_nuts_values = rows_to_drop[f'{self.config.nuts_level}_key'].unique()
                
        # drop rows
        dfr         = df.copy().dropna(subset=['node'])
        dfr['node'] = dfr['node'].astype(int)

        # Drop the nuts_key column after tokenization - it's now redundant
        nuts_key_col = f'{self.config.nuts_level}_key'
        if drop_key:
            if nuts_key_col in dfr.columns:
                dfr = dfr.drop(columns=[nuts_key_col])        

        number_dropped_rows = len(rows_to_drop)

        if number_dropped_rows > 0:
            
            # Define exception conditions for dropped rows. 
            # If nodes are droppped and neither of these conditions 
            # are satisfied, then there's an issue.
            exception_conditions = [
                # only Berlin city is droppped
                (number_dropped_rows == 1 and '11000' in dropped_nuts_values and self.config.split_berlin),
                # only Berlin districts are dropped
                (number_dropped_rows == len(berlin_district_ids) and set(berlin_district_ids).issubset(dropped_nuts_values) and not self.config.split_berlin)
            ]
            
            if not any(exception_conditions):
                print(f'{warning_emoji}dropping non-tokenized nodes in {df.head()}: dropped {number_dropped_rows} nodes: {rows_to_drop}')    
        return dfr  

    def _prepare_gisd(self, raw_gisd: pd.DataFrame, tokenization_map: Dict) -> pd.DataFrame:
        """
        Tokenize GISD data and prepare for merging.
        GISD has columns: nuts{X}_key, year, and various GISD indicators
        """
        gisd_tokenized = self._tokenize_df(raw_gisd, tokenization_map)
        
        if self.config.verbose > 1:
            print(f'{checkmark} GISD data tokenized')
        
        return gisd_tokenized

    # def _tokenize_data(self, dfs: Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]]) -> Tuple[Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]], Dict[str, Dict[Union[int,str],Union[int,str]]]]:
    #     """
    #     Tokenize nuts-levels into 'node' column.
    #     """
    #     unique_ids = sorted(dfs['epipopdata'][f'{self.config.nuts_level}_key'].unique())
    #     id_idx     = {} # nuts  : int
    #     idx_id     = {} # int   : nuts

    #     for idx, id in enumerate(unique_ids):
    #         id_idx[id] = idx                  # id (nuts1, nuts2 or nuts3) : node_id (zero based)
    #         idx_id[idx] = str(id)             # node_id (zero based): id (nuts1, nuts2 or nuts3)

    #     tokenization_map = {"id_idx": id_idx, "idx_id": idx_id} 
    #     df_dict = {}

    #     for dfname, df in dfs.items():
    #         dfc         = df.copy()
    #         print(dfname)
    #         dfc['node'] = dfc[f'{self.config.nuts_level}_key']
    #         dfc['node'] = dfc[f'{self.config.nuts_level}_key'].map(id_idx)  

    #         # get nuts-levels for rows to be dropped
    #         rows_to_drop        = dfc[dfc['node'].isna()]
    #         dropped_nuts_values = rows_to_drop[f'{self.config.nuts_level}_key'].unique()
                    
    #         # drop rows
    #         dfr         = dfc.copy().dropna(subset=['node'])
    #         dfr['node'] = dfr['node'].astype(int)

    #         number_dropped_rows = len(rows_to_drop)

    #         if number_dropped_rows > 0:
                
    #             # Define exception conditions for dropped rows. 
    #             # If nodes are droppped and neither of these conditions 
    #             # are satisfied, then there's an issue.
    #             exception_conditions = [
    #                 # only Berlin city is droppped
    #                 (number_dropped_rows == 1 and '11000' in dropped_nuts_values and self.config.split_berlin),
    #                 # only Berlin districts are dropped
    #                 (number_dropped_rows == len(berlin_district_ids) and set(berlin_district_ids).issubset(dropped_nuts_values) and not self.config.split_berlin)
    #             ]
                
    #             if not any(exception_conditions):
    #                 print(f'{warning_emoji}dropping non-tokenized nodes in {dfname}: dropped {number_dropped_rows} nodes')

    #         df_dict[dfname] = dfr
     
    #     if self.config.verbose > 1:
    #         print(f'{checkmark} nodes tokenized')       
    #     return df_dict, tokenization_map       

    def orchestrate(self, rawdata: 'RawEpiData') -> Tuple['HarmonizedData', 'ContextData']:
        """
        The function that orchestrates all others
        """
        if self.config.split_berlin:
            if rawdata.population_berlin is None:
                raise AttributeError("'population_berlin' attribute is not found in rawdata")
            
            population_data = self._add_berlin_districts(rawdata.population, rawdata.population_berlin)
            raw_epidata     = rawdata.disease
        else:
            raw_epidata     = self._mutate_berlin_district_ids(rawdata.disease)
            population_data = rawdata.population

        epidemiology_data   = self._add_nuts_column(raw_epidata)
        population_data     = self._add_nuts_column(population_data)
        aggregated_dfs      = self._aggregate_by_nuts(epidemiology_data, population_data)
        epipopdata          = self._merge_epipopdata(aggregated_dfs[0], aggregated_dfs[1])
        nutsnames      = self._get_nuts_data(rawdata.nuts_names)

        tokenization_map    = self._get_tokenization_map(epipopdata)

        epipopdata          = self._tokenize_df(epipopdata, tokenization_map)
        shapedata           = self._tokenize_df(rawdata.shapedata, tokenization_map)
        nutsnames           = self._tokenize_df(nutsnames, tokenization_map, drop_key = False)     
        
        # Prepare GISD if provided
        gisd_harmonized = None
        if rawdata.gisd is not None:
            gisd_harmonized = self._prepare_gisd(rawdata.gisd, tokenization_map)

        harmdata = HarmonizedData(
            data = epipopdata,
            gisd = gisd_harmonized  # Add this
        )
        
        ctxdata = ContextData(
            nuts_level          = self.config.nuts_level,
            num_nodes           = len(tokenization_map['id_idx']),
            shapedata           = shapedata,
            nuts_names          = nutsnames,
            tokenization_map    = tokenization_map            
        )

        if self.config.verbose:
            print(f'{checkmark}{checkmark} All raw data nuts-harmonized: harmonized and context data generated')       
        if self.config.verbose > 1:
            print("")              

        return harmdata, ctxdata

# ============= PREPROCESSING CLASS =============
class EpiDataProcessor:
    """ 
    Processing raw data: adding and removing columns,

    Parameters
    ----------
    config: 'EpiConfig'

    Examples
    --------
    >>> processeddata = EpiDataProcessor(config).orchestrate(harmeddata)

    Returns
    -------
    .orchestrate() --> 'ProcessedEpiData'

    Downstream
    ----------
    output 'ProcessedEpiData' is input for EpiFeatureBuilder.orchestrate()        
    """        
    def __init__(self, config: 'EpiConfig'):
        self.config = config

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
       
    def _drop_cols(self, epipopdata: pd.DataFrame) -> pd.DataFrame:
        """drop redundant columns"""
        cols_to_drop = []
        
        # Only need to drop year now (nuts_key already dropped during tokenization)
        if 'year' in epipopdata.columns:
            cols_to_drop.append('year')
        
        if self.config.verbose > 1:
            dropped_msg = f'{checkmark} year column removed' if cols_to_drop else f'{checkmark} no redundant columns to remove'
            print(dropped_msg)
        
        return epipopdata.drop(columns=cols_to_drop, errors='ignore')

    def _filter_mindate(self, df, min_date: pd.Timestamp) -> pd.DataFrame:
        if self.config.verbose:
            print(f"{checkmark} min_date has been extended to {self.config.min_date_extended.date()}")  

        if self.config.verbose > 1:
            print(f'{checkmark} filtered on min date')            
        
        return df.loc[df['timestamp'] >= min_date].reset_index(drop = True)                

    def _filter_maxdate(self, df, max_date: pd.Timestamp) -> pd.DataFrame:

        if self.config.verbose > 1:
            print(f'{checkmark} filtered on max date')  

        return df.loc[df['timestamp'] < max_date].reset_index(drop = True)       

    def _merge_gisd(self, epipopdata: pd.DataFrame, gisd: pd.DataFrame) -> pd.DataFrame:
        """
        Merge GISD data with epidemiology data by node and year.
        
        Parameters
        ----------
        epipopdata : pd.DataFrame
            Main data with columns including 'node' and 'timestamp'
        gisd : pd.DataFrame
            GISD data with columns including 'node', 'year', and GISD indicators
        
        Returns
        -------
        pd.DataFrame
            Merged dataframe with GISD features
        """
        # Extract year from timestamp if not already present
        if 'year' not in epipopdata.columns:
            epipopdata = epipopdata.copy()
            epipopdata['year'] = epipopdata[self.config.temporal_column].dt.year
        
        # Merge on node and year
        merged = pd.merge(
            epipopdata,
            gisd,
            on=['node', 'year'],
            how='left',
            suffixes=('', '_gisd')
        )
        
        # Drop the year column if it was added temporarily
        if 'year' in merged.columns:
            merged = merged.drop(columns=['year'])
        
        if self.config.verbose > 1:
            print(f'{checkmark} GISD data merged')
        
        return merged

    def orchestrate(self, harmonizeddata: 'HarmonizedData') -> 'ProcessedEpiData':
        """
        The function that orchestrates all others
        """
        epipopdata = self._add_incidence_column(harmonizeddata.data)

        if self.config.target_column != 'cases':
            epipopdata = self._drop_cases_column(epipopdata)        

        epipopdata = self._filter_maxdate(epipopdata, self.config.max_date)
        epipopdata = self._filter_mindate(epipopdata, self.config.min_date_extended)
        
        # Merge GISD data BEFORE dropping year column (it needs year for merging)
        if harmonizeddata.gisd is not None:
            # Add year temporarily if not present
            if 'year' not in epipopdata.columns:
                epipopdata['year'] = epipopdata[self.config.temporal_column].dt.year
            
            epipopdata = self._merge_gisd(epipopdata, harmonizeddata.gisd)
        
        # Now drop redundant columns (including year and any nuts_key columns)
        epipopdata = self._drop_cols(epipopdata)

        if self.config.verbose:
            print(f'{checkmark}{checkmark} All data preprocessed')   
        if self.config.verbose > 1:
            print("")

        return ProcessedEpiData(data=epipopdata)

# ============= FEATURE CLASS =============            
class EpiFeatureBuilder:
    """ 
    Feature building, separate per function

    Parameters
    ----------
    config: 'EpiConfig'
    column_registration: 'ColumnRegistration'

    Examples
    --------
    >>> featuredata = EpiFeatureBuilder(config).orchestrate(processeddata)

    Returns
    -------
    .orchestrate() --> 'FeatureEpiData'

    Downstream
    ----------
    output 'FeatureEpiData' is input for EpiNormalizer.orchestrate()            
    """        
    def __init__(self, config: 'EpiConfig', column_registration: ColumnRegistration):
        self.config             = config
        self.column_registration= column_registration

    def _time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """add time_index feature (sin/cos for weekly values)"""
        dfc = df.copy()
        # Extract year and ISO week number
        iso_calendar    = dfc[self.config.temporal_column].dt.isocalendar()
        years           = iso_calendar['year']
        weeks           = iso_calendar['week']        

        # Function to check if a year has 53 weeks
        def has_53_weeks(year: int) -> bool:
            dec_28 = pd.Timestamp(year=year, month=12, day=28)
            return dec_28.isocalendar()[1] == 53
        
        # Cache years and their week counts (52 or 53)
        unique_years    = years.unique()
        year_week_count = {year: 53 if has_53_weeks(year) else 52 for year in unique_years}
        # Map each year in df to its week count
        weeks_in_year   = years.map(year_week_count)        

        # Column names
        sin_col = f'{self.config.temporal_column}_sin'
        cos_col = f'{self.config.temporal_column}_cos'

        # Compute sine and cosine transformation to encode cyclical nature of weeks in a year
        dfc[sin_col] = np.sin(2 * np.pi * weeks / weeks_in_year)
        dfc[cos_col] = np.cos(2 * np.pi * weeks / weeks_in_year)

        if self.config.verbose > 1:
            print(f'{checkmark} time index {sin_col}, {cos_col} included as feature')                 
        return dfc

    def _lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged features and future target.
        
        Timeline:
        - lag_column_t0: current week (timestamp)
        - lag_column_t_1: 1 week ago
        - lag_column_t_2: 2 weeks ago
        - target: horizon_leadtime weeks ahead
        """
        dfc = df.copy()
        
        # Step 1: Explicitly create t0 for the lag column (current value, unshifted)
        dfc[f'{self.config.lag_column}_t0'] = dfc[self.config.lag_column]
        
        # Step 2: Create lagged features (t_1, t_2, etc.)
        for lag in range(1, self.config.lag_num):
            feature = f'{self.config.lag_column}_t_{lag}'
            dfc[feature] = dfc.groupby(self.config.id_column)[self.config.lag_column].shift(lag)
            
            self.column_registration.add_column(
                feature, 
                'feature',
                needs_normalization=True,
                transformation_group=f'{self.config.lag_column}_t0' if self.config.lag_column != self.config.target_column else 'target'
            )
        
        # Step 3: Shift target into the future (only if it's a separate column or needs shifting)
        if self.config.horizon_leadtime > 0:
            dfc[f'{self.config.target_column}_future'] = dfc.groupby(
                self.config.id_column
            )[self.config.target_column].shift(-self.config.horizon_leadtime)
        else:
            dfc[f'{self.config.target_column}_future'] = dfc[self.config.target_column]
        
        # Step 4: Drop the original unshifted columns (we have t0 and future versions now)
        dfc = dfc.drop(columns=[self.config.lag_column, self.config.target_column])
        
        # Drop rows with NaN from shifting
        dfc = dfc.dropna().reset_index(drop=True)
    
        if self.config.verbose > 1:
            print(f'{checkmark} lags added') 

        return dfc

    def _rename_lagt0(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        identical df with column name {self.config.lag_column} to {self.config.lag_column}_t0
        """
        old_featurename = self.config.lag_column
        new_featurename = f'{self.config.lag_column}_t0'

        renamed_df = df.rename(columns = {old_featurename: new_featurename})   

        return renamed_df
    
    def _rename_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename future target to 'target'"""
        if self.config.verbose > 1:
            print(f'{checkmark} target column renamed as such') 
        return df.rename(columns={f'{self.config.target_column}_future': 'target'})

    def _reorder_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """simply rearranges columns in a predefined order"""
        dfc = df.copy()
        first_cols      = ['timestamp','node','target']
        feature_cols    = [col for col in df.columns if col not in first_cols]

        if self.config.verbose > 1:
            print(f'{checkmark} columns reordered') 

        return dfc[first_cols + feature_cols]

    def _update_colregistration_postfeaturebuilding(self):
        """update column registry based on config"""
        # if lag col == target col, then the lag will be normalized following identical parameters as the target col
        lag0_featurename = f'{self.config.lag_column}_t0'

        # TARGET
        #       if target == 'incidence' (!='cases')    -> normalization needed (transformation => individual)
        #       if target == 'cases'                    -> no normalization needed
        if self.config.target_column!='cases':
            self.column_registration.add_column(
                    'target', 
                    'target',
                    needs_normalization=True,
                    transformation_group=None
                )    
        else:
            self.column_registration.add_column(
                    'target', 
                    'target',
                    needs_normalization=False,
                    transformation_group='NA'
                )                
        
        # LAG
        #       if lag_column == target_column          -> normalization parameters from TARGET are used
        #       else                                    -> individual transformation
        if self.config.lag_column == self.config.target_column:            
            self.column_registration.add_column(
                    lag0_featurename, 
                    'feature',
                    needs_normalization=True,
                    transformation_group='target'
                )      
        else:
            self.column_registration.add_column(
                    lag0_featurename, 
                    'feature',
                    needs_normalization=True,
                    transformation_group=None
                )      

        # POPULATION_SIZE
        if self.config.include_population:
            self.column_registration.add_column(
                'population_size', 
                'feature', 
                needs_normalization=True,
                transformation_group=None
            )          

        # TIME INDEX
        if self.config.time_index:
            sin_col = f'{self.config.temporal_column}_sin'
            cos_col = f'{self.config.temporal_column}_cos'

            self.column_registration.add_column(
                sin_col, 
                'feature', 
                needs_normalization=True,
                transformation_group=None  # Independent normalization
            )    

            self.column_registration.add_column(
                cos_col, 
                'feature', 
                needs_normalization=True,
                transformation_group=None  # Independent normalization
            )              
        
        # GISD FEATURES
        if self.config.include_gisd:
            self.column_registration.add_column(
                    'gisd',
                    'feature',
                    needs_normalization=False,
                    transformation_group=None  # Independent normalization for each GISD feature
                )
            
            if self.config.verbose > 1:
                print(f'{checkmark} GISD features registered in column registry')

        if self.config.verbose > 1:
            print(f'{checkmark} column_registry updated')  

    def orchestrate(self, processed_data: 'ProcessedEpiData') -> 'FeatureEpiData':
        """orchestrates entire feature addition"""
        feature_data = processed_data.data
        self.data = feature_data  # Store reference for GISD column detection

        # Feature: population_size => colregistry done in update the registry function
        if self.config.include_population:
            if self.config.verbose > 1:
                print(f'{checkmark} population size included as feature')    
        else:
            # remove column population_size
            feature_data.drop(columns = ['population_size'], inplace = True) 
            if self.config.verbose > 1:
                print(f'{checkmark} population size removed')    
        
        # Feature: time_index (timestamp_sin, timestamp_cos)        
        if self.config.time_index:
            feature_data = self._time_index(feature_data)
        
        feature_data = self._lags(feature_data)
        feature_data = self._rename_lagt0(feature_data)
        feature_data = self._rename_target(feature_data)   
        feature_data = self._reorder_df(feature_data) 

        self._update_colregistration_postfeaturebuilding()

        if self.config.verbose:
            print(f'{checkmark}{checkmark} All features built') 
        if self.config.verbose > 1:
            print("")

        return FeatureEpiData(data=feature_data)

# ============= Normalize CLASS ============= 
class EpiNormalizer:
    """ 
    Set splits (train/val/test columns) and transforms variables (normalization and log-transforms)

    Parameters
    ----------
    config: 'EpiConfig'
    column_registration: 'ColumnRegistration'

    Examples
    --------
    >>> normalizeddata = EpiNormalizer(config).orchestrate(featuredata)

    Returns
    -------
    .orchestrate() --> 'NormalizedEpiData'

    Downstream
    ----------
    output 'NormalizedEpiData' is input for EpiDataFinalizer.orchestrate()   

    Note:
    ----
    TODO I definitely need to add test function to validate the normalization and the absence of data leakage!   

    See Also
    --------
    # normalization functions:
    #   - pipeline_minmax_normalization (returns parameters for normalization -> input is TRAINING DATA ONLY)
    #   - apply_minmax_scaling          (applies same normalization onto train/val/test)
    #   - pipeline_zscore_normalization (returns parameters for normaliztation -> input is TRAINING DATA ONLY)
    #   - apply_zscore_scaling          (applies same normalization onto train/val/test)
    """        

    def __init__(self, config: 'EpiConfig', column_registration: ColumnRegistration):
        self.config                 = config 
        self.column_registration    = column_registration
        self.normalization_functions= {
            'pipeline': {'minmax': pipeline_minmax_normalization,   'zscore': pipeline_zscore_normalization},
            'apply':    {'minmax': apply_minmax_scaling,            'zscore': apply_zscore_scaling}
        }

    def _set_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """create split columns"""
        df['train'] = df[self.config.temporal_column] < self.config.split_trainval_ts
        df['val']   = (df[self.config.temporal_column] >= self.config.split_trainval_ts) & (df[self.config.temporal_column] < self.config.split_valtest_ts)
        df['test']  = df[self.config.temporal_column] >= self.config.split_valtest_ts
        
        # Add split columns to registry
        for split_col in ['train', 'val', 'test']:
            self.column_registration.add_column(split_col, 'split')

        if self.config.verbose > 1:
            print(f'{checkmark} split columns (train/val/test) added')             
        
        return df

    def _log_transform(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """log_transform columns specified"""
        df_transformed                              = df.copy()
           
        if col not in df_transformed.columns:
            raise ValueError(f'{col} not found in df')

        df_transformed[col]                         = np.log(df_transformed[col] + self.config.log_shift)
      
        if self.config.verbose > 1:
            print(f'{checkmark} {col} logged')     

        return df_transformed

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """normalizes data and stores information in column_registration"""
        train_df        = df[df['train']]
        normalized_df   = df.copy()

        if self.config.normalization_method not in self.normalization_functions['apply']:
            raise KeyError(f'No normalization function {self.config.normalization_method} found.')

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
        """runs all normalization functions"""
        split_data      = self._set_splits(feature_data.data)
        cols_to_log     = []

        if self.config.log_transform:
            for col in self.config.log_transform:

                if col == self.config.target_column:
                    cols_to_log += ['target']
                    self._update_colregistry_postlog('target')

                if col == self.config.lag_column:
                    cols_to_log += [f'{self.config.lag_column}_t0'] + [f'{self.config.lag_column}_t_{lag}' for lag in range(1, self.config.lag_num)]
                    if self.column_registration.get_by_name(f'{self.config.lag_column}_t0').transformation_group is None:
                        self._update_colregistry_postlog(f'{self.config.lag_column}_t0')

                else:
                    cols_to_log += [col]
                    self._update_colregistry_postlog(col)

        for col in cols_to_log:
            split_data = self._log_transform(split_data, col)
        
        normalized_data = self._normalize(split_data)

        if self.config.verbose:
            print(f'{checkmark}{checkmark} All data normalized')         
        if self.config.verbose > 1:
            print("")
        return NormalizedEpiData(data=normalized_data)

# ============= Finalize CLASS ============= 
class EpiDataFinalizer:
    """ 
    Finalizes the data into a shared df that may be imported by model-specific DataLoader objects.

    Parameters
    ----------
    config: 'EpiConfig'
    column_registration: 'ColumnRegistration'

    Examples
    --------
    >>> finaldata = EpiDataFinalizer(config).orchestrate(normalizeddata)

    Returns
    -------
    .orchestrate() --> 'FinalizedEpiData'

    Downstream
    ----------
    output 'FinalizedEpiData' is input for dataloaders (ShallowDataLoaderManager, GraphDataLoaderManager)
    """  
    def __init__(self, config: 'EpiConfig', column_registration: ColumnRegistration):
        self.config             = config 
        self.column_registration= column_registration

    def _add_horizons(self, df: pd.DataFrame) -> pd.DataFrame:
        """loop over horizons, add shifted target under new name f'target_lead{steps_ahead}' where steps_ahead = horizon_leadtime + horizon_idx"""
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
                transformation_group='target',              # Use same normalization as base target
                needs_normalization=True
            )
        if self.config.verbose > 1:
            print(f'{checkmark} targets for all horizons added')              
        return df

    def _drop_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.verbose > 1:
            print(f"{checkmark} nans droppped")
        return df.dropna()

    def _set_targettype_integer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        return df with columns of which name includes 'target' as int
        Required when predicting casenumbers instead of incidence rates
        """
        for col in df.columns:
            if 'target' in col:
                df[col] = df[col].astype(int)
       
        if self.config.verbose > 1:
            print(f"{checkmark} target columns set as integer")

        return df

    def orchestrate(self, normalized_data: NormalizedEpiData) -> 'FinalizedEpiData':
        """runs entire finalization"""
        dfc = normalized_data.data

        # 'target' is already at t+horizon_leadtime
        # Rename it to reflect its actual position
        base_lead = self.config.horizon_leadtime
        dfc = dfc.rename(columns={'target': f'target_lead{base_lead}'})
        
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

        dfc_nanfree = self._drop_nans(dfc)

        # if target == 'cases' => target columns should be integers
        if self.config.target_column == 'cases':
            dfc_nanfree = self._set_targettype_integer(dfc_nanfree)

        # Groundtruth uses the first target horizon
        groundtruth_df = dfc_nanfree.copy()[['timestamp', 'node', f'target_lead{base_lead}']]
        groundtruth_df = groundtruth_df.rename(columns={f'target_lead{base_lead}': 'target'})

        if self.config.verbose:
            print(f'{checkmark}{checkmark} All data finalized') 
        if self.config.verbose > 1:
            print("")

        return FinalizedEpiData(
            data=dfc_nanfree,
            config=self.config,
            groundtruth=groundtruth_df
        )

# ====================================================
# ============ MAIN LOADER (ORCHESTRATOR) ============
# ====================================================
class DataOrchestrationContainerNotFound(Exception):
    def __init__(self, datastage: str, previous_method: str):
        super().__init__(f"No {datastage} attribute found for DataOrchestrator. Run {previous_method}() first")

class DataOrchestrator:
    """ 
    Main dataprepper - orchestrator.
    Outsources heavy lifting to subclasses, each of which has some functionality,
    wrapped into their `orchestrate()` method.

    Parameters
    ----------
    config: EpiConfig

    Attributes
    ----------
    General attributes include:
        - config (identical to the input config)
        - column_registry (dictionary of all columns, their type and transformations)

    An attribute is set at each datastage:
        - data_raw
        - data_context
        - data_harmonized
        - data_processed
        - data_feature
        - data_normalized
        - data_final
    
    Examples
    --------
    #### run dataorchestrator one
    >>> data_orchestrator = DataOrchestrator(config).build()

    #### run dataorchestrator stepwise
    >>> data_orchestrator = (DataOrchestrator(config)
                            .load_raw()
                            .harmonize_raw()
                            .process_data()
                            .build_features()
                            .normalize()
                            .finalize()
                            )
    
    Downstream
    ----------
    dataloader objects: ShallowDataLoaderManager, GraphDataLoaderManager
    These classes extract EpiConfig and data_final and prepare those into model-specific
    datasets.

    See Also
    --------
    #### EpiConfig
    dataclass object that stores configuration parameters to run the orchestration process smoothly

    #### datacontainers
    Each task within this orchestrator has a designated class and datacontainer.
    For example, self.reader is an isntance of EpiDataReader, which produces an
    instance of RawEpiData. These datacontainers can be found in .datacontainers.py
    """
    
    def __init__(self, config: 'EpiConfig'):
        self.config         = config

        # Initialize column registration
        self.column_registration = ColumnRegistration()
        self.column_registration.add_column(
            self.config.temporal_column, 
            'context'
        )
        self.column_registration.add_column(
            self.config.id_column, 
            'context'
        )   

        # Initialize pipeline components
        self.reader         = EpiDataReader(config)
        self.harmonizer     = NUTSHarmonizer(config)
        self.processor      = EpiDataProcessor(config)
        self.feature_builder= EpiFeatureBuilder(config, self.column_registration)
        self.normalizer     = EpiNormalizer(config, self.column_registration)
        self.finalizer      = EpiDataFinalizer(config, self.column_registration)
        
        # Store results at each stage
        self._data_raw       = None
        self._data_harmonized= None
        self._data_context   = None
        self._data_processed = None
        self._data_feature   = None
        self._data_normalized= None
        self._data_final     = None
          
    def load_raw(self) -> 'DataOrchestrator':
        """Load raw data from files"""
        self._data_raw = self.reader.orchestrate()
        return self
    
    def harmonize_raw(self) -> 'DataOrchestrator':
        """Harmonize data on NUTS-level"""        
        self._data_harmonized, self._data_context = self.harmonizer.orchestrate(self.data_raw)        
        return self
    
    def process_data(self) -> 'DataOrchestrator':
        """Preprocess the harmonized data"""
        self._data_processed = self.processor.orchestrate(self._data_harmonized)
        return self

    def build_features(self) -> 'DataOrchestrator':
        """build features. Note that this method adjusts self.column_registry."""
        self._data_feature = self.feature_builder.orchestrate(self.data_processed)
        return self        
   
    def normalize(self) -> 'DataOrchestrator':
        """normalize data. Note that this method adjusts self.column_registry."""   
        self._data_normalized = self.normalizer.orchestrate(self.data_feature)   
        return self      

    def finalize(self) -> 'DataOrchestrator':
        """Finalize data."""
        self._data_final = self.finalizer.orchestrate(self.data_normalized)
        return self        

    def build(self) -> 'DataOrchestrator':
        """Execute full pipeline and return final dataset."""
        return (self
            .load_raw()
            .harmonize_raw()
            .process_data()
            .build_features()
            .normalize()
            .finalize()
        )
    
    def preview(self, node_idx: int, status = Literal['processed']): 
        """preview dataloader"""
        if status == 'processed':
            fig, ax = plt.subplots(1,1, figsize = (14,4))
            data = self.data_processed.data
            sns.lineplot(data[data['node'] == node_idx], x = 'timestamp', y=f'{self.config.target_column}')
            ax.grid()
            ax.set_xlabel("")
        else:
            raise ValueError(f'currently no other data stage than "processed" implemented for the previewer.')
                 

        ax.set_title(f"processed {self.config.target_column} in node {node_idx}") 
        plt.tight_layout()  

    def __repr__(self):
        stages = []
        if self._data_raw is not None:
            stages.append("raw")
        if self._data_harmonized is not None:
            stages.append("harmonized")            
        if self._data_processed is not None:
            stages.append('processed')
        if self._data_feature is not None:
            stages.append('features') 
        if self._data_normalized is not None:
            stages.append('normalized')
        if self._data_final is not None:
            stages.append('finalized')            
        
        return f"<DataOrchestrator(disease={self.config.disease}, data stages={stages})>"
    
    @property
    def data_raw(self) -> RawEpiData:
        if not self._data_raw:
            raise DataOrchestrationContainerNotFound(datastage = 'data_raw', previous_method = 'load_raw')
        return self._data_raw    
    
    @property
    def data_context(self) -> ContextData:
        if not self._data_context:
            raise DataOrchestrationContainerNotFound(datastage = 'data_context', previous_method = 'harmonize_raw')

        return self._data_context      

    @property
    def data_harmonized(self) -> HarmonizedData:
        if not self._data_harmonized:
            raise DataOrchestrationContainerNotFound(datastage = 'data_harmonized', previous_method = 'harmonize_raw')

        return self._data_harmonized        
           
    @property
    def data_processed(self) -> ProcessedEpiData:
        if not self._data_processed:
            raise DataOrchestrationContainerNotFound(datastage = 'data_processed', previous_method = 'harmonize_raw')

        return self._data_processed    

    @property 
    def data_feature(self) -> FeatureEpiData:
        if not self._data_feature:
            raise DataOrchestrationContainerNotFound(datastage = 'data_feature', previous_method = 'data_processed')        

        return self._data_feature    
    
    @property
    def data_normalized(self) -> NormalizedEpiData:
        if not self._data_normalized:
            raise DataOrchestrationContainerNotFound(datastage = 'data_normalized', previous_method = 'build_features')        

        return self._data_normalized    

    @property
    def data_final(self) -> FinalizedEpiData:
        if not self._data_final:
            raise DataOrchestrationContainerNotFound(datastage = 'data_final', previous_method = 'normalize')        

        return self._data_final 