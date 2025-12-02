import pandas as pd
import geopandas as gpd
from typing import Tuple, Dict, Union, Literal
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ...utils.textformatting import warning_emoji, checkmark
from ...utils.constants import berlin_district_ids
from .normalization import apply_minmax_scaling, apply_zscore_scaling, pipeline_minmax_normalization, pipeline_zscore_normalization
from .column_registry import ColumnRegistration, ColEntryMissingError, ColEntryMissingTransformationError, ColEntryMissingTransformationReferralError
from .epiconfig import EpiConfig
from .datastagecontainers import RawEpiData, ContextData, HarmonizedData, ProcessedEpiData, FeatureEpiData, NormalizedEpiData, FinalizedEpiData, ProcessedEpiData

# ============= READER CLASS =============

class EpiDataReader:
    """
    """
    
    def __init__(self, config: 'EpiConfig'):
        self.config = config
    
    def _load_disease_data(self) -> pd.DataFrame:
        """
        Load raw disease case data downloaded from SurvStat.
        
        Returns:
        --------
        pd.DataFrame with columns: [timestamp, nuts3, cases, week, year]
        """
        filepath = self.config.get_disease_path()
                
        df = pd.read_csv(
            filepath,
            parse_dates=['timestamp'],
            dtype={'kz_kreis': str, 'cases':int}
        ).rename(columns={'kz_kreis': 'nuts3'})
        
        if self.config.verbose > 1:
            print(f"{checkmark} Loaded raw disease data")
        
        return df
    
    def _load_population_data(self) -> pd.DataFrame:
        """
        Load population size data.
        
        Returns:
        --------
        pd.DataFrame with columns: [nuts3, year, population_size]
        """
        filepath = self.config.get_population_path()
        
        df = pd.read_csv(
            filepath,
            dtype={'nuts3': str}
        )
        
        if self.config.verbose > 1:
            print(f"{checkmark} Loaded raw population data")
        
        return df
    
    def _load_population_data_berlin_districts(self) -> pd.DataFrame:
        """
        Load population size data for districts in Berlin.
        
        Returns:
        --------
        pd.DataFrame with columns: [nuts3, population_size]
        """        
        filepath = self.config.get_population_berlin_districts_path()
        
        df = pd.read_csv(
            filepath,
            dtype={'nuts3': str})
        
        if self.config.verbose > 1:
            print(f"{checkmark} Loaded raw berlin districts population data")
        
        return df        

    def _load_shapedata(self) -> gpd.GeoDataFrame:
        """
        Load geographic shapefile.
        
        Returns:
        --------
        GeoDataFrame with geometry and nuts code
        """
        filepath = self.config.get_shapefile_path()
        
        gdf = gpd.read_file(filepath).drop(columns=['level'], errors='ignore')
        
        if self.config.verbose > 1:
            print(f"{checkmark} Loaded raw shapedata ({len(gdf)} regions)")
        
        return gdf
    
    def _load_harmonization_data(self) -> pd.DataFrame:
        """Load NUTS harmonization mapping."""
        filepath = self.config.get_harmonization_path()
               
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        
        if self.config.verbose > 1:
            print(f"{checkmark} Loaded raw harmonization data")
        
        return df
    
    def _load_nuts_names(self) -> pd.DataFrame:
        """
        this datafile has been inspired by the SurvNet database, 
        though not all kreisen are in it, so I've made some additions.
        Look at DataCleaning in survnet Environment
        """
        # main file, additions-file
        filepath = self.config.get_nuts_names_path()   
        
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        
        if self.config.verbose > 1:
            print(f"{checkmark} Loaded raw NUTS names")
        
        return df
    
    def orchestrate(self) -> RawEpiData:
        """
        Load all required data files.
        
        Returns:
        --------
        RawEpiData : Container with all raw dataframes
        """
        rawdata = RawEpiData(
            disease             = self._load_disease_data(),
            population          = self._load_population_data(),
            shapedata           = self._load_shapedata(),
            harmonization       = self._load_harmonization_data(),
            nuts_names          = self._load_nuts_names(),
            population_berlin   = self._load_population_data_berlin_districts() if self.config.split_berlin else None
        )
        return rawdata
    
# ============= HARMONIZATION CLASS =============

class NUTSHarmonizer:
    """
    """
    def __init__(self, config: 'EpiConfig'):
        self.config = config    

    def _add_berlin_districts(self, population_nuts3: pd.DataFrame, population_berlin_districts: pd.DataFrame) -> pd.DataFrame:
        """
        based on population data for nuts3 and for berlin districts, concatenate into one df with 412 nuts3 values.
        """
        total_population_berlin                         = sum(population_berlin_districts['population_size'])
        relative_population_size_berlin_districts       = population_berlin_districts.copy()
        relative_population_size_berlin_districts['population_size']= relative_population_size_berlin_districts['population_size'] / total_population_berlin

        df_11000 = population_nuts3[population_nuts3['nuts3'] == '11000'][['year', 'population_size']].set_index('year')
            
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
        if berlin is not supposed to be split, mutate all nuts3 values of the districts into
        berlin ones (11000) for the subsequent aggregation onto nuts3/nuts2/nuts1 levels.
        """
        epidemiology_df.loc[epidemiology_df['nuts3'].isin(berlin_district_ids), 'nuts3'] = '11000'
        
        if self.config.verbose > 1:
            print(f'{checkmark} berlin districts renamed into berlin city')  

        return epidemiology_df

    def _aggregate_by_nuts(self, epidemiology_df: pd.DataFrame, population_df: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
        """aggregates epidemiology and epopulation data per nuts level"""
        epidemiology_df_aggr = epidemiology_df.groupby(['timestamp', self.config.nuts_level]).aggregate({'cases':'sum'}).reset_index()     
        population_df_aggr   = population_df.groupby(['year', self.config.nuts_level]).aggregate({'population_size':'sum'}).reset_index() 

        
        if self.config.verbose > 1:
            print(f'{checkmark} epidemiology and population data aggregated on nuts')          
        return epidemiology_df_aggr, population_df_aggr

    def _add_nuts_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """adds nuts-level column"""
        if self.config.nuts_level == "nuts1":
            df['nuts1']= df['nuts3'].str[:2]     

        elif self.config.nuts_level =='nuts2':
            df['nuts2']= df['nuts3'].str[:3]

        return df

    def _merge_epipopdata(self, epidemiology_df: pd.DataFrame, population_df: pd.DataFrame) -> pd.DataFrame:
        """
        merge epidemiology with population data, by extracting the year from epidemiology data. 
        The year column is dropped
        """
        if 'year' not in epidemiology_df.columns:
            epidemiology_df['year'] = epidemiology_df['timestamp'].dt.year 

        return pd.merge(epidemiology_df, population_df, on = [self.config.nuts_level,'year'])

    def _get_nuts_data(self, raw_nuts_names: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the nuts data with only the unique values.
        Initially the raw_nuts_names is a df of all nuts3/nuts2/nuts1 levels.
        So if nuts_level is 2 or 1, many entries can be dropped.
        """
        columns     = [f'{self.config.nuts_level}_key',f'{self.config.nuts_level}_name']
        unique_nuts = raw_nuts_names[columns].drop_duplicates().reset_index(drop=True)
        unique_nuts.rename(columns = {f'{self.config.nuts_level}_key':self.config.nuts_level}, inplace=True)

        return unique_nuts

    def _tokenize_data(self, dfs: Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]]) -> Tuple[Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]], Dict[str, Dict[Union[int,str],Union[int,str]]]]:
        """
        Tokenize nuts-levels into 'node' column.
        """
        unique_ids = sorted(dfs['epipopdata'][f'{self.config.nuts_level}'].unique())
        id_idx     = {} # nuts  : int
        idx_id     = {} # int   : nuts

        for idx, id in enumerate(unique_ids):
            id_idx[id] = idx                  # id (nuts1, nuts2 or nuts3) : node_id (zero based)
            idx_id[idx] = str(id)             # node_id (zero based): id (nuts1, nuts2 or nuts3)

        tokenization_map = {"id_idx": id_idx, "idx_id": idx_id} 
        df_dict = {}

        for dfname, df in dfs.items():
            dfc         = df.copy()

            dfc['node'] = dfc[self.config.nuts_level]
            dfc['node'] = dfc[f'{self.config.nuts_level}'].map(id_idx)  

            # get nuts-levels for rows to be dropped
            rows_to_drop        = dfc[dfc['node'].isna()]
            dropped_nuts_values = rows_to_drop[self.config.nuts_level].unique()
                    
            # drop rows
            dfr         = dfc.copy().dropna(subset=['node'])
            dfr['node'] = dfr['node'].astype(int)

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
                    print(f'{warning_emoji}dropping non-tokenized nodes in {dfname}: dropped {number_dropped_rows} nodes')

            df_dict[dfname] = dfr
     
        if self.config.verbose > 1:
            print(f'{checkmark} nodes tokenized')       
        return df_dict, tokenization_map       

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
        nutsnames           = self._get_nuts_data(rawdata.nuts_names)

        tokenized_datasets, tokenization_map = self._tokenize_data(dfs = {'epipopdata': epipopdata, 'shapedata': rawdata.shapedata, 'nutsnames': nutsnames})

        harmdata = HarmonizedData(
            data          = tokenized_datasets['epipopdata'],
        )
        ctxdata = ContextData(
            nuts_level          = self.config.nuts_level,
            num_nodes           = len(tokenization_map['id_idx']),
            shapedata           = gpd.GeoDataFrame(tokenized_datasets['shapedata']),
            nuts_names          = tokenized_datasets['nutsnames'],
            tokenization_map    = tokenization_map            
        )

        return harmdata, ctxdata

# ============= PREPROCESSING CLASS =============
class EpiDataPreprocessor:
    """ 
    """
    def __init__(self, config: 'EpiConfig'):
        self.config = config

    def _add_incidence_column(self, epipopdata: pd.DataFrame) -> pd.DataFrame:
        """adds incidence column"""
        epipopdata['incidence'] = epipopdata['cases'] / epipopdata['population_size'] * self.config.incidence_scalar
        return epipopdata
    
    def _drop_cases_column(self, epipopdata: pd.DataFrame) -> pd.DataFrame:
        """drops cases column"""
        return epipopdata.drop(columns=['cases']) 
       
    def _drop_cols(self, epipopdata: pd.DataFrame) -> pd.DataFrame:
        """drop redundant columns"""
        return epipopdata.drop(columns = [self.config.nuts_level, 'year'])

    def _filter_mindate(self, df, min_date: pd.Timestamp) -> pd.DataFrame:
        return df.loc[df['timestamp'] >= min_date].reset_index(drop = True)                

    def _filter_maxdate(self, df, max_date: pd.Timestamp) -> pd.DataFrame:
        return df.loc[df['timestamp'] < max_date].reset_index(drop = True)       

    def orchestrate(self, harmonizeddata: 'HarmonizedData') -> 'ProcessedEpiData':
        """
        The function that orchestrates all others
        """
        epipopdata      = self._add_incidence_column(harmonizeddata.data)
        
        print(f"target: {self.config.target_column}")

        if self.config.target_column != 'cases':
            epipopdata      = self._drop_cases_column(epipopdata)        

        epipopdata      = self._filter_maxdate(epipopdata, self.config.max_date)
        epipopdata      = self._filter_mindate(epipopdata, self.config.min_date_extended)     

        if self.config.verbose:
            print(f"{checkmark} min_date has been extended to {self.config.min_date_extended.date()}")

        if self.config.verbose > 1:
            print(f'{checkmark} epidata filtered between {self.config.min_date_extended.date()} - {self.config.max_date.date()}')               

        epipopdata = self._drop_cols(epipopdata)

        return ProcessedEpiData(data = epipopdata)

# ============= FEATURE CLASS ============= 
            
class EpiFeatureBuilder:
    """
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
        return dfc

    def _log_transform(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """log_transform target"""
        df_transformed                              = df.copy()
        df_transformed[col]                         = np.log(df_transformed[col] + self.config.log_shift)

        self.column_registration.update_transformation(
            col, 
            {'log': self.config.log_shift}
        )         
        return df_transformed

    def _lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Include current and lagged values of the target.
        
        - lag0 = current value at time t
        - lag1 = t-1 (yesterday)
        - lag2 = t-2, etc.
        
        Targets are shifted forward by horizon_leadtime after this.
        """
        for lag in range(self.config.lag_num):  # 0 to lags-1
            feature = f'{self.config.lag_column}_lag{lag}'
            df[feature] = df.groupby(self.config.id_column)[self.config.lag_column].shift(lag)
            
            self.column_registration.add_column(
                feature, 
                'feature',
                needs_normalization=True,
                transformation_group=self.config.lag_column
            )
        
        df = df.dropna().reset_index(drop=True)
        
        # Now shift target forward by horizon_leadtime so we're predicting the future
        if self.config.horizon_leadtime > 0:
            df[self.config.target_column] = df.groupby(
                self.config.id_column
            )[self.config.target_column].shift(-self.config.horizon_leadtime)
            df = df.dropna().reset_index(drop=True)
        
        return df

    def orchestrate(self, processed_data: 'ProcessedEpiData') -> 'FeatureEpiData':
        """orchestrates entire feature addition"""
        feature_data = processed_data.data

        # if population_size is a feature
        if self.config.include_population:
            # add to column registration
            self.column_registration.add_column(
                    'population_size', 
                    'feature', 
                    needs_normalization=True,
                    transformation_group=None  # Independent normalization
                )            
            if self.config.verbose > 1:
                print(f'{checkmark} population size included as feature')    
        else:
            # remove column population_size
            feature_data.drop(columns = ['population_size'], inplace = True) 
        
        if self.config.time_index:
            feature_data = self._time_index(feature_data)
            if self.config.verbose > 1:
                print(f'{checkmark} time index (sin/cos) included as feature')   

        if self.config.log_transform:
            for col in self.config.log_transform:
                feature_data = self._log_transform(feature_data, col)
                if self.config.verbose > 1:
                    print(f'{checkmark} {col} log-transformed')    
        
        feature_data = self._lags(feature_data)
        if self.config.verbose > 1:
            print(f'{checkmark} lags included as feature')   

        # feature_data.rename(columns = {self.config.target_column: 'target'}, inplace = True)
        # if self.config.verbose > 1:
        #     print(f'{checkmark}  {self.config.target_column} renamed to \'target\'')         

        return FeatureEpiData(data=feature_data)

# ============= Normalize CLASS ============= 

class EpiNormalizer:
    """
    Normalizes required columns.

    Parameters:
    ----------
    config: EpiConfig    
    column_registration: ColumnRegistration
        Shared registry for tracking all columns

    feature_data: FeatureEpiData
        Datacontainer holds preprocessed data with features     

    Returns: (`orchestrate`())
    -------
    instance of NormalizedEpiData
        Datacontainer holds normalized data 

    Warnings:
    --------
    An Error is thrown when a normalization method is inputted which is not supported.

    Note:
    ----
    The optional log-transformation of target is done in EpiFeatureBuilder
    TODO I definitely need to add test function to validate the normalization and the absence of data leakage!
    """
    def __init__(self, config: 'EpiConfig', column_registration: ColumnRegistration):
        self.config = config 
        self.column_registration = column_registration
        self.normalization_functions = {
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
        
        return df
    
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """normalizes data and stores information in column_registration"""
        train_df        = df[df['train']]
        normalized_df   = df.copy()

        if self.config.normalization_method not in self.normalization_functions['apply']:
            raise KeyError(f'No normalization function {self.config.normalization_method} found.')

        # First pass: Calculate normalization parameters for columns that need independent normalization
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

        # Second pass: Apply normalization to all columns
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

                else:
                    raise ColEntryMissingTransformationError(col_entry.column_name)
            
            # dependent transformation: expect a referral
            else:
                # Use reference column's normalization
                ref_col_entry = self.column_registration.get_by_name(col_entry.transformation_group)
                
                if ref_col_entry.transformation is None:
                    raise ColEntryMissingTransformationReferralError(col_entry.column_name, ref_col_entry.column_name)
                
                params = {col_entry.column_name: ref_col_entry.transformation['normalization']}
            
            # Apply normalization
            normalized_df = self.normalization_functions['apply'][self.config.normalization_method](
                normalized_df, 
                [col_entry.column_name], 
                params
            )

        return normalized_df

    def orchestrate(self, feature_data: 'FeatureEpiData') -> 'NormalizedEpiData':
        """runs all normalization functions"""
        split_data      = self._set_splits(feature_data.data)

        if self.config.verbose > 1:
            print(f'{checkmark} data split into train/val/test')   
        normalized_data = self._normalize(split_data)

        if self.config.verbose > 1:
            print(f'{checkmark} data normalized')   
        return NormalizedEpiData(data=normalized_data)

# ============= Finalize CLASS ============= 
 
class EpiDataFinalizer:
    """
    Finalizes epidata into horizon-specific dfs.
    """
    def __init__(self, config, column_registration: ColumnRegistration):
        self.config             = config 
        self.column_registration= column_registration

    def orchestrate(self, normalized_data: NormalizedEpiData) -> 'FinalizedEpiData':
        """runs entire finalization"""
        dfc = normalized_data.data

        # Create horizon-specific targets
        # Each horizon is the absolute number of steps ahead from "today"
        for horizon in range(self.config.horizon_size):
            steps_ahead = self.config.horizon_leadtime + horizon
            target      = f'target_ahead{steps_ahead}'
            
            # Shift from the base target (which is already at t+horizon_leadtime)
            if horizon == 0:
                dfc[target] = dfc[self.config.target_column]
            else:
                dfc[target] = dfc.groupby(
                    self.config.id_column
                )[self.config.target_column].shift(-horizon)
        
        self.column_registration.add_column(
            'target',
            'target',
            transformation_group=self.config.target_column,
            needs_normalization=True
        )
        self.column_registration.add_column(
            'pred',
            'pred',
            transformation_group=self.config.target_column,
            needs_normalization=True
        )     

        if self.config.target_column == 'cases':
            dfc.drop(columns=['incidence'], inplace=True)      

        if self.config.verbose > 1:
            print(f'{checkmark} targets for all horizons added')  

        # Drop any rows with NaN in target columns
        # dfc = dfc.dropna(subset=target_horizon_columns).reset_index(drop=True)     

        return FinalizedEpiData(
            data=dfc.drop(columns = [self.config.target_column]),
            config=self.config,
            groundtruth= dfc.rename(columns = {self.config.target_column: 'target'})[['timestamp','node','target']]
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

    Parameters:
    ----------
    config: EpiConfig

    Attributes:
    ----------
    General attributes include:
        - config (identical to the input config)
        - column_registry (dictionary of all columns, their type and transformations)

    An attribute is set at each datastage:
        - raw_data
        - context_data
        - processed_data
        - feature_data
        - split_data
        - normalized_data
        - finalized_data

    Downstream:
    ----------

    See Also:
    --------
    EpiConfig

    Each task within this orchestrator has a designated class and datacontainer.
    For example, self.reader is an isntance of EpiDataReader, which produces an
    instance of RawEpiData. These datacontainers can be found in .datacontainers.py

    Examples:
    --------
    run dataorchestrator one
    >>> data_orchestrator = DataOrchestrator(config).build()

    run dataorchestrator stepwise
    >>> data_orchestrator = (DataOrchestrator(config)
                            .load_raw()
                            .harmonize_raw()
                            .process_data()
                            .build_features()
                            .normalize()
                            .finalize()
                            )
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
        if self.config.target_column != 'incidence':
            self.column_registration.add_column(
                self.config.target_column, 
                'target', 
                needs_normalization=False,
                transformation_group=None  # Independent normalization
            )   
            self.column_registration.add_column(
                'incidence', 
                'feature', 
                needs_normalization=True,
                transformation_group=None  # Independent normalization
            )                      
        else:
            self.column_registration.add_column(
                'incidence', 
                'target', 
                needs_normalization=True,
                transformation_group=None  # Independent normalization
            )        
     
        # Initialize pipeline components
        self.reader         = EpiDataReader(config)
        self.harmonizer     = NUTSHarmonizer(config)
        self.preprocessor   = EpiDataPreprocessor(config)
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
        if self.config.verbose:
            print(f'{checkmark}{checkmark} All raw data loaded')
        if self.config.verbose > 1:
            print("")
        return self
    
    def harmonize_raw(self) -> 'DataOrchestrator':
        """Harmonize data on NUTS-level"""        

        self._data_harmonized, self._data_context = self.harmonizer.orchestrate(self.data_raw) 
        if self.config.verbose:
            print(f'{checkmark}{checkmark} All raw data nuts-harmonized')       
        if self.config.verbose > 1:
            print("")             
        return self
    
    def process_data(self) -> 'DataOrchestrator':
        """Preprocess the harmonized data"""
       
        self._data_processed = self.preprocessor.orchestrate(self._data_harmonized)
        
        if self.config.verbose:
            print(f'{checkmark}{checkmark} All data preprocessed')   
        if self.config.verbose > 1:
            print("")
        return self

    def build_features(self) -> 'DataOrchestrator':
        """build features. Note that this method adjusts self.column_registry."""
       
        self._data_feature = self.feature_builder.orchestrate(self.data_processed)

        if self.config.verbose:
            print(f'{checkmark}{checkmark} All features built') 
        if self.config.verbose > 1:
            print("")
        return self        
   
    def normalize(self) -> 'DataOrchestrator':
        """normalize data. Note that this method adjusts self.column_registry."""   

        self._data_normalized = self.normalizer.orchestrate(self.data_feature)   

        if self.config.verbose:
            print(f'{checkmark}{checkmark} All data normalized')         
        if self.config.verbose > 1:
            print("")
        return self      

    def finalize(self) -> 'DataOrchestrator':
        """Finalize data."""
        
        self._data_final = self.finalizer.orchestrate(self.data_normalized)

        if self.config.verbose:
            print(f'{checkmark}{checkmark} All data finalized') 
        if self.config.verbose > 1:
            print("")
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
            data = self.data_processed.epipopdata
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