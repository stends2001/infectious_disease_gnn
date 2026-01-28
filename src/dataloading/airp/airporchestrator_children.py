from typing import Tuple, Dict, Any, Union, List, TYPE_CHECKING
import itertools
import pandas as pd 
import geopandas as gpd
import numpy as np

if TYPE_CHECKING:
    from .airpconfig import AirpConfig

from .airpdatacontainers import RawAirpData, ProcessedAirpData, ContextAirpData, FeatureAirpData, NormalizedAirpData

# ============= READER CLASS =============
class AirpDataReader:
    """
    Only dataloading

    Parameters
    ----------
    config: 'AirpConfig'

    Examples
    --------
    >>> rawdata = AirpDataReader(config).orchestrate()

    Returns
    -------
    .orchestrate() --> RawAirpData
    """
    
    def __init__(self, config: 'AirpConfig'):
        self.config = config
    
    def _load_flights_data(self) -> pd.DataFrame:
        """
        loads (cleaned) flights data

        df looks like:
        _______________________________________________________________________________________________________________
        | 'timestamp' | 'ori_country' | 'dest_airport_name' | 'dest_airport_code ' | 'dest_city_code' | 'passengers' |
        """

        filepath = self.config.get_flights_path()

        df = pd.read_csv(
            filepath,
            parse_dates = ['timestamp'],
            dtype = {'passengers': int}
        )

        return df
    
    def _load_worldharm_lookup_data(self) -> pd.DataFrame:
        """
        loads world harmonization data

        df looks like:
        ____________________________________________________
        | 'country' | 'iso_code' | 'continent' | 'region ' |
        """
        filepath = self.config.get_worldharm_path()

        df = pd.read_csv(
            filepath,
            sep = "\t"
        )

        return df

    def _load_population_data(self) -> pd.DataFrame:
        """
        loads (cleaned) population data

        df looks like:
        __________________________________________
        | 'country' | 'year' | 'population_size' |
        """
        filepath = self.config.get_global_popsize_path()

        df = pd.read_csv(
            filepath,
            dtype = {'population_size': int}
        )

        return df

    def _load_case_data(self) -> pd.DataFrame:
        """
        loads WHO case data

        df looks like:
        ______________________________________________
        | 'country' | 'year' | 'timestamp' | 'cases' |
        """
        filepath = self.config.get_global_cases_path()

        df = pd.read_csv(
            filepath,
            parse_dates=['timestamp'],
            dtype = {'cases': int}
        )

        return df
    
    def _load_mcv1_data(self) -> pd.DataFrame:
        """
        loads WHO vaccination mcv1 data

        df looks like:
        _______________________________
        | 'country' | 'year' | 'mcv1' |
        """        
        filepath = self.config.get_global_mcv1_path()

        df = pd.read_csv(
            filepath,
        )

        return df        

    def _load_mcv2_data(self) -> pd.DataFrame:
        """
        loads WHO vaccination mcv2 data

        df looks like:
        _______________________________
        | 'country' | 'year' | 'mcv2' |
        """     
        filepath = self.config.get_global_mcv2_path()

        df = pd.read_csv(
            filepath,
        )

        return df        

    def _load_airports_shapedata(self) -> gpd.GeoDataFrame:
        """
        loads shapedata for IATA airports 

        df looks like:
        ___________________________________________________________________
        | 'code' | 'city_code' | 'city' | 'country' | 'type' | 'geometry' |

        """     
        filepath = self.config.get_airports_shapedata()

        gdf = gpd.read_file(filepath)

        return gdf         

    def orchestrate(self) -> RawAirpData:
        """
        Load all required data files
        
        Returns
        --------
        RawAirpData : Container with all raw dataframes
        """

        rawdata = RawAirpData(
            flights     = self._load_flights_data(),
            worldharm   = self._load_worldharm_lookup_data(),
            airportharm = self._load_airports_shapedata(),            
            popsize     = self._load_population_data(),
            mv_cases    = self._load_case_data(),
            mcv1        = self._load_mcv1_data(),
            mcv2        = self._load_mcv2_data()
        )
        return rawdata

# ============= PROCESSOR CLASS ==========
class AirpDataProcessor:
    """
    Processes RawAirpData

    Parameters
    ----------
    config: 'AirpConfig'
    rawdata: 'RawAirpData'

    Examples
    --------
    >>> processed_data = AirpDataProcessor(config, rawdata).orchestrate()

    Returns
    -------
    .orchestrate() -> Tuple['ProcessedAirpData', 'ContextAirpData']
    """
    def __init__(self, config: 'AirpConfig', rawdata: 'RawAirpData'):
        self.config = config 
        self.rawdata= rawdata

    def _return_tokenization_map(self, df: pd.DataFrame, token_col: str) -> Tuple[Dict[str,int], Dict[int,str]]:
        """
        from input dataframe returns two dictionaries of the tokenization:
        - {token_col-value  : token-value}      => [str, int]
        - {token-value      : token_col-value}  => [int, str]

        tokens are associated alphabetically
        """
        unique_countries    = sorted(list(df[token_col].unique()))
        colvalue_token       = {}
        token_colvalue       = {}

        for idx, cc in enumerate(unique_countries):
            colvalue_token[cc] = idx 
            token_colvalue[idx]= cc 

        return colvalue_token, token_colvalue
    
    def _apply_tokenization(self, df: pd.DataFrame, tokenization_map: Dict[str, int], token_col: str):
        """create new column 'id' from 'country' column, using mapping"""
        dfc         = df.copy()
        dfc['id']   = dfc[token_col].map(tokenization_map)
        dfc.reset_index(drop = True, inplace = True)   
        return dfc     

    def _filter_dates(self, df: pd.DataFrame, min_date: str, max_date: str) -> pd.DataFrame:
        """filter on min and max date. for columns without timestamp column, a year column is used. If year is also not present, error is raised"""
        dfc = df.copy()     

        if 'timestamp' not in df.columns:
        
            if 'year' in df.columns:
                dfc['timestamp'] = pd.to_datetime(dfc['year'].astype(str) + '-01-01')      

                dfc = dfc[dfc['timestamp']< max_date]
                dfc = dfc[dfc['timestamp']>=min_date]

                dfc = dfc.drop(columns = 'timestamp')    

            else:
                raise ValueError(f'no column "year" or "timestamp" found for df {dfc}')
        
        else:
            dfc = dfc[dfc['timestamp']< max_date]
            dfc = dfc[dfc['timestamp']>=min_date]        

        dfc.reset_index(drop = True, inplace = True)   

        return dfc
    
    def _impute_missing_cases(self, df: pd.DataFrame) -> pd.DataFrame:
        """imputes any missing pair of timestamp/country with 0 cases"""
        unique_countries = df['country'].unique()
        unique_timestamps= df['timestamp'].unique()

        all_combinations = pd.DataFrame(list(itertools.product(unique_countries, unique_timestamps)), 
                                        columns=['country', 'timestamp'])

        imputed_df              = all_combinations.merge(df.copy(), on=['country', 'timestamp'], how='left')
        imputed_df['timestamp'] = pd.to_datetime(imputed_df['timestamp'])
        imputed_df['year']      = imputed_df['timestamp'].dt.year           # type: ignore[attr-defined] => we know its a pd.datetime series
        
        imputed_df['cases'] = imputed_df['cases'].fillna(0)  
        return imputed_df    

    def _create_incidence(self, epi_df: pd.DataFrame, pop_df: pd.DataFrame) -> pd.DataFrame:
        epipop_data              = pd.merge(pop_df, epi_df, on = ['country','year'])
        epipop_data['incidence'] = epipop_data['cases'] / epipop_data['population_size'] * self.config.incidence_scalar
        return epipop_data
    
    def _filter_airport_selection(self, df_airp_harm: gpd.GeoDataFrame, df_flights: pd.DataFrame) -> gpd.GeoDataFrame:
        airport_selection    = df_airp_harm.copy()
        unique_airport_codes = list(set(df_flights['dest_airport_code']))
        
        airport_selection    = airport_selection[airport_selection['code'].isin(unique_airport_codes)].reset_index(drop = True)
        return airport_selection

    def _validate_num_airports(self, airport_selection: gpd.GeoDataFrame, df_flights: pd.DataFrame) -> int:
        set_airports_iata           = set(airport_selection['code'])
        set_airports_flights_data   = set(df_flights['dest_airport_code'])

        if set_airports_flights_data != set_airports_iata:
            raise ValueError(f'set airports iata is unequal to set airports flights data!')
        
        return len(set_airports_iata)

    def orchestrate(self) -> Tuple['ProcessedAirpData', 'ContextAirpData']:
        # data filtering
        df_pop      = self._filter_dates(self.rawdata.popsize, min_date= self.config.min_date, max_date = self.config.max_date)
        df_epi      = self._filter_dates(self.rawdata.mv_cases, min_date= self.config.min_date, max_date = self.config.max_date)
        df_flights  = self._filter_dates(self.rawdata.flights, min_date= self.config.min_date, max_date = self.config.max_date)

        # filter iata airports
        airport_harm = self._filter_airport_selection(self.rawdata.airportharm, df_flights)
        num_airports = self._validate_num_airports(airport_harm, df_flights)

        # epi data processing
        df_epi      = self._impute_missing_cases(df_epi)
        df_epi      = self._create_incidence(df_epi, df_pop)

        # tokenizing -> first we get the maps
        country_token, token_country    = self._return_tokenization_map(self.rawdata.worldharm, 'country')
        airpcode_token, token_airpcode  = self._return_tokenization_map(df_flights, 'dest_airport_code')

        # apply tokenization
        # tokenize - countries (L1)
        world_harm_t                 = self._apply_tokenization(self.rawdata.worldharm, country_token, 'country').rename(columns = {'id': 'id1'})
        epi_df_t                     = self._apply_tokenization(df_epi, country_token, 'country').rename(columns = {'id': 'id1'})
        flights_df_t                 = self._apply_tokenization(df_flights,country_token, 'ori_country').rename(columns = {'id': 'id1'})
        # tokenize - airports (L2)
        flights_df_tt                = self._apply_tokenization(flights_df_t, airpcode_token, 'dest_airport_code').rename(columns = {'id': 'id2'})
        airport_harm_t               = self._apply_tokenization(airport_harm, airpcode_token, 'code').rename(columns = {'id': 'id2'})
        
        # TODO: clean dfs by removing columns

        processed_data = ProcessedAirpData(flights_df_tt, epi_df_t)
        context_data   = ContextAirpData(world_harm_t, 
                                         airport_harm_t, 
                                         tokenization_map_airports=(airpcode_token, token_airpcode), 
                                         tokenization_map_countries=(country_token, token_country),
                                         num_airports = num_airports)

        return processed_data, context_data

# ============= FEATUREBUILDER CLASS =====
class AirpFeatureBuilder:
    """
    Processes ProcessedAirpData into the X data for graph layer 2

    Parameters
    ----------
    config: 'AirpConfig'
    processeddata: 'ProcessedAirpData'

    Examples
    --------
    >>> feature_data = AirpFeatureBuilder(config, processeddata).orchestrate()

    Returns
    -------
    .orchestrate() -> ProcessedAirpData'
    """
    def __init__(self, config: 'AirpConfig', processeddata: 'ProcessedAirpData'):
        self.config         = config 
        self.processeddata  = processeddata        

    def _merge_epiflightdata(self, epi_df: pd.DataFrame, flights_df: pd.DataFrame) -> pd.DataFrame:
        # TODO: currently I'm dropping columns here => needs to be done elsewhere
        epi_df          = epi_df.copy()[['timestamp','incidence','id1']]
        flights_df      = flights_df.copy()[['timestamp','id1','id2','passengers']]
        epiflightdata   = pd.merge(epi_df,flights_df, on = ['id1','timestamp'])
        return epiflightdata
    
    def _calculate_risk_scores(self, epiflights: pd.DataFrame) -> pd.DataFrame:
        epiflights['rs'] = epiflights['passengers'] * epiflights['incidence']
        return epiflights

    def _log_passengers(self, epiflights: pd.DataFrame) -> pd.DataFrame:
        epiflights['passengers'] = np.log1p(epiflights['passengers'])
        return epiflights

    def _aggregate_risk_scores(self, epiflights: pd.DataFrame) -> pd.DataFrame:
        aggregated_risk_scores = epiflights.groupby(['timestamp','id2'])['rs'].sum().reset_index(drop = False)
        return aggregated_risk_scores
    
    def orchestrate(self) -> 'FeatureAirpData':
        epiflights = self._merge_epiflightdata(self.processeddata.epidata,self.processeddata.flightsdata)
        
        # log passengers
        epiflights = self._log_passengers(epiflights)

        # risk score = log(passengers) * incidence
        epiflights = self._calculate_risk_scores(epiflights)

        # aggregate per receiving airport
        risk_scores = self._aggregate_risk_scores(epiflights)

        feature_data = FeatureAirpData(risk_scores)

        return feature_data

# ============= NORMALIZER CLASS =====             
class AirpNormalizer:
    """
    Normalizes FeatureAirpDAta

    Parameters
    ----------
    config: 'AirpConfig'
    featuredata: 'FeatureAirpData'

    Examples
    --------
    >>> normalized_data = AirpNormalizer(config, featuredata).orchestrate()

    Returns
    -------
    .orchestrate() -> NormalizedAirpData'
    """
    def __init__(self, config: 'AirpConfig', featuredata: 'FeatureAirpData', contextdata: 'ContextAirpData'):
        self.config         = config 
        self.featuredata    = featuredata
        self.contextdata    = contextdata

    def _return_zscore_params(self, df: pd.DataFrame, col: Union[str, List[str]]) -> Dict[str,Any]:

        if isinstance(col, str):
            col = [col]

        params_dict = {}

        for cc in col:
            mean, std       = df[cc].mean(), df[cc].std()
            params_dict[cc]= {'method' : 'zscore', 'params' : {'mean': mean, 'std' : std}}

        return params_dict
    
    def _return_minmax_params(self, df: pd.DataFrame, col: Union[str, List[str]]) -> Dict[str, Any]:

        if isinstance(col, str):
            col = [col]

        params_dict = {}

        for cc in col:
            min, max        = df[cc].min(), df[cc].max()
            params_dict[cc] = {'method': 'minmax', 'params': {'min': min, 'max' : max}}
        return params_dict        

    def _apply_normalization(self, df: pd.DataFrame, normalization_dict: Dict[str, Any]) -> pd.DataFrame:
        dfnormalized = df.copy()

        columns = normalization_dict.keys()

        for col in columns:
            if col not in df.columns:
                raise ValueError(f'missing column {col} in df')
            
            col_dict    = normalization_dict[col]
            col_method  = col_dict['method'] 
            col_params  = col_dict['params']
            
            # apply minmax
            if col_method == 'minmax':
                min  = col_params['min']
                max  = col_params['max']

                if max - min == 0:
                    print(f'no deviation found when normalizing {col}. column is put to 0')
                    dfnormalized[col] = 0.0
                else:
                    dfnormalized[col] = (df[col] - min) / (max - min)     

            # apply zscore
            elif col_method == 'zscore':
                mean = col_params['mean']
                std  = col_params['std']

                if std == 0:
                    print(f'no deviation found when normalizing {col}. column is put to 0')
                    dfnormalized[col] = 0
                else:
                    dfnormalized[col] = (df[col] - mean) / std

            else:
                raise ValueError(f'unavailable normalization method found for {col}: {col_method}. supported methods are "zscore" or "minmax"')            

        return dfnormalized        

    def _impute_missing_combinations(self, df: pd.DataFrame, col1: str, col2: str, imputed_col: str) -> pd.DataFrame:
        """TODO: remove redundant/repeated logic in processor for cases per country who data"""
        unique_col1 = df[col1].unique()
        unique_col2 = df[col2].unique()

        all_combinations = pd.DataFrame(list(itertools.product(unique_col1, unique_col2)), 
                                        columns=[col1, col2])

        imputed_df              = all_combinations.merge(df.copy(), on=[col1, col2], how='left')        
        imputed_df[imputed_col] = imputed_df[imputed_col].fillna(0)  
        return imputed_df    

    def _widen_ids(self, df: pd.DataFrame, reverse: bool = False) -> pd.DataFrame:
        """when normalizing airports' risk scores per airport we'll have them in different columns to make the normalizing easier"""
        df_transformed = df.copy()

        if reverse:
            df_transformed = df_transformed.melt(id_vars = 'timestamp' ,value_name='rs', var_name='id2')
            df_transformed['id2'] = df_transformed['id2'].astype(int)
            
        else:
            df_transformed = df.pivot_table(index = 'timestamp', values= 'rs', columns = 'id2').reset_index(drop = False)
            df_transformed = df_transformed.rename(columns = {c: str(c) for c in df_transformed.columns})

        return df_transformed
    
    def orchestrate(self) -> 'NormalizedAirpData':
        # add combinations of columns > NaN risk means 0 passengers means 0 risk
        imputed_df = self._impute_missing_combinations(self.featuredata.data, 'timestamp','id2','rs')

        # normalization
        # if normalization_group == 'collectively'
        if self.config.normalization_group == 'collectively':
            normalization_parameters= self._return_zscore_params(imputed_df, col = 'rs')
            normalized_df           = self._apply_normalization(imputed_df,normalization_parameters)            
        
        # if normalization_group == 'individually' => airport wise normalization
        elif self.config.normalization_group == 'individually':
            seperated_columns               = self._widen_ids(imputed_df)
            # make list of strings of all ids
            airport_ids_columns             = [str(id) for id in range(0,self.contextdata.num_airports)]
            normalization_parameters        = self._return_zscore_params(seperated_columns, col = airport_ids_columns)
            normalized_df                   = self._apply_normalization(seperated_columns, normalization_parameters)
            normalized_df                   = self._widen_ids(normalized_df, reverse = True)       

        else:
            raise ValueError(f'unrecognised value for AirpConfig.normalization_group: {self.config.normalization_group}. Please choose between "collectively" and "individually"')     

        normalized_airp_data = NormalizedAirpData(normalized_df, normalization_parameters)
        return normalized_airp_data

