import time
from typing import assert_never, TYPE_CHECKING , Dict, Tuple, Literal, Optional, Union
import pandas as pd
import geopandas as gpd

if TYPE_CHECKING:
    from ...epiconfig import EpiConfig

from ....utils.textformatting import checkmark
from ....utils.constants import berlin_district_ids, berlin_id
from ..utils.issues import EpiDataOrchestrationError
from ..utils.temporal_summary import EpiDataTemporalSummary

from ..epidatacontainers.rawepidata import RawEpiData
from ..epidatacontainers.harmonizedepidata import HarmonizedEpiData
from ..epidatacontainers.contextepidata import ContextEpiData

# ============= HARMONIZATION CLASS =============
class EpiDataHarmonizer:
    """
    """
    def __init__(self, epiconfig: 'EpiConfig'):
        self.epiconfig = epiconfig   

    def _mutate_berlin_districts(self, epidemiology_df: pd.DataFrame) -> pd.DataFrame:
        """
        When berlin not to be split -> mutate all nuts3 values of the districts into
        berlin ones (11000) for the subsequent aggregation onto nuts3/nuts2/nuts1 levels.
        """
        epidemiology_df.loc[epidemiology_df['nuts3_key'].isin(berlin_district_ids), 'nuts3_key'] = berlin_id

        if self.epiconfig.verbose > 1:
            print(f'{checkmark} berlin districts renamed into berlin city')  

        return epidemiology_df

    def _add_key_column(self, epi_df: pd.DataFrame, regio_harm: pd.DataFrame) -> pd.DataFrame:
        """adds key column"""

        regio_harm_cp = regio_harm.copy()

        match self.epiconfig.country:

            # get reporting - unit

            case 'netherlands':
                merge_key = 'lau_key'

            case 'germany':
                merge_key = 'nuts3_key'

            case _:
                assert_never(self.epiconfig.country)

        if merge_key != f'{self.epiconfig.level}_key':
            regio_harm_cp = regio_harm_cp.copy()[[merge_key,f'{self.epiconfig.level}_key']].rename(columns = {f'{self.epiconfig.level}_key' : 'key'})
        else:
            regio_harm_cp       = regio_harm_cp.copy()[[merge_key]]
            regio_harm_cp['key']= regio_harm_cp[merge_key]         

        merge = pd.merge(epi_df, regio_harm_cp , on = merge_key)

        if self.epiconfig.verbose > 1:
            print(f'{checkmark} nuts column added')  

        return merge

    def _aggregate_cases(self, epidemiology_df: pd.DataFrame) -> pd.DataFrame:
        """aggregates epidemiology and population data per nuts level"""
        cases_nuts_aggregated = epidemiology_df.groupby(['timestamp', 'key']).aggregate({'cases':'sum'}).reset_index()     

        if self.epiconfig.verbose > 1:
            print(f'{checkmark} epidemiology data aggregated on nuts')    

        return cases_nuts_aggregated

    def _filter_data_on_level(self, df: pd.DataFrame, drop_level = True) -> pd.DataFrame:
        dfc             = df.copy()
        dfc             = dfc[dfc['level'] == self.epiconfig.level].reset_index(drop = True)
        if drop_level:
            dfc = dfc.drop(columns = ['level'])
        return dfc

    def _set_year_col(self, df: pd.DataFrame) -> pd.DataFrame:
        df['year'] = df['timestamp'].dt.year 
        return df

    def _merge_epipopdata(self, epidemiology_df: pd.DataFrame, population_df: pd.DataFrame) -> pd.DataFrame:
        """
        merge epidemiology with population data, by extracting the year from epidemiology data. 
        The year column is dropped
        """

        if self.epiconfig.verbose > 1:
            print(f'{checkmark} epidemiological- and population data merged')  

        return pd.merge(epidemiology_df, population_df, on = [f'key','year'])

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
                    .groupby(f'key')
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
                    .groupby(f'key')
                    .resample('MS')
                    .agg({
                        'cases':            'sum',
                        'population_size':  'mean',
                    })
                    .reset_index(drop = False)
                )
            
        timestamps: pd.Series[pd.Timestamp] = resampled_df['timestamp']
        resampled_df['year']                = timestamps.dt.year
        resampled_df                        = resampled_df[[self.epiconfig.temporal_column,f'key','cases','year','population_size']]
        return resampled_df

    def _apply_tokenization(self, df: pd.DataFrame, tokenization_map: Dict[str, int], drop_key: bool = True):
        """
        applies tokeninization found in tokenization map, through:

        dfc[token_colname]   = dfc[col_to_tokenize].map(tokenization_map).astype('Int64')
        """
        dfc                             = df.copy()
        dfc[self.epiconfig.id_column]   = dfc['key'].map(tokenization_map).astype('Int64')
        dfc.reset_index(drop = True, inplace = True)   

        if drop_key:
            dfc.drop(columns = 'key', inplace=True)

        return dfc  

    def _ensure_geodataframe(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        if isinstance(df, pd.DataFrame):
            df = gpd.GeoDataFrame(df)
        return df

    def _get_keynames(self, df: pd.DataFrame):
        keynames = df[[f'{self.epiconfig.level}_key',f'{self.epiconfig.level}_name']].drop_duplicates()
        return keynames.reset_index(drop = True).rename(columns = {f'{self.epiconfig.level}_key' : 'key'})

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

        if self.epiconfig.country == 'germany':
            raw_epidata         = self._mutate_berlin_districts(rawdata.disease)
        else:
            raw_epidata         = rawdata.disease

        epidata             = self._add_key_column(raw_epidata, rawdata.region_harmonization)
        epidata             = self._aggregate_cases(epidata)
        epidata             = self._set_year_col(epidata)

        popdata_lvl         = self._filter_data_on_level(rawdata.population_size)
        local_shapedata     = self._filter_data_on_level(rawdata.shapedata.copy())

        epipopdata          = self._merge_epipopdata(epidata, popdata_lvl)
        epipopdata          = self._resample(epipopdata, self.epiconfig.temporal_frequency)

        keynames            = self._get_keynames(rawdata.region_harmonization.copy())

        # apply tokens
        mapping             = rawdata.tokenization_map
        epipopdata          = self._apply_tokenization(epipopdata, mapping)
        local_shapedata     = self._apply_tokenization(local_shapedata, mapping)
        keynames            = self._apply_tokenization(keynames, mapping, False)

        population_size_data= self._apply_tokenization(popdata_lvl, mapping)

        features = {    
            "population_size_data"      : ("feature_popsize",       "population_size"),              
            "population_density_data"   : ("feature_popdens",       "population_density"),
            "population_age_data"       : ("feature_popage",        "population_age"),
            "gisd_data"                 : ("feature_gisd",          "gisd"),
            "kreise_classes_data"       : ("feature_kreise_classes","kreise_classes"),
            "border_regions_data"       : ("feature_borders",       "borders"),
        }

        feature_datasets: Dict[str, Optional[Union[pd.DataFrame, gpd.GeoDataFrame]]] = {dataname: None for dataname in features}

        for out_name, (feature_flag, raw_attr) in features.items():
            if getattr(self.epiconfig, feature_flag):
                data = getattr(rawdata, raw_attr)
                data = self._filter_data_on_level(data)
                data = self._apply_tokenization(data, mapping)
                feature_datasets[out_name] = data                       


        harmdata = HarmonizedEpiData(
            epidata             = epipopdata,

            _population_size    = feature_datasets['population_size_data'],
            _population_density = feature_datasets['population_density_data'],
            _population_age     = feature_datasets['population_age_data'],
            _gisd               = feature_datasets['gisd_data'],            
            _kreise_classes     = feature_datasets['kreise_classes_data'],
            _borders            = feature_datasets['border_regions_data'],
            _vacmap             = None

        )
        
        ctxdata = ContextEpiData(
            country             = self.epiconfig.country,
            level               = self.epiconfig.level,
            global_shapedata    = self._ensure_geodataframe(rawdata.shapedata),
            local_shapedata     = self._ensure_geodataframe(local_shapedata),
            population_size     = population_size_data,
            nodenames           = keynames,
            region_harmonization= rawdata.region_harmonization,
            tokenization_map    = mapping,
            temporal_summary    = self._return_temporal_summary()            
        )

        time_end = time.time()
        if self.epiconfig.verbose > 2:
            print(f'Execution of Harmonizer took {round(time_end - time_start,3)}s')  
        if self.epiconfig.verbose > 1:
            print("")              

        return harmdata, ctxdata
