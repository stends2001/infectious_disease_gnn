import time
import json
from typing import assert_never, TYPE_CHECKING, Dict
import pandas as pd
import geopandas as gpd
import numpy as np
from ....utils.textformatting import checkmark

from ...epiconfig import EpiConfig

from ..utils.issues import EpiDataOrchestrationError
from ...columnregistration import ColumnRegistry
from ..containers import ProcessedEpiData, FeatureEpiData
from ..utils.temporal_summary import EpiDataTemporalSummary

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
                 epiconfig:             EpiConfig, 
                 column_registration:   ColumnRegistry,
                 temporal_summary:      EpiDataTemporalSummary):
        
        self.epiconfig          = epiconfig
        self.column_registration= column_registration
        self.temporal_summary   = temporal_summary

    def _register_static_features(self) -> None:
        """
        Register all feature columns into the column registry based on epiconfig flags.
        No data manipulation — purely registry updates.
        Call before any merging or feature construction.
        """
        if self.epiconfig.feature_popsize:
            self.column_registration.add_column(
                'population_size', 'feature',
                needs_normalization=True, transformation_group='self'
            )

        if self.epiconfig.feature_popdens:
            self.column_registration.add_column(
                'population_density', 'feature',
                needs_normalization=True, transformation_group='self'
            )

        if self.epiconfig.feature_gisd:
            self.column_registration.add_column(
                'gisd_score', 'feature',
                needs_normalization=False
            )

        # popage, kreise_classes, borders: column names depend on the actual data
        # so registration happens during merge — see _merge_features()

    def _merge_features(self, df: pd.DataFrame, processed_data: 'ProcessedEpiData') -> pd.DataFrame:
        """
        Merge all optional feature dataframes into the main dataframe.
        Also registers columns whose names are data-dependent (popage, kreise, borders).
        """
        node_year_key = [self.epiconfig.id_column, 'year']
        node_key      = [self.epiconfig.id_column]

        if self.epiconfig.feature_popsize:
            df = df.merge(processed_data.population_size, on=node_year_key)

        if self.epiconfig.feature_popdens:
            df = df.merge(processed_data.population_density, on=node_year_key)

        if self.epiconfig.feature_gisd:
            df = df.merge(processed_data.gisd, on=node_year_key)

        if self.epiconfig.feature_popage:
            for col in processed_data.population_age.columns:
                if col not in ['year', self.epiconfig.id_column]:
                    self.column_registration.add_column(col, 'feature', needs_normalization=False)
            df = df.merge(processed_data.population_age, on=node_year_key)

        if self.epiconfig.feature_kreise_classes:
            for col in processed_data.kreise_classes.columns:
                if col != self.epiconfig.id_column:
                    self.column_registration.add_column(col, 'feature', needs_normalization=False)
            df = df.merge(processed_data.kreise_classes, on=node_key)

        if self.epiconfig.feature_borders:
            for col in processed_data.borders.columns:
                if col != self.epiconfig.id_column:
                    self.column_registration.add_column(col, 'feature', needs_normalization=False)
            df = df.merge(processed_data.borders, on=node_key)

        return df
    
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
        return dfc.reset_index(drop=True)
    
    def _drop_final_timesteps(self, df: pd.DataFrame) -> pd.DataFrame:
        """by shifting, we get NaNs for the final `horizon_leadtime` timesteps"""
        dfc = df.copy()
        # Drop only the last horizon_leadtime rows per node, not all NaN rows
        tail_mask = dfc.groupby(self.epiconfig.id_column).cumcount(ascending=False) < self.epiconfig.horizon_leadtime
        dfc = dfc[~tail_mask]        
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

    def orchestrate(self, processed_data: 'ProcessedEpiData') -> 'FeatureEpiData':
        time_start   = time.time()
        feature_data = processed_data.epidata.copy()

        # 1. register fixed-name features upfront
        self._register_static_features()

        # 2. merge all optional feature dataframes
        #    (data-dependent column names registered inside _merge_features)
        feature_data = self._merge_features(feature_data, processed_data)

        # 3. construct derived features — each method registers its own columns
        feature_data = self._add_time_index(feature_data)
        feature_data = self._lag_variable(feature_data)

        # 4. construct target
        feature_data = self._shift_target(feature_data)
        feature_data = self._drop_final_timesteps(feature_data)
        feature_data = self._rename_target(feature_data)

        # 6. finalise column order
        feature_data = self._reorder_df(feature_data)

        time_end = time.time()
        if self.epiconfig.verbose > 2:
            print(f'Execution of EpiFeatureBuilder took {round(time_end - time_start,3)}s')

        return FeatureEpiData(data=feature_data)