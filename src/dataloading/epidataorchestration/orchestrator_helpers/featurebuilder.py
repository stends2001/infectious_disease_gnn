import time
import json
from typing import assert_never, TYPE_CHECKING, Dict
import pandas as pd
import geopandas as gpd
import numpy as np
from ....utils.textformatting import checkmark

if TYPE_CHECKING:
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

    def _add_delta_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute first difference of target column, store t-1 anchor for reversal."""
        col = self.epiconfig.target_column
        df[f'{col}_anchor'] = df.groupby(self.epiconfig.id_column)[col].shift(1)
        df[col] = df.groupby(self.epiconfig.id_column)[col].diff()

        return df.dropna().reset_index(drop=True)

    def orchestrate(self, processed_data: 'ProcessedEpiData') -> 'FeatureEpiData':
        time_start   = time.time()
        feature_data = processed_data.epidata.copy()

        if self.epiconfig.feature_popsize:
            self.column_registration.add_column(
                'population_size',
                'feature',
                needs_normalization=True,
                transformation_group='self'
            )            
            feature_data = pd.merge(feature_data, processed_data.population_size, on = [self.epiconfig.id_column, 'year'])   
                     
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
                f'gisd_score',
                'feature',
                needs_normalization=False
            )               
            feature_data = pd.merge(feature_data, processed_data.gisd, on = [self.epiconfig.id_column, 'year'])           

        if self.epiconfig.feature_popage:
            processed_feature_popage = processed_data.population_age

            for cc in processed_feature_popage.columns:
                if cc not in ['year',self.epiconfig.id_column]:
                    self.column_registration.add_column(
                        cc,
                        'feature',
                        needs_normalization=False
                    )         
                                          
            feature_data = pd.merge(feature_data, processed_feature_popage, on = [self.epiconfig.id_column, 'year'])               

        if self.epiconfig.feature_kreise_classes:
            for cc in processed_data.kreise_classes.columns:
                if cc != self.epiconfig.id_column:
                    self.column_registration.add_column(
                        cc,
                        'feature',
                        needs_normalization=False
                    )     

            feature_data = pd.merge(feature_data, processed_data.kreise_classes, on = self.epiconfig.id_column)  

        if self.epiconfig.feature_borders:
            for cc in processed_data.borders.columns:
                if cc != self.epiconfig.id_column:
                    self.column_registration.add_column(
                        cc,
                        'feature',
                        needs_normalization=False
                    )     
            feature_data = pd.merge(feature_data, processed_data.borders, on = self.epiconfig.id_column)                              

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
        feature_data = self._drop_final_timesteps(feature_data)
        feature_data = self._rename_target(feature_data)
        feature_data = self._reorder_df(feature_data)

        time_end = time.time()
        if self.epiconfig.verbose > 2:
            print(f'Execution of EpiFeatureBuilder took {round(time_end - time_start,3)}s')

        return FeatureEpiData(data=feature_data)