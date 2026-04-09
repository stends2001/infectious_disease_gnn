import time
from typing import TYPE_CHECKING, assert_never
import pandas as pd
import numpy as np
from src.utils.textformatting import checkmark

if TYPE_CHECKING:
    from src.dataloading.epiconfig.epiconfig import EpiConfig

from ..containers import FeatureEpiData, TransformedEpiData
from ...columnregistration import ColumnRegistry

from ..utils.temporal_summary import EpiDataTemporalSummary
from ..utils.normalization import pipeline_minmax_normalization, pipeline_zscore_normalization, apply_minmax_scaling, apply_zscore_scaling, apply_log
from ..utils.issues import EpiDataOrchestrationError

# ============= Transformer CLASS ============= 
class EpiDataTransformer:
    """  
    """      
    def __init__(self, 
                 epiconfig:             EpiConfig, 
                 column_registration:   ColumnRegistry, 
                 temporal_summary:      EpiDataTemporalSummary):
        
        self.epiconfig              = epiconfig 
        self.temporal_summary       = temporal_summary
        self.column_registration    = column_registration
        self.transformation_functions= {
            # two main types of transformations. Those that do not require parameters to be calculated (non-norm) and those that do.

            # non normalization
            'non_normalization' :   
                {'apply'    : {'log'   : apply_log}},

            # normalization
            'normalization' : 
                # round 1 => returning params
               {'pipeline': {'minmax': pipeline_minmax_normalization,   'zscore': pipeline_zscore_normalization},
                # round 2 => applying
                'apply':    {'minmax': apply_minmax_scaling,            'zscore': apply_zscore_scaling}}
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

    def _update_columnregistry_nonnormalization_transformations_params(self):
        """updates column registry to include nonnormalization trasnformations. Currently limited to log"""
        if self.epiconfig.log_transform:
            cols_to_log     = []

            transformation_dict = {'non_normalization' : {'log': {'shift': self.epiconfig.log_shift}}}

            for col in self.epiconfig.log_transform:

                # if col == future target then register log for target
                if col == self.epiconfig.target_column:
                    cols_to_log += ['target']
                    self.column_registration.update_transformation(
                        'target', 
                        transformation_dict
                    )         

                # if col == lag column then do the log for all lagged columns, but only register in transformation for lag0
                # not in combination with previous condition. If target == lag, the lag column follows the normalization of target
                elif col == self.epiconfig.lag_column:
                    cols_to_log += [f'{self.epiconfig.lag_column}_lag{lag}' for lag in range(0, self.epiconfig.lag_num)]
                    self.column_registration.update_transformation(
                        f'{self.epiconfig.lag_column}_lag0', 
                        transformation_dict
                    )                             


                # else register log column-specifically
                else:
                    cols_to_log += [col]
                    self.column_registration.update_transformation(
                        col, 
                        transformation_dict
                    )  

    def _apply_nonnormalization_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """at this point, only non - normalization transformations exist."""
        transformed_df = df.copy()

        for col_entry in self.column_registration.columns:

            match (col_entry.transformation, col_entry._transformation_group):
            
                # Skip columns that don't have normalization attribute
                case (False, _):
                    continue # continue with next col_entry
                  
                # Determine which normalization parameters to use
                # independent transformation first
                case (True, 'self'):
                
                    transformation_dict = col_entry._transformation_params

                    if self.epiconfig.verbose > 2:
                        print(f"{col_entry.column_name} normalized independently")
            
                # dependent transformation: expect a referral
                case (True, str()):

                    # Use reference column's normalization
                    ref_col_entry = self.column_registration.get_by_name(col_entry._transformation_group)
                    
                    transformation_dict = ref_col_entry._transformation_params

                    if self.epiconfig.verbose > 2:
                        print(f"{col_entry.column_name} normalized based on {ref_col_entry.column_name}")

                case _:
                    assert_never(col_entry.transformation, col_entry._transformation_group)
            
            # at this point we have a transformation dictionary => transformation parameters for this specific column.
            # may include normalization as well as other forms of transformation (log!)
            if transformation_dict is not None:
                
                transformation_params = transformation_dict['non_normalization']

                for transformation_func, params in transformation_params.items():
                    # transformation_func = 'log' ← correct
                    self.transformation_functions['non_normalization']['apply'][transformation_func]

                    # Apply normalization
                    transformed_df = self.transformation_functions['non_normalization']['apply'][transformation_func](
                        transformed_df, 
                        [col_entry.column_name], 
                        {col_entry.column_name : params}
                    )

        if self.epiconfig.verbose > 1:
            print(f'{checkmark} normalization applied')     

        return transformed_df
    
    def _update_columnregistry_norm_params(self, df: pd.DataFrame):
        """
        normalizes data and stores information in column_registration
        NOTE: not only the target is normalized based on the training data: that goes for all features!
        I haven't figured out if this is an issue, but I imagine it is not.
        For what it's worth, it's an easy fix.     
        """
        dfc             = df.copy()
        train_df        = dfc[dfc['train']]
        normalized_df   = dfc.copy()

        # if normalization_method is excplicity set to None then return df not-normalized
        if not self.epiconfig.normalization_method:
            return normalized_df

        elif self.epiconfig.normalization_method not in self.transformation_functions['normalization']['apply']:
            raise EpiDataOrchestrationError(f'No normalization function {self.epiconfig.normalization_method} found.')

        ### First pass ###
        # get all transformation parameters per group (.transformation_group = 'self')
        for col_entry in self.column_registration.columns:
            
            # Skip columns that don't have normalization attribute
            if not col_entry.transformation:
                continue # continue with next col_entry
            
            # Only calculate params for columns with independent normalization (normalization_group == 'self')
            if col_entry.transformation_group == 'self':
                _, norm_parameters = self.transformation_functions['normalization']['pipeline'][self.epiconfig.normalization_method](
                    train_df, 
                    [col_entry.column_name]
                )
                
                # Update transformation with normalization parameters
                self.column_registration.update_transformation(
                    col_entry.column_name,
                    {'normalization' : {f'{self.epiconfig.normalization_method}': norm_parameters[col_entry.column_name]}}
                )

        if self.epiconfig.verbose > 1:
            print(f'{checkmark} normalization parameters retrieved and stored')     

    def _apply_normalization_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """at this point, both non-normalization and normalization trasnformations exist. I filter out the former since they have been done."""
        normalized_df = df.copy()

        for col_entry in self.column_registration.columns:

            match (col_entry.transformation, col_entry._transformation_group):
            
                # Skip columns that don't have normalization attribute
                case (False, _):
                    continue # continue with next col_entry
                  
                # Determine which normalization parameters to use
                # independent transformation first
                case (True, 'self'):
                
                    transformation_dict = col_entry._transformation_params

                    if self.epiconfig.verbose > 2:
                        print(f"{col_entry.column_name} normalized independently")
            
                # dependent transformation: expect a referral
                case (True, str()):

                    # Use reference column's normalization
                    ref_col_entry = self.column_registration.get_by_name(col_entry._transformation_group)
                    
                    transformation_dict = ref_col_entry._transformation_params

                    if self.epiconfig.verbose > 2:
                        print(f"{col_entry.column_name} normalized based on {ref_col_entry.column_name}")

                case _:
                    assert_never(col_entry.transformation, col_entry._transformation_group)
            
            # at this point we have a transformation dictionary => transformation parameters for this specific column.
            # may include normalization as well as other forms of transformation (log!)

            if transformation_dict is not None:
                
                transformation_params = transformation_dict['normalization']

                for transformation_func, params in transformation_params.items():
                    # transformation_func = 'log' ← correct
                    self.transformation_functions['normalization']['apply'][transformation_func]

                    # Apply normalization
                    normalized_df = self.transformation_functions['normalization']['apply'][transformation_func](
                        normalized_df, 
                        [col_entry.column_name], 
                        {col_entry.column_name : params}
                    )

        if self.epiconfig.verbose > 1:
            print(f'{checkmark} normalization applied')     

        return normalized_df

    def orchestrate(self, feature_data: 'FeatureEpiData') -> 'TransformedEpiData':
        time_start      = time.time()

        split_data      = self._set_splits(feature_data.data.copy())

        # add log - params to column registry
        self._update_columnregistry_nonnormalization_transformations_params()    

        # apply log
        normalized_data = self._apply_nonnormalization_transformations(split_data)

        # add norm - params to column registry
        self._update_columnregistry_norm_params(normalized_data)  

        # apply normalization       
        normalized_data = self._apply_normalization_transformations(normalized_data)

        time_end = time.time()
        if self.epiconfig.verbose > 2:
            print(f'Execution of EpiNormalizer took {round(time_end - time_start,3)}s')        
        if self.epiconfig.verbose > 1:
            print("")
        return TransformedEpiData(data=normalized_data)
