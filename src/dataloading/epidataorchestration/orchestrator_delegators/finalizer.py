import time
import json
from typing import TYPE_CHECKING
import pandas as pd
from ....utils.textformatting import checkmark

if TYPE_CHECKING:
    from ...epiconfig import EpiConfig

from ..epidatacontainers import NormalizedEpiData, FinalizedEpiData
from ..utils.normalization import reverse_log, reverse_minmax_scaling, reverse_zscore_scaling
from ...columnregistration.column_registry import ColumnRegistration

class EpiDataFinalizer:
    """
    """   
    def __init__(self, 
                 epiconfig: 'EpiConfig', 
                 column_registration: ColumnRegistration):
        self.epiconfig = epiconfig 
        self.column_registration = column_registration

    def _create_pred_col_entry(self):
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
                        transformation_dict = col_entry.transformation_params
                
                    # if referral-based normalization
                    else:
                        reference_entry = self.column_registration.get_by_name(col_entry.transformation_group)
                        transformation_dict = reference_entry.transformation_params
                
                    # Reverse normalization

                    for transformation_type, type_params in reversed(transformation_dict.items()):

                        if transformation_type == 'normalization':
                            for norm_method, norm_params in type_params.items():
                                if norm_method == 'minmax':
                                    dfc = reverse_minmax_scaling(dfc, norm_params, column=col_entry.column_name)
                                elif norm_method == 'zscore':
                                    dfc = reverse_zscore_scaling(dfc, norm_params, column=col_entry.column_name)

                        elif transformation_type == 'non_normalization':
                            for transf_method, transf_params in type_params.items():                            
                                if transf_method  == 'log':
                                    dfc = reverse_log(dfc, transf_params['shift'], column=col_entry.column_name)       

            return dfc

    def orchestrate(self, normalized_data: NormalizedEpiData) -> 'FinalizedEpiData':
        time_start = time.time()
        dfc         = normalized_data.data

        dfc         = self._add_horizons(dfc)
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