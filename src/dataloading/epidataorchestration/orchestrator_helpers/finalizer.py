import time
import json
from typing import TYPE_CHECKING
import pandas as pd
from ....utils.textformatting import checkmark

from ...epiconfig import EpiConfig

from ..containers import TransformedEpiData, FinalizedEpiData
from ..utils.normalization import reverse_log, reverse_minmax, reverse_zscore
from ...columnregistration import ColumnRegistry

class EpiDataFinalizer:
    """
    """   
    def __init__(self, 
                 epiconfig:             EpiConfig, 
                 column_registration:   ColumnRegistry):
        
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
        """Reverse all transformations in reverse order of application: norm first, then log."""
        dfc = normalized_df.copy()

        if not self.epiconfig.normalization_method:
            return dfc

        for col_entry in self.column_registration.columns:
            if not col_entry.transformation:
                continue

            if col_entry.column_name not in dfc.columns:
                continue

            if col_entry._transformation_group == 'self':
                params = col_entry._transformation_params
            else:
                ref    = self.column_registration.get_by_name(col_entry._transformation_group)
                params = ref._transformation_params

            if params is None:
                continue

            if params.zscore is not None:
                dfc = reverse_zscore(dfc, col_entry.column_name, params.zscore)
            elif params.minmax is not None:
                dfc = reverse_minmax(dfc, col_entry.column_name, params.minmax)

            if params.log is not None:
                dfc = reverse_log(dfc, col_entry.column_name, params.log)

        return dfc

    def orchestrate(self, normalized_data: TransformedEpiData) -> 'FinalizedEpiData':
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