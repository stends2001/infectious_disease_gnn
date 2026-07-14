import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...epidataorchestration.orchestrator import EpiDataOrchestrator

class BaseLineDataLoaderManager:
    """
    Dataloader for baselinemodels, based on 
    dataorchestrator.data_final.data_denorm

    runs by itself from __init__, no need to call any method

    Parameters:
    -----------
    dataorchestrator: 'EpiDataOrchestrator'
    """
    def __init__(self, 
                 dataorchestrator: 'EpiDataOrchestrator'):
        
        self.dataorchestrator     = dataorchestrator
        self.column_registration  = dataorchestrator.column_registration
        self._construct_dataloaders()
        
    def _construct_dataloaders(self):
        """writes the dataloader. main functionality of this class"""
        main_data           = self.dataorchestrator.data_final.data_denorm.copy()
        
        split_colnames      = self.dataorchestrator.column_registration.get_entries_names_by_type('split')
        time_colname        = self.dataorchestrator.config.temporal_column
        id_colname          = self.dataorchestrator.config.id_column

        if self.dataorchestrator.config.horizon_size > 1:
            raise NotImplementedError('BaseLineDataLoaderManager currently only works with horizon size == 1')
        
        # Get target from the first horizon
        base_lead           = self.dataorchestrator.config.horizon_leadtime
        target_colname      = f'target_lead{base_lead}'
        
        # Get the split columns from normalized data (splits are the same)
        timesplits          = self.dataorchestrator.data_final.data[[time_colname] + split_colnames].drop_duplicates().reset_index(drop=True)
        
        # Extract what we need from original data
        main_data_selection = main_data[[time_colname, id_colname, target_colname]]
        main_data_selection = main_data_selection.rename(columns={target_colname: 'target'})
        
        # Merge with splits
        main_data_selection = pd.merge(main_data_selection, timesplits, on='timestamp')

        if self.dataorchestrator.config.prediction_mode == 'classification':
            main_data_selection.loc[main_data_selection['target'] > 0, 'target'] = 1       

        self.dataloader_main = main_data_selection

    def __repr__(self):
        return (f"<BaseLineDataLoaderManager(dataloader at .dataloader_main)>")            