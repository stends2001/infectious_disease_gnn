from ...dataorchestration.dataorchestrator import DataOrchestrator
import pandas as pd

class BaseLineDataLoaderManager:
    """
    """
    def __init__(self, 
                 dataorchestrator: DataOrchestrator):
        
        self.dataorchestrator     = dataorchestrator
        self.column_registration  = dataorchestrator.column_registration

        self._construct_dataloaders()
        
    def _construct_dataloaders(self):
        main_data           = self.dataorchestrator.data_processed.data.copy()
        split_colnames      = self.dataorchestrator.column_registration.get_by_type('split')
        time_colname        = self.dataorchestrator.config.temporal_column
        id_colname          = self.dataorchestrator.config.id_column
        target_colname      = self.dataorchestrator.config.target_column
        timesplits          = self.dataorchestrator.data_normalized.data[[time_colname] + split_colnames].drop_duplicates().reset_index(drop = True)
        main_data_selection = main_data[[time_colname, id_colname,target_colname]].rename(columns = {target_colname:'target'})
        self.dataloader_collection      = pd.merge(main_data_selection, timesplits, on = 'timestamp')     


    def __repr__(self):
        return (f"<BaseLineDataLoaderManager(dataloader at .dataloader_collections)>")            