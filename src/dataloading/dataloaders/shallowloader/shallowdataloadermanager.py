import pandas as pd
import pandas as pd
from typing import Literal, Union, List, Dict
from dataclasses import dataclass

from ...dataorchestration.dataorchestrator import DataOrchestrator
from ....utils.textformatting import warning_emoji, checkmark

@dataclass
class ShallowDataLoader:
    """ 
    Stores a single dataloader for shallow models
    """    
    X: pd.DataFrame
    y: pd.DataFrame

    def __post_init__(self):
            # Validate that X and y have the same number of rows
            if len(self.X) != len(self.y):
                raise ValueError(f"X and y have mismatched lengths: X has {len(self.X)} rows, but y has {len(self.y)} rows.")
            
            if len(self.X) == 0:
                raise ValueError(f"No data found. Please check the timestamps used.")

    def __repr__(self):
        return (f"<ShallowDataLoader(X: {len(self.X)} rows, {len(self.X.columns)} cols, " 
                f"y: {len(self.y)} rows)>")     

@dataclass
class ShallowDataLoaderCollection:
    """ 
    Stores all DataLoaders for shallow models

    See Also:
    --------
    ShallowDataLoader
    """
    train:              'ShallowDataLoader'
    val:                'ShallowDataLoader'   
    test:               'ShallowDataLoader'
    main:               pd.DataFrame

    def __repr__(self):
        return (f"<ShallowDataLoaderCollection(train, val, test, main)>") 

class ShallowDataLoaderManager:
    """
    A manager of multiple collectiosn of dataloaders.
    Each horizon has a specified dataloader collection, which in turn 
    contains 3 dataloaders (t/v/t/) and a pd.dataframe (main = t+v+t).

    Parameters:
    ----------
    dataorchestrator: DataOrchestrator
        The object with entirely build dataset.

    Attributes:
    ----------
    dataloader_collections: Dict['str', 'ShallowDataLoaderCollection]
        instance of ShallowDataLoaderCollection per horizon


    Examples:
    --------
    >>> data_orchestrator = DataOrchestrator(config).build()
    >>> shallowdata       = ShallowDataLoaderManager(data_orchestrator).construct_dataloaders()
    """
    def __init__(self, 
                 dataorchestrator: DataOrchestrator):
        
        self.dataorchestrator     = dataorchestrator
        self.column_registration  = dataorchestrator.column_registration

    def construct_dataloaders(self):
        """
        create dataloader collection classes and store it in `self.dataloader_collections`.
        """
        main_data           = self.dataorchestrator.data_final.data.copy()
        X_col_selection     = [self.dataorchestrator.config.id_column]+self.column_registration.get_by_type('feature')
        self.dataloader_collections    = {}

        for hh in range(self.dataorchestrator.config.horizon_size):
            horizon_name    = f'horizon_{hh}'
            steps_ahead     = int(hh+self.dataorchestrator.config.horizon_leadtime)
            y_col_selection = f'target_lead{steps_ahead}'
            
            main            = main_data[[self.dataorchestrator.config.temporal_column] + X_col_selection + [y_col_selection] + self.column_registration.get_by_type("split")]
            main            = main.rename(columns = {y_col_selection:'target'}).dropna()

            X_train, y_train= main[main['train']][X_col_selection].reset_index(drop = True),  main[main['train']][['target']]
            X_val, y_val    = main[main['val']][X_col_selection].reset_index(drop = True),    main[main['val']][['target']]
            X_test, y_test  = main[main['test']][X_col_selection].reset_index(drop = True),   main[main['test']][['target']]

            dataloader  = ShallowDataLoaderCollection(
                train = ShallowDataLoader(X_train, y_train),
                val   = ShallowDataLoader(X_val, y_val),
                test  = ShallowDataLoader(X_test, y_test),
                main  = main)

            self.dataloader_collections[horizon_name] = dataloader

            if self.dataorchestrator.config.verbose:
                print(f'{checkmark} Dataloader object created for {horizon_name}')

        return self
    
    def __repr__(self):
        return (f"<ShallowDataLoaderManager(dataloaders for {self.dataorchestrator.config.horizon_size} horizons in self.dataloader_collections)>")            