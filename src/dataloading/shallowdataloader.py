import numpy as np
import pandas as pd
from tqdm import tqdm

import pandas as pd



from ..utils.textformatting import warning_emoji
from .epidataloader import EpiDataLoader
from typing import Literal, Union, List, Dict

class ShallowDataLoader(EpiDataLoader):
    """
    """
    def __init__(self, 
                 disease_name: str,
                 data_env_dir: str,
                 min_date:     str     = '2001-01-01',
                 max_date:     str     = '2025-01-01',
                 nuts_level:   Literal['nuts1','nuts2','nuts3'] = 'nuts3',
                 include_population: bool = False,
                 horizon_size: int     = 1,
                 horizon_leadtime:int  = 1,
                 sequence_length: int  = 1,
                 split_berlin: bool    = True,
                 verbose: bool         = True):
        self.task_config = {}

        super().__init__(disease_name, data_env_dir, min_date, max_date, nuts_level, include_population, horizon_size, horizon_leadtime, sequence_length, split_berlin, verbose)

        self.dataloader_main, self.dataloader_train, self.dataloader_val, self.dataloader_test = {}, {}, {}, {}

    def construct_dataloaders(self):

        if self.sequence_length > 1:
            print(f'{warning_emoji} Please note that currently sequence length for shallowdataloaders is not yet implemented')

        main_data = self.data['final']
        X_col_selection     = [self.id_column] + self.feature_columns

        for hh in self.target_horizons:
            y_col_selection = [hh]
            

            main            = main_data[[self.temporal_column] + X_col_selection + y_col_selection + self.split_columns]
            X_train, y_train= main_data[main_data['train']][X_col_selection].reset_index(),  main_data[main_data['train']][y_col_selection]
            X_val, y_val    = main_data[main_data['val']][X_col_selection].reset_index(),  main_data[main_data['val']][y_col_selection]
            X_test, y_test  = main_data[main_data['test']][X_col_selection].reset_index(),  main_data[main_data['test']][y_col_selection]

            self.dataloader_train[hh]   = {'X': X_train, 'y': y_train}
            self.dataloader_val[hh]     = {'X': X_val, 'y': y_val}
            self.dataloader_test[hh]    = {'X': X_test, 'y': y_test}
            self.dataloader_main[hh]    = main