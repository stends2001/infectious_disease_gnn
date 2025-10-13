import os
from typing import Optional, Tuple, List, Union, Literal, Dict
import pandas as pd
from io import StringIO
from ..utils import get_data_env

class PendlerDatenProcessor:
    """
    >>> graphobject = GraphConstructor(epidata=epidata_nuts3)
    >>> # graphobject.preview_shape_object()
    >>> graphobject.generate_graph(method = 'commuter',
    >>>                        self_connection='mean',
    >>>                        commuter_type = 'static',
    >>>                        commuting_threshold = 1_500,
    >>>                        scaling_method='rowwise',
    >>>                        name_addition='t1500')
    >>> # graphobject.preview_graph(graphname = 'commuter_t1500_selfmean_rowwis', node_idx = 26, qualitative= False)
    >>> graphobject.generate_graph(method = 'commuter',
    >>>                         self_connection='mean',
    >>>                         commuter_type = 'static',
    >>>                         commuting_threshold = 1_000,
    >>>                        scaling_method='rowwise',
    >>>                         name_addition='t1500')
    >>> graphobject.generate_graph(method = 'identity')
    >>> graphobject.generate_graph(method = 'boolean_neighbors',
    >>>                         self_connection='mean')
    >>> graphobject.save_graph(graphname = 'all')
    """
    def __init__(self, 
                 raw_folder_path: str,
                 processed_folder_path: str):
        
        self.raw_folder_path = raw_folder_path
        self.processed_folder_path = processed_folder_path

        self.harmfile = pd.read_csv(os.path.join(get_data_env(),'processed/germany/geospatial/harmonization/german_nuts_harmonization.tsv'), sep ="\t", dtype=str)

        self.dtypes = {'Regionalschlüssel'  : 'str', 
                       'Regionalschlüssel.1': 'str'}
        
        self.rename_cols = {'Regionalschlüssel'  :'nuts3_work',
                            'Regionalschlüssel.1':'nuts3_residence',
                            'Insgesamt'          : 'commuters'}
        
        self.columns     = list(self.rename_cols.values())
        self.data = {}

    def import_raw_data(self,
                        year: str):
        
        rawfolder = os.path.join(self.raw_folder_path, year)

        all_data = []  # to accumulate all processed DataFrames

        for file in os.listdir(rawfolder):  # iterate files in the folder
            path = os.path.join(rawfolder, file)


            trimmed_data = clean_csv_file(path)

            if trimmed_data is None:
                raise ImportError(f'No data found inside {path}')

            trimmed_csv = pd.read_csv(trimmed_data, sep=";", dtype=self.dtypes).rename(columns=self.rename_cols)
            trimmed_csv = trimmed_csv[self.columns]
            
            filtered_csv= trimmed_csv[trimmed_csv['nuts3_work'].isin(list(self.harmfile['nuts3'].unique()))]
            filtered_csv=filtered_csv[trimmed_csv['nuts3_residence'].isin(list(self.harmfile['nuts3'].unique()))]

            all_data.append(filtered_csv)  # append the processed dataframe

        self.data[year] = pd.concat(all_data, ignore_index=True)
        

        return self
    

def clean_csv_file(path: str) -> Optional[StringIO]:

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    start_index = next((i for i, line in enumerate(lines) if line.count(';') > 3), None)
        
    if start_index is not None:
        # Slice and remove trailing garbage lines
        trimmed_lines = lines[start_index:]
        valid_lines   = [line for line in trimmed_lines if line.count(';') > 3]

        cleaned_lines = [line.replace('.','') for line in valid_lines]

        # Create a temporary in-memory CSV
        return StringIO(''.join(cleaned_lines))

    else:
        return None

