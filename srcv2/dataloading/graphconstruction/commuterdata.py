import os
from typing import Optional, Tuple, List, Union, Literal, Dict
import pandas as pd
from io import StringIO
from ...utils import get_data_env
from ...utils.textformatting import checkmark

dir_commuting_raw   = os.path.join(get_data_env(),'raw/germany/mobility/commuter_data/auspendler/')
dir_commuting_pcd   = os.path.join(get_data_env(),'processed/germany/mobility/commuter_data/')
dir_harmfile        = os.path.join(get_data_env(),'processed/germany/geospatial/harmonization/german_nuts_harmonization.tsv')


class CommuterDataProcessor:
    """
    Processor of commuter-data
    
    Returns single pd.DataFrame with the number of commuters between two Kreisen.

    Examples:
    --------
    >>> pr = CommuterDataProcessor()
    >>> # currently only 2024 is available!
    >>> pr.process_data('2024')
    """
    def __init__(self):
        self.harmfile       = pd.read_csv(dir_harmfile, sep ="\t", dtype=str)
        self.rename_cols    = {'Regionalschlüssel'  : 'nuts3_work',
                               'Regionalschlüssel.1': 'nuts3_residence',
                               'Insgesamt'          : 'commuters'}
        self.columns        = list(self.rename_cols.values())
        self.data           = None

    def process_data(self, years: Union[List[str], str]):
        """ 
        Loop over all years (folder) to return a merged df.

        Parameters:
        ----------
        years: Union[List[str], str]:
            list of years over which to loop

        Returns:
        -------
        pd.DataFrame   
            df with columns: 'nuts3_work', 'nuts3_residence','commuters','year'
        """
        # make years iterable if it isn't
        if isinstance(years, str):
            years = [years]

        # looping over years
        for ii, yy in enumerate(years):
            # get concatenated df
            yearly_df = self._concatenate_yearly_data(yy)
            if ii == 0:
                all_data = yearly_df
            else:
                all_data = pd.concat([all_data, yearly_df], ignore_index=True)  # type: ignore => all_data will not be unbound

        self.data = all_data # type: ignore
        print(f'{checkmark} data processed for {years}')

    def save_data(self):
        """ 
        saves self.data
        """
        if self.data is not None:
            self.data.to_csv(os.path.join(dir_commuting_pcd,'commuting_data.csv'), index=False)
            print(f'{checkmark} data saved ')

    def _concatenate_yearly_data(self, year: str) -> pd.DataFrame:
        """ 
        clean and concatenate all datafiles for a year
        """
        rawfolder = os.path.join(dir_commuting_raw, year)
        all_data = []  # to accumulate all processed DataFrames

        for file in os.listdir(rawfolder):  # iterate files in the folder
            
            path        = os.path.join(rawfolder, file)
            trimmed_data= clean_csv_file(path)

            if trimmed_data is None:
                raise ImportError(f'No data found inside {path}')

            trimmed_csv = pd.read_csv(trimmed_data, sep=";", dtype={'Regionalschlüssel'  : 'str', 'Regionalschlüssel.1': 'str'}).rename(columns=self.rename_cols)
            trimmed_csv = trimmed_csv[self.columns]
          
            filtered_csv= trimmed_csv[trimmed_csv['nuts3_work'].isin(list(self.harmfile['nuts3'].unique()))]
            filtered_csv=filtered_csv[trimmed_csv['nuts3_residence'].isin(list(self.harmfile['nuts3'].unique()))]

            all_data.append(filtered_csv)  # append the processed dataframe

        df          = pd.concat(all_data, ignore_index=True)
        df['year']  = year
        return df
    
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


class CommuterDataLoader:
    """ 
    Simple dataloader object to return commuting_data

    Parameters:
    ----------
    years: Union[List[str], str]
        the years for which to select data
    """
    def __init__(self, years: Union[List[str],str]):
        if isinstance(years, str):
            years = [years]
        self.years = years

    def import_data(self) -> pd.DataFrame:
        """imports and returns the dataframe of selected years"""
        df = pd.read_csv(os.path.join(dir_commuting_pcd, 'commuting_data.csv'), dtype={'nuts3_work':'str','nuts3_residence':'str','year':'str'})
        df = df[df['year'].isin(self.years)]
        return df

