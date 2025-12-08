import os
from typing import Optional, Tuple, List, Union, Literal, Dict
from collections.abc import Iterable
import pandas as pd
from io import StringIO
from ..utils import get_data_env
from ..utils.textformatting import checkmark
from tqdm import tqdm

dir_commuting_raw   = os.path.join(get_data_env(),'raw/germany/mobility/commuter_data/auspendler/')
dir_commuting_pcd   = os.path.join(get_data_env(),'processed/germany/mobility/commuter_data/')
dir_harmfile        = os.path.join(get_data_env(),'processed/germany/geospatial/harmonization/german_nuts_harmonization.tsv')

class CommuterDataProcessor:
    """
    Processor of commuter-data
    
    Returns single pd.DataFrame with the number of commuters between two Kreisen.

    Examples
    --------
    >>> processor = CommuterDataProcessor()
    >>> processor.process_data(years = range(2002,2025))
    >>> processor.save_data() 
    >>> # returns "✓ data loaded"
    """
    def __init__(self):
        self.harmfile       = pd.read_csv(dir_harmfile, sep ="\t", dtype=str)
        self.rename_cols    = {'Regionalschlüssel'  : 'nuts3_work',
                               'Regionalschlüssel.1': 'nuts3_residence',
                               'Insgesamt'          : 'commuters'}
        self.columns        = list(self.rename_cols.values())
        self.data           = None

    def process_data(self, years: Iterable[Union[int,str]]):
        """ 
        Loop over all years (folder) to return a merged df.

        Parameters
        ----------
        years: Union[List[str], str]:
            list of years over which to loop

        Returns
        -------
        pd.DataFrame   
            df with columns: 'nuts3_work', 'nuts3_residence','commuters','year'
        """
        # make years iterable if it isn't
        if isinstance(years, str):
            years = [years]

        # looping over years
        for ii, yy in tqdm(enumerate(years), desc = 'processing raw commuter data year - collections', disable=False):
            # get concatenated df
            yearly_df = self._concatenate_yearly_data(str(yy))
            if ii == 0:
                all_data = yearly_df
            else:
                all_data = pd.concat([all_data, yearly_df], ignore_index=True)  # type: ignore => all_data will not be unbound

        self.data = all_data # type: ignore
        print(f'{checkmark} data processed for {[yy for yy in years]}')

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
            filtered_csv=filtered_csv[filtered_csv['nuts3_residence'].isin(list(self.harmfile['nuts3'].unique()))]

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
    """Simple dataloader object to return commuting_data.

    Parameters
    ----------
    years : Union[List[str], str]
        The years for which to select data.
    """

    def __init__(self, years: Union[List[str], str]):
        if isinstance(years, str):
            years = [years]

        # Normalize all years to strings in one go
        self.years = [str(y) for y in years]

        self.data = None

    def return_data(self) -> pd.DataFrame:
        self.data = self._import_data()
        return self.data

    def _import_data(self) -> pd.DataFrame:
        """Imports and returns the dataframe of selected years."""
        df = pd.read_csv(
            os.path.join(dir_commuting_pcd, 'commuting_data.csv'),
            dtype={'nuts3_work': 'str', 'nuts3_residence': 'str', 'year': 'str'}
        )
        df = df[df['year'].isin(self.years)]
        print(f"{checkmark} data loaded")
        return df

    def __repr__(self) -> str:
        if self.data is not None:
            return f"<CommuterDataLoader(data={len(self.data)} rows, years={self.years})>"
        return f"<CommuterDataLoader(no data loaded, years={self.years})>"
