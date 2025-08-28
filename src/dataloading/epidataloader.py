from typing import Literal, Optional, Tuple
import re 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .normalization import pipeline_normalization, apply_minmax_scaling, pipeline_zscore_normalization, apply_zscore_scaling
from tqdm import tqdm
from matplotlib.lines import Line2D
import os
import geopandas as gpd

import numpy as np
import pandas as pd

class EpiDataLoader:

    """ 
    Prepares data through:

    - adding features
    - tokenizing ID-columns
    - splitting and normalizing data

    Consistently returns self, with the exception of the preview function.

    Parameters:
    ----------
    disease_name: str
        name of survstat file to be openend
    data_env_dir: str
        path of the data - module on Wissdaten
    min_date: str
        first date to be included (>=)
    max_date: str
        first date to be excluded (<)
    aggr_level: Literal['03','02'] = '03'
        whether to aggregate onto bundeslander level      

    Examples:
    ---------

    >>> trial = EpiDataLoader('influenza', data_env, aggr_level= '02', min_date='2012-06-01',max_date='2020-06-01')
    >>> trial.add_time_features()
    >>> trial.log_transform_target()
    >>> trial.normalize('2018-06-01','2019-06-01','zscore')
    >>> trial.add_lagged_features(range(2,3))
    >>> trial.preview()    
        
    """

    def __init__(self, 
                 disease_name: str,
                 data_env_dir: str,
                 min_date:str     = '2001-01-01',
                 max_date:str     = '2020-06-01',
                 aggr_level: Literal['03','02'] = '03'
                 ):
        
        self.min_date           = pd.to_datetime(min_date)
        self.max_date           = pd.to_datetime(max_date)
        self.temporal_column    = 'timestamp'
        self.target_column      = 'incidence'
        self.id_column          = 'node' 

        self.disease            = disease_name
        self.data_env_dir       = data_env_dir
        self.aggr_level         = aggr_level

        raw_shapedata, raw_epidemiological_data = self._import_datasets()
        epidemiological_data                    = self._preprocess_epidemiological_data(raw_epidemiological_data)
        self.epidemiological_data, self.shapedata         = self._tokenize_id(epidemiological_data, raw_shapedata)

        pop_data = self.epidemiological_data.groupby(self.id_column)['population_size'].mean().reset_index(drop=False)
        self.population_by_node = pop_data

        self.feature_columns  = ['population_size']

        self.epidemiological_data         = self._filter_mindate(self.epidemiological_data,self.min_date)
        self.epidemiological_data         = self._filter_maxdate(self.epidemiological_data,self.max_date)

    def _import_datasets(self) -> Tuple[pd.DataFrame,pd.DataFrame]:
        """
        sets self.epidemiological data and self.shapedata 
        depending on self.aggr_level          
        """
        disease_data     = pd.read_csv(os.path.join(self.data_env_dir, f'processed/germany/epidemiology/casedata/survstat/{self.disease}.csv'), parse_dates = ['timestamp'], dtype = {'kz_kreis': str}).rename(columns = {'kz_kreis':'kz_03'})
        population_data  = pd.read_csv(os.path.join(self.data_env_dir, 'processed/germany/sociodemography/population_size_03.csv'), dtype = {'kz_2021': 'str'}).rename(columns = {'kz_2021':'kz_03'})
        shapedata        = gpd.read_file(os.path.join(self.data_env_dir, f'processed/germany/geospatial/shapefiles/shape_{self.aggr_level}.shp')).rename(columns={'kz':f'kz_{self.aggr_level}'}).drop(labels = ['level'],axis = 1)

        epidemiological_data = pd.merge(disease_data, population_data, on = ['kz_03','year'])

        return shapedata, epidemiological_data
    
    def _preprocess_epidemiological_data(self, raw_epidemiological_data) -> pd.DataFrame:
        """ 
        removes columns, sets incidence, and aggregates to BL level if necessary.
        """
        if self.aggr_level == '02':
            raw_epidemiological_data['kz_02'] = raw_epidemiological_data['kz_03'].str[:2]
            raw_epidemiological_data = raw_epidemiological_data.groupby(['timestamp','kz_02']).aggregate({'population_size':'sum', 'cases':'sum'}).reset_index()

        else:
            raw_epidemiological_data.drop(columns=['week','year'], inplace=True)

        raw_epidemiological_data['incidence'] = raw_epidemiological_data['cases'] / raw_epidemiological_data['population_size'] *10_000
        raw_epidemiological_data.drop(columns=['cases'], inplace=True)

        preprocessed_epidemiological_data = raw_epidemiological_data
        return preprocessed_epidemiological_data
    
    def _tokenize_id(self, epidemiological_data, shapedata) -> Tuple[pd.DataFrame,pd.DataFrame]:
        """
        tokenizes id column into node numbers
        """

        unique_ids = sorted(epidemiological_data[f'kz_{self.aggr_level}'].unique())
        id_idx = {}
        idx_id = {}

        for idx, id in enumerate(unique_ids):
            id_idx[id] = idx                  # id (kz_02 or kz_03) : node_id (zero based)
            idx_id[idx] = id                  # node_id (zero based): id (kz_02 or kz_03)

        self.tokens = {"id_idx": id_idx, "idx_id": idx_id} 

        shapedata[self.id_column]            = shapedata[f'kz_{self.aggr_level}'].map(id_idx)
        epidemiological_data[self.id_column] = epidemiological_data[f'kz_{self.aggr_level}'].map(id_idx)

        shapedata = shapedata.dropna()

        shapedata[self.id_column] =  shapedata[self.id_column].astype(int)
        epidemiological_data[self.id_column] =  epidemiological_data[self.id_column].astype(int)
        
        return epidemiological_data.drop(columns = [f'kz_{self.aggr_level}']), shapedata.drop(columns = [f'kz_{self.aggr_level}'])

    def _filter_mindate(self, df, min_date: str):
        dfc = df.copy()
        return dfc[dfc[self.temporal_column] >= min_date].reset_index(drop = True)                

    def _filter_maxdate(self, df, max_date: str):
        dfc = df.copy()
        return dfc[dfc[self.temporal_column] < max_date].reset_index(drop = True)                

    def add_time_features(self):
        """
        Adds classic cyclical time features (_sin and _cos) based on the temporal column.
        Works dynamically for weekly data.
        """
        dfc = self.epidemiological_data.copy()

        # Number of weeks per year (approximate)
        periods_per_year = 52

        # Extract the week number of the year from the temporal column (assuming datetime dtype)
        # Use pandas dt accessor for datetime
        t_of_year = dfc[self.temporal_column].dt.isocalendar().week

        # Column names
        sin_col = f'{self.temporal_column}_sin'
        cos_col = f'{self.temporal_column}_cos'

        # Compute sine and cosine transformation to encode cyclical nature of weeks in a year
        dfc[sin_col] = np.sin(2 * np.pi * t_of_year / periods_per_year)
        dfc[cos_col] = np.cos(2 * np.pi * t_of_year / periods_per_year)

        self.epidemiological_data = dfc

        self.feature_columns = self.feature_columns + [sin_col, cos_col]
        
        return self
    
    def normalize(self, 
                  split_trainval: str, 
                  split_valtest: str,
                  method : Literal['minmax','zscore'] = 'zscore'):
        
        """
        does the normalization of train/val/test separately
        """

        split_trainval = pd.to_datetime(split_trainval)
        split_valtest  = pd.to_datetime(split_valtest)

        self.split_trainval = split_trainval
        self.split_valtest  = split_valtest

        dataset = self.epidemiological_data.copy()

        XYt_trainval, XYt_test = self._split_XYt(dataset, split_valtest)
        XYt_train, XYt_val     = self._split_XYt(XYt_trainval, split_trainval)
        
        norm_columns = self.feature_columns + [self.target_column]


        if method == 'minmax':
            XYt_train_norm, norm_parameters = pipeline_normalization(XYt_train, norm_columns)
            XYt_val_norm                    = apply_minmax_scaling(XYt_val,     norm_columns, norm_parameters)
            XYt_test_norm                   = apply_minmax_scaling(XYt_test,    norm_columns, norm_parameters)

        if method == 'zscore':
            XYt_train_norm, norm_parameters = pipeline_zscore_normalization(XYt_train, norm_columns)
            XYt_val_norm                    = apply_zscore_scaling(XYt_val,     norm_columns, norm_parameters)
            XYt_test_norm                   = apply_zscore_scaling(XYt_test,    norm_columns, norm_parameters)


        self.XYt_train = XYt_train_norm
        self.XYt_val   = XYt_val_norm
        self.XYt_test  = XYt_test_norm

        self.epidemiological_data_normalized = pd.concat([XYt_train_norm,XYt_val_norm,XYt_test_norm])
        norm_parameters['preds'] = norm_parameters[self.target_column]
        self.norm_params = {'method': method, "params": norm_parameters}
        return self

    def _split_XYt(self, XYt, split_date):
        XYt_lower = XYt[XYt[self.temporal_column] < split_date].reset_index(drop=True)
        XYt_upper = XYt[XYt[self.temporal_column] >= split_date].reset_index(drop=True)
        return XYt_lower, XYt_upper

    def add_lagged_features(self, lags: range):
        if not hasattr(self, 'epidemiological_data_normalized'):
            raise AttributeError("The attribute 'epidemiological_data_normalized' is missing in this instance.\nNormalize first, then lag!")

        dfc = self.epidemiological_data_normalized.copy()

        new_features = []
        for lag in lags:
            feature = f'{self.target_column}_lag{lag}'
            dfc[feature] = dfc.groupby(self.id_column)[self.target_column].shift(lag)
            new_features.append(feature)

        self.feature_columns = self.feature_columns + new_features

        dfc = dfc.dropna()

        self.epidemiological_data_normalized = dfc

        # **Re-split the data into train/val/test so these include lagged features**
        self.XYt_trainval, self.XYt_test = self._split_XYt(dfc, self.split_valtest)
        self.XYt_train, self.XYt_val     = self._split_XYt(self.XYt_trainval, self.split_trainval)

        return self

    def preview(self,
                id: int = 1):
        """
        previews split and normalized data for a specific node, by default token 8.
        """

        traincolor = '#4a90d9'
        valcolor   = "#1b9e77"
        testcolor  = '#d94e4e'

        cases_train_aggr    = self.XYt_train.groupby(self.temporal_column)[self.target_column].sum().reset_index(drop = False)[self.target_column]
        cases_val_aggr      = self.XYt_val.groupby(self.temporal_column)[self.target_column].sum().reset_index(drop = False)[self.target_column]
        cases_test_aggr     = self.XYt_test.groupby(self.temporal_column)[self.target_column].sum().reset_index(drop = False)[self.target_column]   

        XYt_train = self.XYt_train[self.XYt_train[self.id_column] == id]
        XYt_val   = self.XYt_val[self.XYt_val[self.id_column] == id]
        XYt_test  = self.XYt_test[self.XYt_test[self.id_column] == id]

        time_axis_train     = XYt_train[self.temporal_column]
        time_axis_val       = XYt_val[self.temporal_column]
        time_axis_test      = XYt_test[self.temporal_column]

        cases_train         = XYt_train[self.target_column]
        cases_val           = XYt_val[self.target_column]
        cases_test          = XYt_test[self.target_column]

        fig, axes = plt.subplots(2,1, figsize = (22,6))
        axes      = axes.flatten()
        
        ax        = axes[0]
        ax.plot(time_axis_train, cases_train_aggr, color = traincolor,  label = 'train')
        ax.plot(time_axis_val, cases_val_aggr,color = valcolor,         label = 'val')
        ax.plot(time_axis_test, cases_test_aggr,color = testcolor,      label = 'test')
        ax.grid()
        ax.legend()
        ax.set_title(f'national')    

        ax = axes[1]
        axes      = axes.flatten()
        ax.plot(time_axis_train, cases_train, color = traincolor,  label = 'train')
        ax.plot(time_axis_val, cases_val,color = valcolor,         label = 'val')
        ax.plot(time_axis_test, cases_test,color = testcolor,      label = 'test')
        ax.grid()
        ax.legend()
        ax.set_title(f'{self.id_column}: {id}')    

        return fig, ax

    def log_transform_target(self, shift: float = 1) -> pd.DataFrame:
        """
        Applies log(x + shift) transform to specified columns to handle zeros.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe.
        columns : list
            List of column names to log-transform.
        shift : float
            Small constant added before log to avoid log(0).

        Returns:
        --------
        df_transformed : pd.DataFrame
            DataFrame with specified columns log-transformed.
        """
        df_transformed = self.epidemiological_data.copy()
        df_transformed[self.target_column] = np.log(df_transformed[self.target_column] + shift)
        self.epidemiological_data = df_transformed
        return self


    def __repr__(self):
        disease = getattr(self, 'disease', 'N/A')
        aggr_level = getattr(self, 'aggr_level', 'N/A')
        min_date = getattr(self, 'min_date', 'N/A')
        max_date = getattr(self, 'max_date', 'N/A')
        n_regions = len(self.tokens['id_idx']) if hasattr(self, 'tokens') else 'N/A'
        n_rows = len(self.epidemiological_data) if hasattr(self, 'epidemiological_data') else 'N/A'
        norm_status = 'Yes' if hasattr(self, 'epidemiological_data_normalized') else 'No'

        return (f"<EpiDataLoader(disease={disease}, aggr_level={aggr_level}, "
                f"date_range=({min_date.date() if min_date != 'N/A' else 'N/A'} - {max_date.date() if max_date != 'N/A' else 'N/A'}), "
                f"regions={n_regions}, data_rows={n_rows}, normalized={norm_status})>")
