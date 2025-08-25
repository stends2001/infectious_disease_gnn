from typing import Literal, Optional 
import re 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .normalization import pipeline_normalization, apply_minmax_scaling, pipeline_zscore_normalization, apply_zscore_scaling
from tqdm import tqdm
from matplotlib.lines import Line2D
import os
import geopandas as gpd

class EpiDataLoader:

    """ 
    Prepares data through:
    - adding features
    - tokenizing ID-columns
    - splitting and normalizing data

    Consistently returns self, with the exception of the preview function.

    Parameters:
    ----------
    dataset: EpiDataPrepper
        pre-processed epidemiological dataset
    """

    def __init__(self, 
                 disease_name: str,
                 data_env_dir: str,
                 min_date = '2001-01-01',
                 max_date = '2020-06-01'
                 ):
        self.min_date           = pd.to_datetime(min_date)
        self.max_date           = pd.to_datetime(max_date)
        self.temporal_column    = 'timestamp'
        self.target_column      = 'incidence'
        self.id_column          = 'id' 
        self.data_env_dir       = data_env_dir
        
        self.initial_df = self._import_preprocess(disease_name)

        self.feature_columns  = []

        self.df         = self._filter_mindate(self.initial_df,self.min_date)
        self.df         = self._filter_maxdate(self.df,self.max_date)

    def _import_preprocess(self, disease_name):
        disease_data     = pd.read_csv(os.path.join(self.data_env_dir, f'processed/germany/epidemiology/casedata/survstat/{disease_name}.csv'), parse_dates = ['timestamp'], dtype = {'kz_kreis': str})
        population_data  = pd.read_csv(os.path.join(self.data_env_dir, 'processed/germany/sociodemography/population_size_03.csv'), dtype = {'kz_2021': 'str'}).rename(columns = {'kz_2021':'kz_kreis'})
        incidence_data   = pd.merge(disease_data, population_data, on = ['kz_kreis','year'])
        incidence_data['incidence']   = incidence_data['cases'] / incidence_data['population_size'] * 100_000
        incidence_data   = incidence_data.drop(labels = ['week','cases','year'], axis = 1)

        unique_ids = sorted(incidence_data['kz_kreis'].unique())
        id_idx = {}
        idx_id = {}
        for idx, id in enumerate(unique_ids):
            id_idx[id] = idx
            idx_id[idx] = id

        tokenization = {"id_idx": id_idx,
                        "idx_id": idx_id}           
        shapes = gpd.read_file(os.path.join(self.data_env_dir, 'processed/germany/geospatial/shapefiles/shape_03.shp')).rename(columns = {'kz':'kz_kreis'}).drop(labels = ['level'],axis = 1)
        shapes['id'] = shapes['kz_kreis'].map(tokenization['id_idx']).astype('Int64')
        incidence_data = gpd.GeoDataFrame(pd.merge(incidence_data,shapes, on = 'kz_kreis', how = 'left'))
        incidence_data       = incidence_data.drop(labels = ['kz_kreis'], axis = 1)

        self.tokens = tokenization
        return incidence_data

    def _filter_mindate(self, df, min_date: str):
        dfc = df.copy()
        return dfc[dfc[self.temporal_column] >= min_date]                

    def _filter_maxdate(self, df, max_date: str):
        dfc = df.copy()
        return dfc[dfc[self.temporal_column] < max_date]                

    def add_time_features(self):
        """
        Adds classic cyclical time features (_sin and _cos) based on the temporal column.
        Works dynamically for weekly data.
        """
        dfc = self.df.copy()

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

        self.df = dfc

        self.feature_columns = self.feature_columns + [sin_col, cos_col]
        
        return self
    
    def normalize(self, 
                  split_trainval, 
                  split_valtest,
                  method = 'minmax'):
        split_trainval = pd.to_datetime(split_trainval)
        split_valtest  = pd.to_datetime(split_valtest)

        self.split_trainval = split_trainval
        self.split_valtest  = split_valtest

        dataset = self.df.copy()

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
        self.XYt_val = XYt_val_norm
        self.XYt_test = XYt_test_norm

        self.df = pd.concat([XYt_train_norm,XYt_val_norm,XYt_test_norm])
        norm_parameters['preds'] = norm_parameters[self.target_column]
        self.norm_params = {'method': method, "params": norm_parameters}
        return self

    def split_data(self):

        """
        splits data in training, validation and testing data using splitting dates.
        also normalizes the data, and saves the data in X, Y and t separately.
        """
        dataset = self.df.copy()

        XYt_trainval_norm, XYt_test_norm = self._split_XYt(dataset, self.split_valtest)
        XYt_train_norm, XYt_val_norm     = self._split_XYt(XYt_trainval_norm, self.split_trainval)


        self.XYt_train = XYt_train_norm
        self.XYt_val = XYt_val_norm
        self.XYt_test = XYt_test_norm
        return self

    def _split_XYt(self, XYt, split_date):
        XYt_lower = XYt[XYt[self.temporal_column] < split_date].reset_index(drop=True)
        XYt_upper = XYt[XYt[self.temporal_column] >= split_date].reset_index(drop=True)
        return XYt_lower, XYt_upper

    def add_lagged_features(self, 
                            lags):
        

        dfc = self.df.copy()
        new_features = []
        for lag in lags:
            feature = f'{self.target_column}_lag{lag}'
            dfc[feature] = dfc.groupby(self.id_column)[self.target_column].shift(lag)
            new_features.append(feature)
        
        self.feature_columns = self.feature_columns + new_features

        dfc = dfc.dropna()

        self.df = dfc
        return self

    def preview(self,
                id: int = 391):
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

        fig, axes = plt.subplots(3,1, figsize = (22,6), gridspec_kw={'height_ratios': [2, 2, 1]})
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



        ax = axes[2]

        colors = {
            'train': traincolor,
            'validation': valcolor,
            'test': testcolor
        }

        datasets = {
            'train': XYt_train,
            'validation': XYt_val,
            'test': XYt_test
        }

        # Linestyle per feature, opacity fixed to 1.0
        feature_styles = {
            'timestamp_sin': {'linestyle': '-'},    # solid
            'timestamp_cos': {'linestyle': '--'},   # dashed
        }

        # Plot lines with split color and linestyle, opacity = 1
        for split, dataset in datasets.items():
            color = colors[split]
            for key, style in feature_styles.items():
                y = dataset[key]
                ax.plot(dataset[self.temporal_column], y,
                        color=color,
                        alpha=1.0,
                        linestyle=style['linestyle'],
                        linewidth=0.8)

        # Legend for features (grey lines with linestyle)
        grey_color = 'grey'
        legend_handles_features = [
            Line2D([0], [0], color=grey_color, linestyle=style['linestyle'], linewidth=3,
                alpha=1.0, label=feature)
            for feature, style in feature_styles.items()
        ]

        ax.legend(handles=legend_handles_features, title='Features', loc='upper left')

        ax.grid()


        return fig, ax

    def copy(self):
        self.XYt_train = self.XYt_train.copy()
        self.XYt_val = self.XYt_val.copy()
        self.XYt_test = self.XYt_test.copy()
        return self