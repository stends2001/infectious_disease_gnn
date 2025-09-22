import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Dict, List, Literal, Any, Union
import torch
from ..dataloading import EpiDataLoader
from ..metrics.losses import spike_weighted_mse, mse, spike_detection_loss, temporal_smoothness_loss, spatial_consistency_loss
from ..dataloading.normalization import reverse_zscore_scaling, reverse_log
from ..utils.constants import traincolor, valcolor, testcolor
import geopandas as gpd

class BaseModel:

    """
    Parent class for all models

    DeepLearningModelCore builds off of this by inheritance

    TODO: de-normalize predictions    
    """

    def __init__(self, 
                 dataloader: EpiDataLoader, 
                 name:       Optional[str] = None):
        
        self.dataloader = dataloader
        self.name       = name if name else "unknown"    
        self.evaluation_datasets = {} 
        self.model_color= None

    def forecast(self):
        """supposed to create the attribute `evaluation_df`"""
        raise NotImplementedError("Each model must implement its own forecast method.")

    def denorm_predictions(self):
        print('denormalizing predictions not implemented yet!!!!')
        return self
    
    def show_forecasts(self,
                       dataset: Literal['train','val','test'],
                       node_idx: Union[List[int], int] = 1,
                       timeframe: Optional[List[str]] = None):

        evaluation_df = self.evaluation_datasets[dataset]

        if isinstance(node_idx, int):
            node_idx = [node_idx]
        n_plots = len(node_idx)

        fig, axes = plt.subplots(n_plots+1 ,1, figsize = (16,3 * n_plots))
        axes      = axes.flatten()

        if timeframe:
            date0 = timeframe[0]
            date1 = timeframe[1]

            evaluation_df = evaluation_df[evaluation_df[self.dataloader.temporal_column]>= date0]
            evaluation_df = evaluation_df[evaluation_df[self.dataloader.temporal_column]< date1] 


        evaluation_df_aggr= evaluation_df.copy().groupby(self.dataloader.temporal_column).agg({'incidence': 'sum','pred': 'sum'})
        ax = axes[0]
        sns.lineplot(data=evaluation_df_aggr, x=self.dataloader.temporal_column, y=self.dataloader.target_column, color=testcolor, marker = "o",           ax=ax)
        sns.lineplot(data=evaluation_df_aggr, x=self.dataloader.temporal_column, y='pred', color=self.model_color, markeredgecolor='black', marker = "x",  ax=ax)
        ax.set_title(f'National aggregation of incidence')
        ax.set_xlabel("")            
        ax.grid()          


        for counter, id  in enumerate(node_idx):

            ax = axes[counter + 1]

            df_node = evaluation_df[evaluation_df[self.dataloader.id_column] == id]

            sns.lineplot(data=df_node, x=self.dataloader.temporal_column, y=self.dataloader.target_column, color=testcolor, marker = "o",           ax=ax)
            sns.lineplot(data=df_node, x=self.dataloader.temporal_column, y='pred', color=self.model_color, markeredgecolor='black', marker = "x",  ax=ax)
            ax.set_title(f'predictions {self.dataloader.id_column}: {id}')
            ax.set_xlabel("")            
            ax.grid()  

        plt.tight_layout()
        plt.show()

    def show_forecasts_maps(self,
                            dataset: Literal['train','val','test'],                       
                            tt: int,
                            scale: Literal['constant','equal','individual']):
        
        evaluation_df = self.evaluation_datasets[dataset]
        dates = evaluation_df[self.dataloader.temporal_column].unique()

        date       = dates[tt]
        shapedata = self.dataloader.data['context']['shapedata']
        map_tt    = gpd.GeoDataFrame(pd.merge(evaluation_df[evaluation_df[self.dataloader.temporal_column] == date], shapedata, on = 'node'))


        if scale == 'constant':
            vmin, vmax = evaluation_df['incidence'].min(), evaluation_df['incidence'].max()

        elif scale == 'equal':
            vmin, vmax = map_tt['incidence'].min(), map_tt['incidence'].max()

        else:
            vmin = vmax = None

        fig, axes = plt.subplots(1,2, figsize = (14,8))
        axes = axes.flatten()

        map_tt.plot(column = 'incidence', legend = True, cmap = 'coolwarm', ax = axes[0], vmin = vmin, vmax= vmax)
        axes[0].set_title('true incidence')
        map_tt.plot(column = 'pred',legend = True, cmap = 'coolwarm', ax = axes[1], vmin = vmin, vmax= vmax)
        axes[1].set_title('predicted incidence')

        fig.suptitle(f'Predicted incidence rates at {date.date()}')

        fig.show()
