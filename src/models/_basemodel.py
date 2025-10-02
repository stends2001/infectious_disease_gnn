import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Dict, List, Literal, Any, Union, Tuple
import torch
from ..dataloading.epidataloader import EpiDataLoader
from ..dataloading.gnndataloader import GNNDataLoader
from ..metrics.losses import spike_weighted_mse, mse, spike_detection_loss, temporal_smoothness_loss, spatial_consistency_loss
from ..dataloading.normalization import reverse_zscore_scaling, reverse_log
from ..utils.constants import traincolor, valcolor, testcolor
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors


from ..configmanager.modelconfigmanager import ModelConfigManager

from matplotlib.figure import Figure
from matplotlib.axes import Axes


class BaseModel:

    """
    Parent class for all models. 

    Children:
    --------
    ShallowModel
        non deep-learning models

    DeepModel
        deep learning models

    Parameters:
    ----------
    dataloader: Union['EpiDataLoader','GNNDataLoader']
        the dataloader class from which to take the actual dataloaders.
        For shallow models, use an instance of EpiDataLoader. For 
        deep models use GNNDataLoader.
    name: Optional[str]
        the name associated with the model. Mostly used for plotting and
        saving predicitons.

    TODO: de-normalize predictions 
    TODO: repair show_forecasts_maps   
    """

    def __init__(self, 
                 dataloader: Union['EpiDataLoader', 'GNNDataLoader'], 
                 name:       Optional[str] = None):
        
        self.dataloader         = dataloader
        self.name               = name if name else "unknown"    
        self.evaluation_datasets= {} 
        self.model_color        = None
        self.config_info        = {}
        self.config_info['name'] = name

       # Managers
        self.config_manager = ModelConfigManager()
        self.weights_manager = None  # Only for DeepModel

    def forecast(self):
        """supposed to create the attribute `evaluation_df`"""
        raise NotImplementedError("Each model must implement its own forecast method.")

    def _denorm_predictions(self, df: pd.DataFrame) -> pd.DataFrame:

        zscore_rev = reverse_zscore_scaling(df, params = self.dataloader.transform_params['normalization']['params'])
        log_rev    = reverse_log(zscore_rev, params = self.dataloader.transform_params['log']) 
        return log_rev

    def show_forecasts(self,
                       dataset: Literal['train','val','test'],
                       node_idx: Union[List[int], int] = 1,
                       timeframe: Optional[List[str]] = None,
                       target_h: Optional[int] = 0,
                       transformed: bool = False) -> Tuple[Figure, Axes]:
        """
        Visualizes forecasts made

        Parameters:
        ----------
        dataset: Literal['train','val','test']
            based on whcih dataloader predictions have been made
        node_idx: Union[List[int], int] = 1
            node - label to plot predictions of. If multiple supplied
            (in a list) then multiple subplots are shown.
        timeframe: Optional[List[str]] = None
            List of min-date and max-date
        target_h: Optional[int] = 0
            target-horizon. Please supply in case DeepModel has been
            used.
        transformed: bool = False
            whether to show the transformed or the nontransformed data

        Returns:
        -------
        (fig, axes) -> first figure is the national aggregation of 
        incidence, the remaining figures represent one per node.
        """
    
        target_column = self.dataloader.target_column
        pred_column   = 'pred'
        evaluation_df = self.evaluation_datasets[dataset][f'horizon_{target_h}']

        if not transformed:
            evaluation_df = self._denorm_predictions(evaluation_df)

        if isinstance(node_idx, int):
            node_idx = [node_idx]

        n_plots = len(node_idx)

        fig, axes = plt.subplots(n_plots+1 ,1, figsize = (16, 5 + (5 * n_plots)))
        axes      = axes.flatten()

        if timeframe:
            date0 = timeframe[0]
            date1 = timeframe[1]

            evaluation_df = evaluation_df[evaluation_df[self.dataloader.temporal_column]>= date0]
            evaluation_df = evaluation_df[evaluation_df[self.dataloader.temporal_column]< date1] 


        evaluation_df_aggr= evaluation_df.copy().groupby(self.dataloader.temporal_column).agg({target_column: 'sum', pred_column: 'sum'})
        ax = axes[0]
        sns.lineplot(data=evaluation_df_aggr, x=self.dataloader.temporal_column, y=target_column, color=testcolor, marker = "o",           ax=ax)
        sns.lineplot(data=evaluation_df_aggr, x=self.dataloader.temporal_column, y=pred_column, color=self.model_color, markeredgecolor='black', marker = "x",  ax=ax)
        ax.set_title(f'National aggregation of incidence')
        ax.set_xlabel("")            
        ax.grid()          


        for counter, id  in enumerate(node_idx):

            ax = axes[counter + 1]

            df_node = evaluation_df[evaluation_df[self.dataloader.id_column] == id]

            sns.lineplot(data=df_node, x=self.dataloader.temporal_column, y=target_column, color=testcolor, marker = "o",           ax=ax)
            sns.lineplot(data=df_node, x=self.dataloader.temporal_column, y=pred_column, color=self.model_color, markeredgecolor='black', marker = "x",  ax=ax)
            ax.set_title(f'predictions {self.dataloader.id_column}: {id}')
            ax.set_xlabel("")            
            ax.grid()  

        plt.tight_layout()
        plt.suptitle(f'Predictions by {self.name}')
        return fig, axes

    def show_forecasts_maps(self,
                            dataset: Literal['train','val','test'],                       
                            tt: int,
                            scale: Literal['constant','equal','individual'],
                            target_h: Optional[int] = 0,
                       transformed: bool = False) -> Tuple[Figure, Axes]:

        target_column = self.dataloader.target_column
        pred_column   = 'pred'
        evaluation_df = self.evaluation_datasets[dataset][f'horizon_{target_h}']

        if not transformed:
            evaluation_df = self._denorm_predictions(evaluation_df)

        dates      = evaluation_df[self.dataloader.temporal_column].unique()
        date       = dates[tt]
        print(date)
        shapedata  = self.dataloader.data['context']['shapedata']
        map_tt     = gpd.GeoDataFrame(pd.merge(evaluation_df[evaluation_df[self.dataloader.temporal_column] == date], shapedata, on = 'node'))

        if scale == 'constant':
            vmin, vmax = evaluation_df[target_column].min(), evaluation_df[target_column].max()
        elif scale == 'equal':
            vmin, vmax = map_tt[target_column].min(), map_tt[target_column].max()
        else:
            vmin = vmax = None

        fig, axes = plt.subplots(1, 2, figsize=(12,9))
        axes = axes.flatten()

        cmap = 'Blues'

        # Plot without individual legend/colorbar
        map_tt.plot(column=target_column, cmap=cmap, ax=axes[0], vmin=vmin, vmax=vmax, legend=False)
        # axes[0].set_title('true incidence')
        axes[0].axis('off')  # Remove axes box and ticks

        map_tt.plot(column=pred_column, cmap=cmap, ax=axes[1], vmin=vmin, vmax=vmax, legend=False)
        # axes[1].set_title('predicted incidence')
        axes[1].axis('off')  # Remove axes box and ticks

        axes[0].text(0.01, 0.98, 'A', transform=axes[0].transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

        axes[1].text(0.01, 0.98, 'B', transform=axes[1].transAxes, fontsize=16, fontweight='bold', va='top', ha='left')



        # Create a single colorbar for both plots:
        sm = cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])  # the public API to set data for the colorbar

        cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.03, pad=0.02)
        cbar.set_label('transformed incidence')

        fig.show()
        return fig, axes
    
    def save_model(self):
        """
        Save model configuration (hyperparameters, settings).
        This is the main save method - child classes should override
        if they need to save additional things (like weights).
        
        Returns:
        -------
        str : The assigned model ID
        """
        if self.name == 'unknown':
            raise ValueError('Model needs a valid name before saving')
        
        model_id = self.config_manager.register_entry(self.config_info)
        self.config_info['id'] = model_id
