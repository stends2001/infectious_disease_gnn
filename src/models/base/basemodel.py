import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Dict, List, Literal, Any, Union, Tuple
import torch
from ...dataloading.epidataloader import EpiDataLoader
from ...dataloading.deepdataloader import DeepDataLoader
from ...dataloading.normalization import reverse_zscore_scaling, reverse_log, reverse_minmax_scaling
from ...utils import testcolor
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from ..registry import MODELSREGISTRY
from ..utils import MODELSCOLORPALETTE

from ...configmanager.modelconfigmanager import ModelConfigManager

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from ...utils.helpers import sum_preserve_nan

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
    dataloader: Union['EpiDataLoader','DeepDataLoader']
        the dataloader class from which to take the actual dataloaders.
        For shallow models, use an instance of EpiDataLoader. For 
        deep models use DeepDataLoader.
    name: Optional[str]
        the name associated with the model. Mostly used for plotting and
        saving predicitons. 
    """

    def __init__(self, 
                 dataloader: Union['EpiDataLoader', 'DeepDataLoader'], 
                 name:       Optional[str] = None):
        
        self.dataloader         = dataloader
        self.name               = name if name else "unknown"    
        self.evaluation_datasets= {} 
        self.model_color        = None
        self.config_info        = {}

        self.model_class    = self.__class__.__name__
        self.model_color    = self.get_model_color()

        self.config_info['name'] = name
        self.config_info['model_class'] = self.model_class

       # Managers
        self.config_manager = ModelConfigManager()
        self.weights_manager = None  # Only for DeepModel

    def forecast(self):
        """supposed to create the attribute `evaluation_df`"""
        raise NotImplementedError("Each model must implement its own forecast method.")

    def _denorm_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.dataloader.transform_params['normalization']['method'] == 'zscore':
            normalization_rev = reverse_zscore_scaling(df, params = self.dataloader.transform_params['normalization']['params'])
        elif self.dataloader.transform_params['normalization']['method'] == 'minmax':
            normalization_rev = reverse_minmax_scaling(df, params = self.dataloader.transform_params['normalization']['params'])   
        else:
            raise ValueError(f'normalization method unknown {self.dataloader.transform_params["normalization"]["method"]}')        

        if 'log' in self.dataloader.transform_params.keys():
            log_rev    = reverse_log(normalization_rev, params = self.dataloader.transform_params['log']) 
            return log_rev

        else:
            return normalization_rev
        
    def show_forecasts(self,
                       dataset: Literal['train','val','test'],
                       node_idx: Union[List[int], int] = 1,
                       timeframe: Optional[List[str]] = None,
                       target_h: int = 0,
                       transformed: bool = False,
                       show_all_horizons: bool = False) -> Tuple[Figure, Axes]:
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
        show_all_horizons: bool = False
            whether to show all horizons (in separate lines)
        Returns:
        -------
        (fig, axes) -> first figure is the national aggregation of 
        incidence, the remaining figures represent one per node.
        """
        
        target_column = self.dataloader.target_column
        pred_column   = 'pred'

        if show_all_horizons:
            available_horizons = []
            for h in range(self.dataloader.horizon_size):
                horizon_key = f'horizon_{h}'
                if dataset in self.evaluation_datasets:
                    if transformed and 'transformed' in self.evaluation_datasets[dataset]:
                        if horizon_key in self.evaluation_datasets[dataset]['transformed']:
                            available_horizons.append(h)
                    elif not transformed and 'nontransformed' in self.evaluation_datasets[dataset]:
                        if horizon_key in self.evaluation_datasets[dataset]['nontransformed']:
                            available_horizons.append(h)
        else:
            available_horizons = [target_h]

        colors = generate_tints(self.model_color, len(available_horizons))

        if isinstance(node_idx, int):
            node_idx = [node_idx]

        n_plots = len(node_idx)

        fig, axes = plt.subplots(n_plots+1 ,1, figsize = (16, 5 + (5 * n_plots)))
        axes      = axes.flatten()
        ax        = axes[0]

        date0, date1 = None, None

        if available_horizons:
            h             = available_horizons[0]
            evaluation_df = self.evaluation_datasets[dataset]['transformed'][f'horizon_{h}'] if transformed else self.evaluation_datasets[dataset]['nontransformed'][f'horizon_{h}']

            if timeframe:
                date0 = timeframe[0]
                date1 = timeframe[1]

                evaluation_df = evaluation_df[evaluation_df[self.dataloader.temporal_column]>= date0]
                evaluation_df = evaluation_df[evaluation_df[self.dataloader.temporal_column]< date1] 

            if transformed:
                evaluation_df_aggr= evaluation_df.copy().groupby(self.dataloader.temporal_column).agg({target_column: sum_preserve_nan, pred_column: sum_preserve_nan})

                title = f'Nationally aggregated transformed {self.dataloader.target_column} values'
            else:
                merged_df = pd.merge(self.evaluation_datasets['test']['nontransformed'][f'horizon_{target_h}'][['timestamp','node',self.dataloader.target_column,'pred']], self.dataloader.data['context']['epidemiological_data'][['timestamp','node','cases','population_size']], on = ['timestamp','node'])
                merged_df["cases_true"] = merged_df["incidence"] * merged_df["population_size"]
                merged_df["cases_pred"] = merged_df["pred"] * merged_df["population_size"]

                # Group by timestamp, sum reconstructed cases and population
                evaluation_df_aggr = merged_df.groupby("timestamp").agg({
                    "cases_true": "sum",
                    "cases_pred": "sum",
                    "population_size": "sum"
                }).reset_index()

                # Calculate national-level incidence rates
                evaluation_df_aggr[target_column] = evaluation_df_aggr["cases_true"] / evaluation_df_aggr["population_size"]
                evaluation_df_aggr[pred_column] = evaluation_df_aggr["cases_pred"] / evaluation_df_aggr["population_size"] 
                title = f'National incidence rate (per {self.dataloader.incidence_scalar})'

            sns.lineplot(data=evaluation_df_aggr, x=self.dataloader.temporal_column, y=target_column, color=testcolor, marker = "o", ax=ax)
            
            for i,h in enumerate(available_horizons):
                eval_data = self.evaluation_datasets[dataset]['transformed'][f'horizon_{h}'] if transformed else self.evaluation_datasets[dataset]['nontransformed'][f'horizon_{h}']

                if timeframe and date0 is not None and date1 is not None:
                    eval_data = eval_data[eval_data[self.dataloader.temporal_column]>=date0]
                    eval_data = eval_data[eval_data[self.dataloader.temporal_column]<date1]

                if transformed:
                    eval_aggr = eval_data.copy().groupby(self.dataloader.temporal_column).agg({pred_column:sum_preserve_nan})
                else:
                    merged_df   = pd.merge(self.evaluation_datasets[dataset]['nontransformed'][f'horizon_{h}'][['timestamp','node','pred']], self.dataloader.data['context']['epidemiological_data'][['timestamp','node','population_size']], on = ['timestamp','node'])
                    merged_df['cases_pred'] = merged_df['pred'] * merged_df['population_size']
                    eval_aggr = merged_df.groupby('timestamp').agg({'cases_pred': 'sum','population_size': 'sum'}).reset_index()
                    eval_aggr[pred_column] = eval_aggr['cases_pred'] / eval_aggr['population_size']

                sns.lineplot(data = eval_aggr, x = self.dataloader.temporal_column, y = pred_column, color = colors[i], label = f'pred h={h}', ax = ax)

            ax.set_title(title)
            ax.set_xlabel("")            
            ax.grid()        




        for counter, id  in enumerate(node_idx):

            ax = axes[counter + 1]

            if available_horizons:
                h             = available_horizons[0]
                evaluation_df = self.evaluation_datasets[dataset]['transformed'][f'horizon_{h}'] if transformed else self.evaluation_datasets[dataset]['nontransformed'][f'horizon_{h}']
                
                if timeframe:
                    date0 = timeframe[0]
                    date1 = timeframe[1]

                    evaluation_df = evaluation_df[evaluation_df[self.dataloader.temporal_column]>= date0]
                    evaluation_df = evaluation_df[evaluation_df[self.dataloader.temporal_column]< date1] 

                df_node = evaluation_df[evaluation_df[self.dataloader.id_column] == id]
                sns.lineplot(data = df_node, x = self.dataloader.temporal_column, y = target_column, color = testcolor, marker = 'o', label = 'True', ax = ax, linewidth = 2)

                for i,h in enumerate(available_horizons):
                    eval_data = self.evaluation_datasets[dataset]['transformed'][f'horizon_{h}'] if transformed else self.evaluation_datasets[dataset]['nontransformed'][f'horizon_{h}']

                    if timeframe and date0 is not None and date1 is not None:
                        eval_data = eval_data[eval_data[self.dataloader.temporal_column]>=date0]
                        eval_data = eval_data[eval_data[self.dataloader.temporal_column]<date1]

                    df_node_h = eval_data[eval_data[self.dataloader.id_column] == id]

                    sns.lineplot(data = df_node_h, x = self.dataloader.temporal_column, y = pred_column, color = colors[i], label = f'pred h={h}' , ax = ax)

                    ax.set_title(f'Predictions {self.dataloader.id_column}: {id}')
                ax.set_xlabel("")            
                ax.grid()   
        
        plt.suptitle(f'Predictions by {self.name}')
        plt.tight_layout()
        return fig, axes

    def show_forecasts_maps(self,
                            dataset: Literal['train','val','test'],                       
                            tt: int,
                            scale: Literal['constant','equal','individual'],
                            target_h: Optional[int] = 0,
                            transformed: bool = False) -> Tuple[Figure, Axes]:

        target_column: str = self.dataloader.target_column # type: ignore
        pred_column   = 'pred'
        evaluation_df = self.evaluation_datasets[dataset]['transformed'][f'horizon_{target_h}'] if transformed else self.evaluation_datasets[dataset]['nontransformed'][f'horizon_{target_h}']

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
       
    def get_model_color(self):

        id = MODELSREGISTRY.get(self.model_class, 0)    # 0 as fallback 
        
        return MODELSCOLORPALETTE[id]

    def _return_training_print(self) -> str:
        total_width = 40
        title = f"Training {self.name}"
        # Calculate padding on each side
        side_padding = (total_width - 4 - len(title)) // 2  # 4 for "==  =="
        # If odd length, add one more space to the right
        extra_space = (total_width - 4 - len(title)) % 2
        statement = "\n".join([
            "",
            "=" * total_width,
            "==" + " " * side_padding + title + " " * (side_padding + extra_space) + "==",
            "=" * total_width
        ])
        return statement


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"    
    

import matplotlib.colors as mcolors

def generate_tints(c, n=3):
    # Convert hex to RGB if needed
    if isinstance(c, str):
        c = mcolors.hex2color(c)
    
    # Calculate lighter and darker factors
    factors = [1 + i * 0.11 for i in range(1, (n // 2) + 1)]  # Lighter tints
    factors += [1 - i * 0.11 for i in range(1, (n // 2) + 1)]  # Darker tints

    # Adjust if n is odd to include the original color in the middle
    if n % 2 == 1:
        factors.insert(n // 2, 1)

    # Apply the factors
    return [tuple(min(1, max(0, x * factor)) for x in c) for factor in factors]