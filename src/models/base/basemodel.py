import seaborn as sns
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from typing import Optional, Dict, List, Literal, Any, Union, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from ...utils import testcolor
from ...dataloading import ShallowDataLoaderManager
from ...dataloading import GraphDataLoaderManager
from ...dataloading.dataorchestration.normalization import reverse_log, reverse_minmax_scaling, reverse_zscore_scaling

from ..registry import MODELSREGISTRY
from src.models.utils import MODELSCOLORPALETTE

from src.configmanagement.modelconfigmanager import ModelConfigManager

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from src.utils.helpers import sum_preserve_nan

from .predictions_manager import PredictionCollection, PredictionManager

class BaseModel:

    """
    """

    def __init__(self, 
                 dataloadermanager: Union[ShallowDataLoaderManager, GraphDataLoaderManager], 
                 name:              Optional[str] = None):
        
        self.dataloadermanager          = dataloadermanager
        self.column_registration        = dataloadermanager.dataorchestrator.column_registration
        self.name                       = name 
        self.config_info                = {}
        self.model_class                = self.__class__.__name__
        self.model_color                = self.get_model_color()

        self.predictions                = PredictionManager(self.dataloadermanager.dataorchestrator.config, self.column_registration)

        # Config
        self.config_info['name']        = name
        self.config_info['model_class'] = self.model_class

        # Managers
        self.config_manager             = ModelConfigManager()
        self.weights_manager            = None                  # Only for DeepModel
        
        # context data for plotting
        self.tokenization_map           = self.dataloadermanager.dataorchestrator.data_context.tokenization_map
        self.nutsnames                  = self.dataloadermanager.dataorchestrator.data_context.nuts_names     
        self.nutslevel                  = self.dataloadermanager.dataorchestrator.config.nuts_level

    def forecast(self):
        """supposed to create the attribute `evaluation_df`"""
        raise NotImplementedError("Each model must implement its own forecast method.")

    def show_forecasts(self,
                       node_idx:    Union[List[int], int]           = 1,
                       dataset:     Literal['train','val','test']   = 'test',
                       plot_type:   Literal['line', 'map']          = 'line',
                       horizon:     int                             = 0,
                       transformed: bool                            = False,
                       ):
        
        predictioncollection = self.predictions.get_preds(dataset)
        weeks_ahead          = int(self.dataloadermanager.dataorchestrator.config.horizon_leadtime + horizon)

        if not transformed:
            evaluation_df = predictioncollection.get_original(horizon)

        else:
            evaluation_df = predictioncollection.get_transformed(horizon)


        if isinstance(node_idx, int):
            node_idx = [node_idx]


        if plot_type == 'line':

            n_plots     = len(node_idx)
            fig, axes   = plt.subplots(n_plots ,1, figsize = (16, 2 + (5 * n_plots)))
            
            if n_plots > 1:
                axes        = axes.flatten()           
            else:
                axes = [axes]
        
            for counter, id in enumerate(node_idx):
                ax = axes[counter]
                nutscode = self.tokenization_map['idx_id'][id]
                nodename = self.nutsnames[self.nutsnames[self.nutslevel] == nutscode][f'{self.nutslevel}_name'].iloc[0]

                evaluation_df_node  = evaluation_df[evaluation_df['node'] == id]

                sns.lineplot(data   = evaluation_df_node, x = 'timestamp', y = 'target',    color = testcolor,          marker = 'o',   label = 'True Incidence', ax = ax, linewidth = 2)
                sns.lineplot(data   = evaluation_df_node, x = 'timestamp', y = 'pred',      color = self.model_color,   marker = 'h',   label = 'Predictions',    ax = ax, linewidth = 2)

                ax.set_title(f'{nodename} [node: {id}]')
                ax.set_xlabel("")            
                ax.grid()   

            title = f'Predictions by {self.name}, {weeks_ahead} weeks ahead' 

            if dataset != 'test':
                title += f" [{dataset}]"

            if transformed:
                title += ' [transformed]'        
            
            plt.suptitle(title)
            plt.tight_layout()
            return fig, axes
        
        else:
            raise ValueError('currently no other plots than lineplots supported')

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
        
        model_id                = self.config_manager.register_entry(self.config_info)
        self.config_info['id']  = model_id
       
    def get_model_color(self):
        """returns color associated with model type"""
        id = MODELSREGISTRY.get(self.model_class, 0)    # 0 as fallback 
        return MODELSCOLORPALETTE[id]

    def _return_model_print(self) -> str:
        """prints the model name for logs"""
        total_width     = 50
        title           = f"{self.name}"
        side_padding    = (total_width - 4 - len(title)) // 2  # 4 for "==  =="
        extra_space     = (total_width - 4 - len(title)) % 2
        statement       = "\n".join([
                                "",
                                "=" * total_width,
                                "==" + " " * side_padding + title + " " * (side_padding + extra_space) + "==",
                                "=" * total_width
                            ])
        return statement

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"    
    
    @property
    def clean_name(self) -> str:
        if self.name is None:
            return 'unnamed'
        return self.name.lower().replace(' ', '_')