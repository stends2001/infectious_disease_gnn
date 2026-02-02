import seaborn as sns
from typing import Optional, List, Literal, Any, Union
from abc import abstractmethod
import matplotlib.pyplot as plt
import matplotlib.figure as Figure
import matplotlib.patheffects as path_effects

from ..utils.modelcolors import model_colors

from ...plotting import convert_managedfigure, ManagedFigure

from .predictions_manager import PredictionManager

from ..registry import MODELSREGISTRY
from ..utils import MODELSCOLORPALETTE

from ...dataloading import ShallowDataLoaderManager, DeepDataLoaderManager, GraphDataLoaderManager,BaseLineDataLoaderManager
from ...utils.colors import testcolor, color_is_light
from ...utils.textformatting import warning_emoji, error_emoji, checkmark
from ...configmanagement.modelconfigmanager import ModelConfigManager

class BaseModel:
    """
    """

    def __init__(self, 
                 dataloadermanager: Union[BaseLineDataLoaderManager,ShallowDataLoaderManager, DeepDataLoaderManager, GraphDataLoaderManager], 
                 name:              Optional[str]        = None,
                 model_color:       Optional[str]        = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            self.name = 'unnamed'
        else:
            self.name = self._clean_name(name)
        
        self.dataloadermanager          = dataloadermanager
        self.column_registration        = dataloadermanager.dataorchestrator.column_registration
        self.verbose                    = verbose
        self.config_info                = {}
        self.model_class                = self.__class__.__name__
        self.predictions                = PredictionManager(self.dataloadermanager.dataorchestrator.config, self.column_registration)
        self.prediction_mode            = self.dataloadermanager.dataorchestrator.config.prediction_mode

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

        self._state = {
            'model_initialized' : False,
            'model_hparams_set' : False,
            'global_hparams_set': False,
            'trained'           : False,
            'forecasted'        : False,
        }

        self.model_color = self._get_modelcolor()

        self._update_status('model_initialized')

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def forecast(self):
        """should create .evaluation_df"""
        pass

    @abstractmethod
    def set_global_hparams(self, **kwargs):
        pass

    @abstractmethod
    def set_model_hparams(self, **kwargs):
        pass    

    @convert_managedfigure
    def show_forecasts(self,
                       node_idx:    Union[List[int], int]           = 1,
                       dataset:     Literal['train','val','test']   = 'test',
                       plot_type:   Literal['line', 'map']          = 'line',
                       horizon:     int                             = 0,
                       transformed: bool                            = False,
                       ) -> ManagedFigure:
        
        predictioncollection = self.predictions.get_preds(dataset)
        timesteps_ahead          = int(self.dataloadermanager.dataorchestrator.config.horizon_leadtime + horizon)

        if self.dataloadermanager.dataorchestrator.config.prediction_mode == 'classification':
            raise ValueError('currently no forecast for classifications supported')

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
                nodename = self.nutsnames[self.nutsnames[f'{self.nutslevel}_key'] == nutscode][f'{self.nutslevel}_name'].iloc[0]

                evaluation_df_node  = evaluation_df[evaluation_df['node'] == id]

                sns.lineplot(data   = evaluation_df_node, x = 'timestamp', y = 'target',    color = testcolor,          marker = 'o',   label = 'ground truth', ax = ax, linewidth = 2)

                # if the luminence of the color is too high for a white background we give a black outline of line and marker
                if color_is_light(self.model_color):                  
                    sns.lineplot(data   = evaluation_df_node, x = 'timestamp', y = 'pred',      color = self.model_color,   marker = 'o',   label = f'predictions {self.name}',    ax = ax, linewidth = 2, markeredgecolor = 'black', markeredgewidth=0.2)
                else:
                    sns.lineplot(data   = evaluation_df_node, x = 'timestamp', y = 'pred',      color = self.model_color,   marker = 'o',   label = f'predictions {self.name}',    ax = ax, linewidth = 2)
                ax.set_title(f'{nodename} [node: {id}]')
                ax.set_xlabel("")            
                ax.grid()   

            title = f'{self.dataloadermanager.dataorchestrator.config.target_column} predictions by {self.name}, {timesteps_ahead}{self.dataloadermanager.dataorchestrator.config.temporal_frequency} ahead' 

            if dataset != 'test':
                title += f" [{dataset}]"

            if transformed:
                title += ' [transformed]'        
            
            plt.suptitle(title)
            plt.close()
            return fig
        
        else:
            raise ValueError('currently no other plots than lineplots supported') 

    def save_model(self):
        """
        Save model configuration (hyperparameters, settings).
        This is the main save method - child classes should override
        if they need to save additional things (like weights).
        
        Returns
        -------
        str : The assigned model ID
        """
        if self.name == 'unknown':
            raise ValueError('Model needs a valid name before saving')
        
        model_id                = self.config_manager.register_entry(self.config_info)
        self.config_info['id']  = model_id
       
    def _print_status_update(self, status: str) -> str:

        statement = ""
        title           = f"{self.name}"
        total_width     = 50
        side_padding    = (total_width - 4 - len(title)) // 2  # 4 for "==  =="
        extra_space     = (total_width - 4 - len(title)) % 2    

        if status == 'model_initialized':
                    
                
                if self.verbose >= 1:

                    statement      += "\n".join([
                                            "",
                                            "=" * total_width,
                                            "==" + " " * side_padding + title + " " * (side_padding + extra_space) + "==",
                                            "=" * total_width
                                        ])
                    
                else:
                    statement += "==" + " " * side_padding + title + " " * (side_padding + extra_space) + "=="

        else:
            statement += f"{status} {checkmark}"

        print(statement)
        
    def _update_status(self, status: Literal['model_initialized','model_hparams_set','global_hparams_set','trained','forecasted']):
        self._state[status] = True 

        print_decisions = {
            'model_initialized':  self.verbose >= 0,
            'model_hparams_set':  self.verbose >= 1,
            'global_hparams_set': self.verbose >= 1,
            'trained':            self.verbose >= 1,
            'forecasted':         self.verbose >= 1
        }

        if print_decisions.get(status, False):
            self._print_status_update(status)

    def _check_state(self, required_states: List[str]) -> None:
        """Validate that required setup steps have been completed."""
        missing = [s for s in required_states if not self._state.get(s, False)]
        if missing:
            raise ValueError(
                f"Missing required setup steps: {', '.join(missing)}. "
                f"Call the corresponding methods first."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"    
    
    def _get_modelcolor(self) -> str:
        lookup_name = self.__class__.__name__.lower()
        
        if hasattr(self, 'model_color'):
            return self.model_color
        
        elif lookup_name not in model_colors:
            raise ValueError(f'no color set for model of class {lookup_name}')
        else:
            return model_colors[lookup_name]
        

    def _clean_name(self, name: str) -> str:
        return name.lower().replace(' ', '_')