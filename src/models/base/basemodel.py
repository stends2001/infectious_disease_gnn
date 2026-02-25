import seaborn as sns
from typing import Optional, List, Literal, Any, Union
from abc import abstractmethod
import matplotlib.pyplot as plt
import matplotlib.figure as Figure
import matplotlib.patheffects as path_effects
import matplotlib.dates as mdates

from ..utils.modelcolors import model_colors

from ...plotting import convert_managedfigure, ManagedFigure

from .predictions_manager import PredictionManager

from ..registry import MODELSREGISTRY
from ..utils import MODELSCOLORPALETTE
from ...dataloading import BaseLineDataLoaderManager, DeepDataLoaderManager, GraphDataLoaderManager
# from ...dataloading import ShallowDataLoaderManager, DeepDataLoaderManager, GraphDataLoaderManager,BaseLineDataLoaderManager
from ...utils.colors import testcolor, color_is_light
from ...utils.textformatting import warning_emoji, error_emoji, checkmark

class BaseModel:

    def __init__(self, 
                 dataloadermanager: Union[BaseLineDataLoaderManager, DeepDataLoaderManager, GraphDataLoaderManager], 
                 name:              Optional[str]        = None,
                 model_color:       Optional[str]        = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            self.name = 'unnamed model'
        else:
            self.name = self._clean_name(name)
        
        self.dataloadermanager          = dataloadermanager
        self.epiconfig                  = self.dataloadermanager.dataorchestrator.config
        self.column_registration        = dataloadermanager.dataorchestrator.column_registration
        self.context_data               = dataloadermanager.dataorchestrator.data_context
        self.temporal_summary           = self.context_data.temporal_summary
        self.verbose                    = verbose
        self.config_info                = {}
        self.model_class                = self.__class__.__name__
        self.predictions                = PredictionManager(self.dataloadermanager.dataorchestrator, self.column_registration, self.temporal_summary)
        self.prediction_mode            = self.dataloadermanager.dataorchestrator.config.prediction_mode

        # Config
        self.config_info['name']        = name
        self.config_info['model_class'] = self.model_class

        # Managers
        # self.config_manager             = ModelConfigManager()
        self.weights_manager            = None                  # Only for DeepModel
        
        # context data for plotting
        self.tokenization_map           = self.context_data.tokenization_map
        self.nutsnames                  = self.context_data.nuts_harm     
        self.nutslevel                  = self.epiconfig.nuts_level

        self._state = {
            'model_initialized' : False,
            'model_hparams_set' : False,
            'global_hparams_set': False,
            'trained'           : False,
            'forecasted'        : False,
        }

        self.model_color = self._get_modelcolor()

        self._update_status('model_initialized')

    def train(self):
        raise NotImplementedError("Child classes of BaseModel must implement train-method")

    def forecast(self):
        """should create .evaluation_df"""
        raise NotImplementedError("Child classes of BaseModel must implement forecast-method")

    def set_global_hparams(self):
        raise NotImplementedError("Child classes of BaseModel must implement set_global_hparams-method")
    
    def set_model_hparams(self):
        raise NotImplementedError("Child classes of BaseModel must implement set_global_hparams-method")   

    @convert_managedfigure
    def show_forecasts(self,
                       node_idx:    Union[int, List[Union[int,Literal['national']]], Literal['national']]  = 1,
                       dataset:     Literal['train','val','test']               = 'test',
                       plot_type:   Literal['line', 'map']                      = 'line',
                       horizon:     int                                         = 0,
                       is_original: bool                                        = True,
                       ) -> ManagedFigure:
                
        predictioncollection = self.predictions.get_preds(dataset)
        x_range              = self.temporal_summary.get_daterange_dataset(dataset, reference = 'target')
        xlimits              = [self.temporal_summary._shift(x_range[0], -1), self.temporal_summary._shift(x_range[1], 1)]
        timesteps_ahead      = int(self.dataloadermanager.dataorchestrator.config.horizon_leadtime + horizon)

        df_pred_aggr = None
        if isinstance(node_idx, list):
            if 'national' in node_idx:
                df_pred_aggr = predictioncollection.get(horizon      = horizon,
                                                        is_original  = True,
                                                        spatially_aggregated = True
                                                        )
            nodes_list = node_idx
        elif isinstance(node_idx, int):
            nodes_list = [node_idx]
        elif isinstance(node_idx, str):
            if node_idx != 'national':
                raise ValueError(f'the only string value for node_idx allowed is "national". Got {node_idx}')
            df_pred_aggr = predictioncollection.get(horizon      = horizon,
                                                    is_original  = True,
                                                    spatially_aggregated = True
                                                    )
            nodes_list = [node_idx]
        else:
            raise ValueError(f"Invalid input for node_idx. Must be Union[int, List[Union[int,Literal['national']]]. Got {node_idx}")            
        
        df_pred = predictioncollection.get(horizon       = horizon,
                                            is_original  = is_original,
                                            spatially_aggregated=False)

        if self.dataloadermanager.dataorchestrator.config.prediction_mode == 'classification':
            raise ValueError('currently no forecast for classifications supported')

        if plot_type == 'line':

            n_plots     = len(nodes_list)

            fig, axes   = plt.subplots(n_plots ,1, figsize = (16, 2 + (5 * n_plots)))
            
            if n_plots > 1:
                axes        = axes.flatten()           
            else:
                axes = [axes]

            for counter, id in enumerate(nodes_list):
                ax = axes[counter]

                # ======== get target data ==========
                
                if id != 'national':
                    nodename        = self.nutsnames[self.nutsnames[f'{self.epiconfig.id_column}'] == id][f'{self.nutslevel}_name'].iloc[0]
                    df_selection    = df_pred[df_pred[self.epiconfig.id_column] == id]
                elif df_pred_aggr is not None: 
                    nodename        = 'nationally'
                    df_selection    = df_pred_aggr
                else:
                    raise ValueError('smth went wrong in getting aggregated df')
                
                sns.lineplot(data   = df_selection, x = self.epiconfig.temporal_column, y = 'target',    color = testcolor,          marker = 'o',   label = 'ground truth', ax = ax, linewidth = 2)

                quantiles = self.epiconfig.quantiles

                if quantiles:
                    # find the index of the quantile closest to 0.5
                    median_idx  = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5))
                    center_col  = f'pred_q{median_idx}'
                    bottom_col  = 'pred_q0'
                    top_col     = f'pred_q{len(quantiles) - 1}'
                else:
                    center_col  = 'pred'
                    bottom_col  = None
                    top_col     = None

                # center line
                if color_is_light(self.model_color):
                    sns.lineplot(data = df_selection, x = self.epiconfig.temporal_column, y = center_col,
                                color = self.model_color, marker='o', label=f'predictions {self.name}',
                                ax=ax, linewidth=2, markeredgecolor='black', markeredgewidth=0.2)
                else:
                    sns.lineplot(data = df_selection, x = self.epiconfig.temporal_column, y = center_col,
                                color=self.model_color, marker='o', label=f'predictions {self.name}',
                                ax=ax, linewidth=2)

                # quantile band
                if bottom_col and top_col:
                    ax.fill_between(
                        df_selection[self.epiconfig.temporal_column],
                        df_selection[bottom_col],
                        df_selection[top_col],
                        color=self.model_color,
                        alpha=0.2,
                        label=f'{quantiles[0]}–{quantiles[-1]} interval'
                    )

                ax.set_xlabel("")   
                ax.set_xlim(xlimits)     
                ax.set_title(f'{nodename} [node: {id}]')    
                ax.grid()   

            title = f'{self.dataloadermanager.dataorchestrator.config.target_column} predictions by {self.name}, {timesteps_ahead}{self.dataloadermanager.dataorchestrator.config.temporal_frequency} ahead' 

            if dataset != 'test':
                title += f" [{dataset}]"

            if not is_original:
                title += ' [transformed]'        
            
            plt.suptitle(title)
            plt.close()
            return fig

    # @abstractmethod
    def save_model(self):
        pass
       
    def _print_status_update(self, status: str) -> None:

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
        if isinstance(required_states, str):
            required_states = [required_states]

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