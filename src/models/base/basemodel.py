from typing import Union, Optional, Literal, List, Tuple, Self, Generic, TypeVar
import pandas as pd
from matplotlib.axes import Axes
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

from .issues import FutureUpdateError, ModelStatusError, ModelInitError
from .predictions_manager import PredictionManager

from ..utils.modelcolors import model_colors

from ...dataloading import BaseLineDataLoaderManager, DeepDataLoaderManager, GraphDataLoaderManager
from ...plotting import ManagedFigure
from ...utils import testcolor, color_is_light, checkmark, align, section

DLM = TypeVar('DLM', bound=Union[BaseLineDataLoaderManager, DeepDataLoaderManager, GraphDataLoaderManager])

class BaseModel(Generic[DLM]):

    """ 
    Parent class of ALL models, baseline, shallow and deep.
    Upon init, all models must supply the following

    Parameters
    ----------
    dataloadermanager: Union[BaseLineDataLoaderManager, DeepDataLoaderManager, GraphDataLoaderManager]

    name: Optional[str] = None 
    
    verbose: int = -1
        the following levels are created for verbose:
        - -1:
        - 0:
        - 1:
        - 2:

    Downstream
    ----------
    BaseModel has 3 types of subclasses:
    - BaseLineModel
    - ShallowModel
    - DeepModel

    Each of which has further subclasses. SimpleGCNModel and LSTMModel are example of a DeepModel.
    """
    _expected_dataloadermanager: str 

    def __init__(self, 
                 dataloadermanager: DLM, 
                 name:              Optional[str]   = None,
                 verbose:           int             = -1):
        
        if not name:
            name = 'unnamed model'
        self.name = self._clean_name(name)

        # static attributes
        self.dataloadermanager          = dataloadermanager
        self.epiconfig                  = self.dataloadermanager.dataorchestrator.config
        self.column_registration        = dataloadermanager.dataorchestrator.column_registration
        self.context_data               = dataloadermanager.dataorchestrator.data_context
        self.temporal_summary           = self.context_data.temporal_summary
        self.verbose                    = verbose
        self.prediction_mode            = self.dataloadermanager.dataorchestrator.config.prediction_mode
        self.model_class                = self.__class__.__name__
        self.model_color                = self._get_modelcolor()
        self.pred_cols                  = self._return_pred_cols()

        # validate
        self._validate_dataloadermanager()
        
        # dynamic (changing) attributes
        self.config_info                = {}
        self.predictions                = PredictionManager(self.dataloadermanager.dataorchestrator, self.column_registration, self.temporal_summary)
        
        # Config
        self.config_info['name']        = self.name
        self.config_info['model_class'] = self.model_class

        # weight managers -> for deepmodel
        self.weights_manager            = None
        
        self._state = {
            'model_initialized' : False,
            'model_hparams_set' : False,
            'global_hparams_set': False,
            'trained'           : False,
            'forecasted'        : False,
        }

        self._update_status('model_initialized')

    # ======= METHODS ====== #
    def show_forecasts(self,
                       node_idx:    Union[int, List[Union[int,Literal['national']]], Literal['national']]  = 0,
                       dataset:     Literal['train','val','test']  = 'test',
                       plot_type:   Literal['line']                = 'line',
                       horizon:     int                            = 0,
                       is_original: bool                           = True,
                       ) -> ManagedFigure:
        
        """
        Plot forecasts

        Parameters
        ----------
        node_idx: Union[int, List[Union[int,Literal['national']]], Literal['national']]  = 0
            the nodes to plot the predicted timeseries for
        dataset: Literal['train','val','test']  = 'test'
        plot_type: Literal['line']
            #TODO at some point Id like to implement a map
        horizon: int
            the horizon for which to plot forecasts
        is_original: bool
            whether to plot original scale (is_original == True) or transformed (is_original == False)

        Returns
        -------
        ManagedFigure
        """
        # ====  validate input ==== #
        if plot_type != 'line':
            raise FutureUpdateError(f"Currently only plot_type == 'line' supported, got {plot_type}")
        if self.prediction_mode == 'classification':        
            raise FutureUpdateError('currently no forecast for classifications supported')            
        
        # ==== get constants ===== #
        x_range             = self.temporal_summary.get_daterange_dataset(dataset, reference = 'target')
        # extend axes to one step before and one after the first and last pred        
        xlimits             = [self.temporal_summary._shift(x_range[0], -1), self.temporal_summary._shift(x_range[1], 1)]
        timesteps_ahead     = int(self.dataloadermanager.dataorchestrator.config.horizon_leadtime + horizon)
        quantiles           = self.epiconfig.quantiles        

        # ==== get predictions ==== #
        nodes_list, df_pred, df_pred_aggr = self._get_forecast_dfs(node_idx, dataset, horizon, is_original)

        # ===== plot ===== #
        # create plots
        n_plots         = len(nodes_list)
        fig, axes_array = plt.subplots(n_plots, 1, figsize=(16, 2 + 5 * n_plots), squeeze=False)
        axes: list[Axes]= list(axes_array.flatten())

        for plot_idx, id in enumerate(nodes_list):
            ax = axes[plot_idx]

            # ======== get pred data ========== #
            if id == 'national':
                if df_pred_aggr is None:
                    raise ValueError('Missing df_aggregated in show_forecasts!')
                df_node     = df_pred_aggr 
                ax_title    = f'nationally aggregated'

            else:
                df_node     = df_pred[df_pred[self.epiconfig.id_column] == id]
                nodename    = self.context_data.nuts_harm[self.context_data.nuts_harm[f'{self.epiconfig.id_column}'] == id][f'{self.epiconfig.nuts_level}_name'].iloc[0]
                ax_title    = f'{nodename} [node: {id}]'
                        
            if quantiles:
                # find the index of the quantile 0.5
                middle_idx  = len(quantiles) // 2
                center_col  = f'pred_q{middle_idx}'
                bottom_col  = 'pred_q0'
                top_col     = f'pred_q{len(quantiles) - 1}'
            else:
                center_col  = 'pred'
                bottom_col  = None
                top_col     = None
            
            # ====== drawing lines ========== #
            # target
            sns.lineplot(data       = df_node, 
                            x          = self.epiconfig.temporal_column, 
                            y          = 'target',    
                            color      = testcolor,          
                            marker     = 'o',   
                            label      = 'ground truth', 
                            ax         = ax, 
                            linewidth  = 2)

            # center pred (either point pred or quantile 0.5)
            sns.lineplot(data           = df_node, 
                        x               = self.epiconfig.temporal_column, 
                        y               = center_col,
                        color           = self.model_color, 
                        marker          = 'o', 
                        label           = 'median predictions' if quantiles else 'point predictions',
                        ax              = ax, 
                        linewidth       = 2, 
                        markeredgewidth = 0.3,

                        # adjust some color-aspects if color is light
                        markeredgecolor = 'black' if color_is_light(self.model_color) else 'white',
                        )

            # quantile band
            if quantiles:
                ax.fill_between(
                    x       = df_node[self.epiconfig.temporal_column],
                    y1      = df_node[bottom_col],
                    y2      = df_node[top_col],
                    color   = self.model_color,
                    alpha   = 0.2,
                    label   = f'uncertainty interval ({quantiles[0]}–{quantiles[-1]})'
                )

            ax.set_xlabel("")   
            # xlimits is a list of two datetime objects
            ax.set_xlim(xlimits)    # type: ignore
            ax.set_title(ax_title)    
            ax.legend()
            ax.grid()   

        suptitle = f'{self.dataloadermanager.dataorchestrator.config.target_column} predictions by {self.name}, {timesteps_ahead}{self.dataloadermanager.dataorchestrator.config.temporal_frequency} ahead' 

        if dataset != 'test':
            suptitle += f" [{dataset}]"

        if not is_original:
            suptitle += ' [transformed]'        
    
        plt.close()

        managed_figure = (ManagedFigure(fig)
            .labels.change_suptitle(suptitle,'bold',14)
        )       

        return managed_figure

    # ======== METHODS TO BE IMPLEMENTED BY SUBCLASSES ======== #
    def train(self):
        raise NotImplementedError("Subclasses of BaseModel must implement train-method")

    def forecast(self):
        """should create .evaluation_df"""
        raise NotImplementedError("Subclasses of BaseModel must implement forecast-method")

    def set_global_hparams(self):
        raise NotImplementedError("Subclasses of BaseModel must implement set_global_hparams-method")
    
    def set_model_hparams(self):
        raise NotImplementedError("Subclasses of BaseModel must implement set_model_hparams-method")   

    def save_model(self):
        raise NotImplementedError("Subclasses of BaseModel must implement save_model-method")
          
    # ======== HIDDEN METHODS ========= #
    
    # helpers
    def _get_forecast_dfs(self, 
                          node_idx:     Union[int, List[Union[int,Literal['national']]], Literal['national']], 
                          dataset:      Literal['train','val','test'], 
                          horizon:      int, 
                          is_original:  bool
                          ) -> Tuple[List[int | Literal['national']] | list[int] | list[str], pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Returns the dfs of the forecasts by interacting with the prediction-manager

        Parameters
        ----------
        node_idx
        dataset
        horizon
        is_original

        Returns
        -------
        nodes_list
        df_pred
        df_pred_aggr
        """
        predictioncollection = self.predictions.get_preds(dataset)

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
        
        return nodes_list, df_pred, df_pred_aggr

    def _return_pred_cols(self) -> List[str]:
        """ 
        return the names of the prediction - columns
        """
        if self.epiconfig._num_quantiles == 0:
            pred_cols= ['pred']
        else:
            pred_cols= [c for c in self.column_registration.pred_columns if c != 'pred']
        return pred_cols    

    # validation - related 
    def _validate_dataloadermanager(self):
        """validate class of dataloadermanager"""
        if not hasattr(self, '_expected_dataloadermanager'):
            raise ModelInitError(f'attribute self._expected_dataloadermanager not set in model {self.name}')

        exp = self._expected_dataloadermanager
        got = self.dataloadermanager.__class__.__name__

        if exp != got:
            raise ModelInitError(f'{self.name} expected a dataloadermanager of class {exp} but got {got}')

    # status - related
    def _print_status_update(self, status: str) -> None:
        """
        prints a status update, depending on verbose
        returns None, but prints directly
        """
        statement       = ""
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
        
    def _update_status(self, 
                       status: Literal['model_initialized',
                                       'model_hparams_set',
                                       'global_hparams_set',
                                       'trained',
                                       'forecasted']):
        """
        updates a key-value in self._state
        """
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

    def _check_state(self, required_states: Union[List[str],str]) -> None:
        """
        Validate that required setup steps have been completed
        """
        if isinstance(required_states, str):
            required_states = [required_states]

        missing = [state for state in required_states if not self._state.get(state, False)]

        if missing:
            raise ModelStatusError(
                f"Missing required setup steps: {', '.join(missing)}. "
                f"Call the corresponding methods first."
            )
    
    # appearance - related
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
    # ======== REPRESENTATION METHODS ===== #
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"    

    def __str__(self) -> str:

        all_keys = (
            ['model name', 'model class'] + list(self._state.keys())
        )

        width = max(len(k) for k in all_keys) if all_keys else 20
        
        # Build output
        lines = [f'<{self.__class__.__name__}(']
        lines.append(align('model name',  self.name, width))
        lines.append(align('model class', self.model_class, width))
        lines.append('')
        
        # Status section
        status_items = {k: "✓" if v else "✗" for k, v in self._state.items()}
        lines.extend(section('status', status_items, width))
        lines.append('')
                
        # Global hparams
        lines.extend(section('global hparams', self.config_info.get('global_hparams', {}), width))
        
        lines.append(')>')
        
        return '\n'.join(lines)