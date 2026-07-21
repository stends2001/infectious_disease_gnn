from typing import Dict,  Union, Optional, Type, Any, Literal, Tuple
import torch 
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path 
import os
import pandas as pd

from .internals import DeepModelInternalsMixin
from .presentation import DeepModelPresentationMixin
from .training import DeepModelTrainMixin
from .forecasting import DeepModelForecastMixin
from .globalhparams import DeepModelGlobalhParamsMixin
from .checkpoint import DeepModelCheckpointMixin

from ....dataloading.databuilders import GraphDataBuilder
from ..strategies.basestrategy import Strategy
from ...base.basemodel import BaseModel
    
import logging
logger = logging.getLogger(__name__)

class DeepModel(
    DeepModelInternalsMixin,
    DeepModelPresentationMixin,
    DeepModelTrainMixin,
    DeepModelForecastMixin,
    DeepModelGlobalhParamsMixin,
    DeepModelCheckpointMixin,
    BaseModel[GraphDataBuilder] # BaseModel comes last for hierarchy of methods-imported
    ):
    """ 
    Parent class of Deep models (LSTM, GNNs, etc.).
    First degree sub-class of BaseModel

    Besides the parameters required by BaseModel, DeepModel instances require
    strategy: Strategy
        model strategies handle training and forecasting
    
    Methods
    -------
    - `load_model()`
        this classmethod allows the loading of a saved model by only feeding in the correct
        dataloadermanager. This is the only classmethod.
    - `save_model()`
        saves the model in a compatible way with `load_model()`.
    
    Attributes
    ----------
    _childclasses: Dict[str, Type["DeepModel"]]
        this is a class-level attribute. All models that inherit from DeepModel, i.e. all 
        deep - architectures, store their class in this dictionary through the 
        `__init_subclass__()`. This dictionary thus gives an overview of all deep - 
        architectures in this codebase.
    
    Examples
    --------
    >>> BASE_CONFIG: Dict[str, Any] = dict(
        disease              = 'influenza',
        country              = 'germany',
        level                = 'nuts3',
        min_date             = '2012-06-01',
        max_date             = '2020-06-01',
        feature_popsize      = False,
        feature_popdens      = True,
        feature_gisd         = False,
        feature_popage       = False,
        feature_kreise_classes = False,
        feature_borders      = False,
        sequence_length      = 4,
        normalization_method = 'zscore',
        log_transform        = ['incidence'],
    )

    >>> cfg = EpiConfig(**BASE_CONFIG, horizon_leadtime=1)
    >>> edo = EpiDataOrchestrator(cfg).build()
    >>> dlm = DeepDataLoaderManager(edo).build()

    >>> lstm = LSTMModel.load_model('lstm_hl1_s123',dlm,'experiment_1')
    
    See Also
    --------
    #### Parentclass
    For more information on basic model-behaviour shared among all models, see BaseModel.

    #### Subclasses
    DeepModel's subclasses are BaseModel's second degree subclasses, including LSTMModel and
    GNN-based architectures.

    #### Helperclasses
    This DeepModel uses the following mixin-classes: short libraries of methods that this class inherits.
    - DeepModelCheckpointMixin
        deals with model saving.
    - DeepModelForecastMixin
        deals with deepmodel - forecasting.
    - DeepModelGlobalhParamsMixin
        deals with global hyper - parameters
    - DeepModelInternalsMixin
        sets basic attributes called when instantiation.
    - DeepModelPresentationMixin
        deals with the (re)presentation of deep models.
        DeepModel overwrites BaseModel's `__repr__()`. The `__str__()` stays untouched.
    - DeepModelTrainMixin
        deals with model training
    """
    
    _childclasses:  Dict[str, Type["DeepModel"]] = {}    
    model:          torch.nn.Module 
    optimizer:      Optimizer
    scheduler:      _LRScheduler

    def __init__(self, 
                 dataloadermanager:     GraphDataBuilder, 
                 strategy:              Strategy,
                 name:                  str,          
                 verbose:               int = -1):

        # DeepModel in itself may not be initted
        if self.__class__ is DeepModel:
            raise TypeError("DeepModel cannot be instantiated directly")

        super().__init__(dataloadermanager = dataloadermanager, name = name, verbose = verbose)        
    
        self.evaluation_datasets                            = {}
        self._residual_quantiles: Dict[Tuple[int, int], Dict[int, float]] = {}


        # using hidden methods in DeepModelInternalsMixin, set attributes
        self._set_device()
        self._set_strategy(strategy)
        self._set_models_directory()

    # ======= DUNDER ======= #
    # __init_subclass__ is run when a subclass is iniated
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        DeepModel._childclasses[cls.__name__.lower()] = cls

    # ======= MAIN METHODS =========== #
    @classmethod
    def load_model(cls,
                   model_name:          str,
                   dataloadermanager:   GraphDataBuilder,
                   dir:                 Union[str, Path],
                   ) -> 'DeepModel':
        """
        Loads a saved model, and sets model hyper-parameters, global-hyperparameters
        and thus, most importantly, self.model. This function does not, however, 
        run `forecast()`! So this needs to be done after loading.

        Parameters
        ----------
        model_name: str
            name under which the model is saved (should be the filename wihtout .pt)
        dataloadermanager: Union[GraphDataLoaderManager, DeepDataLoaderManager]
            dataloadermanager with which the model was trained.
        subdir: Optional[str] = None
            directory in which to find the model. Directory may be named after an experiment.

        Returns
        -------
        This is the only method that returns the instance of the model.
        """

        if isinstance(dir, str):
            dir = Path(dir)
        
        # construct model path
        if model_name.endswith('.pt'):
            filepath = dir / model_name
        else:
            filepath = dir / f"{model_name}.pt"

        # validate model path's existence
        if not filepath.exists():
            raise FileNotFoundError(f"Model not found: {filepath}")

        # load dictionary
        save_dict: Dict[str, Any] = torch.load(filepath, map_location='cpu', weights_only=False)

        # get name of model architecture - class
        model_key: str = save_dict['model_class'].lower()

        # if class doesn't exist in deepmodel's child classes, then raise error
        if model_key not in cls._childclasses:
            raise ValueError(
                f"Unknown model class '{save_dict['model_class']}'. "
                f"Available: {list(cls._childclasses.keys())}"
            )

        # create an instance of model
        child_cls = cls._childclasses[model_key]
        instance  = child_cls(
            name              = save_dict['name'],
            dataloadermanager = dataloadermanager,
        ) # type: ignore
        
        dataloadermanager.dataorchestrator.config.assert_equals(save_dict['epiconfig_summary'], level = 1)

        # compare between raw input timestamps, not the preprocessed ones in temporal_summary
        saved_test_start  = pd.Timestamp(save_dict['epiconfig_summary']['split_valtest'])
        new_train_end     = pd.Timestamp(dataloadermanager.dataorchestrator.config.split_valtest)

        if new_train_end > saved_test_start:
            raise ValueError(
                f"Data leakage: new dataloader's train/val period ({new_train_end}) "
                f"overlaps with saved model's test period ({saved_test_start})"
            )

        # load config into the model
        instance.set_model_hparams(**save_dict['model_hparams'])
        instance.set_global_hparams(**save_dict['global_hparams'])
        instance.model.load_state_dict(save_dict['model_state'])
        instance.model.to(instance.device)
        instance.monitoring_metrics           = save_dict['monitoring_metrics']
        instance.config_info['model_hparams'] = save_dict['model_hparams']
        instance.config_info['global_hparams']= save_dict['global_hparams']
        instance._update_status('trained')

        logger.info("Model %s loaded from %s", model_name, dir)

        return instance

    def calibrate(self, window: int = 3) -> 'DeepModel':
        """
        Post-hoc quantile calibration on val data (conformal-style).
        Call after train() when using a point loss (mse/mae/huber) with epiconfig.quantiles set.

        Fits per-seasonal-timepoint residual quantiles using a rolling window of ±window weeks
        around each week-of-year, pooling across all nodes. This stabilises quantile estimates
        when validation spans only a single season.

        Parameters
        ----------
        window : int
            Number of weeks either side of each week-of-year to include in quantile estimation.
            Default 2 gives a ±2 week window (5 weeks × N nodes values per estimate).
        """
        self._check_status(['trained'])

        quantiles = self.epiconfig.quantiles
        if not quantiles:
            raise ValueError('calibrate() called but epiconfig.quantiles is None.')

        if self.loss.loss_name in ['pinball', 'pinchpinball']:
            raise ValueError(
                'calibrate() is for point-loss models. '
                'You are using pinball — no calibration needed.'
            )

        import numpy as np

        self.model.eval()
        val_loader  = self.dataloadermanager.dataloader_val
        time_splits = self.dataloadermanager.time_splits

        all_preds   = []
        all_targets = []

        with torch.no_grad():
            for snapshot in val_loader:
                snapshot    = snapshot.to(self.device)
                y_hat, _    = self.strategy.forecast_step(
                    model    = self.model,
                    snapshot = snapshot,
                    loss_fn  = self.loss
                )
                all_preds.append(y_hat.squeeze(-1).cpu())   # [num_nodes, horizon_size]
                all_targets.append(snapshot.y.cpu())         # [num_nodes, horizon_size]

        # stack: [num_val_timesteps, num_nodes, horizon_size]
        preds   = torch.stack(all_preds).numpy()
        targets = torch.stack(all_targets).numpy()

        # residuals: [num_val_timesteps, num_nodes, horizon_size]
        residuals = targets - preds

        # get seasonal index for each val timestep
        val_times      = time_splits[time_splits['val']][self.epiconfig.temporal_column].values
        freq           = self.dataloadermanager.dataorchestrator.config.temporal_frequency
        val_timestamps = pd.to_datetime(val_times)

        if freq == 'w':
            t_idx = val_timestamps.isocalendar().week.astype(int).values
        elif freq == 'm':
            t_idx = val_timestamps.month.values
        elif freq == 'd':
            t_idx = val_timestamps.isocalendar().day.astype(int).values
        else:
            raise ValueError(f'Unknown temporal frequency: {freq}')

        unique_t_vals = np.unique(t_idx)
        num_timesteps, num_nodes, horizon_size = residuals.shape

        # ── fit residual quantiles per (horizon, quantile, seasonal_idx) ──
        # rolling window pools residuals from weeks within ±window of each week
        # giving (2*window+1) × num_nodes values per estimate

        residual_quantiles: Dict[Tuple[int, int], Dict[int, float]] = {}

        for hh in range(horizon_size):
            for q_idx, q in enumerate(quantiles):
                seasonal_quantiles: Dict[int, float] = {}

                for unique_t in unique_t_vals:

                    # build circular window around unique_t
                    # handle year wrap-around (e.g. week 51, 52, 1, 2, 3)
                    max_t   = unique_t_vals.max()
                    min_t   = unique_t_vals.min()
                    range_t = max_t - min_t + 1

                    window_weeks = set()
                    for delta in range(-window, window + 1):
                        w = unique_t + delta
                        # wrap around
                        if w > max_t:
                            w = min_t + (w - max_t - 1)
                        elif w < min_t:
                            w = max_t - (min_t - w - 1)
                        window_weeks.add(w)

                    # mask: all timesteps whose week-of-year falls in the window
                    mask      = np.isin(t_idx, list(window_weeks))
                    res_slice = residuals[mask, :, hh].ravel()  # [(2w+1) timesteps × nodes]

                    if len(res_slice) == 0:
                        seasonal_quantiles[unique_t] = 0.0
                        continue

                    seasonal_quantiles[unique_t] = float(np.quantile(res_slice, q))

                residual_quantiles[(hh, q_idx)] = seasonal_quantiles

        self._residual_quantiles  = residual_quantiles
        self._calibration_window  = window
        self._calibration_t_idx   = t_idx
        self._calibration_unique_t = unique_t_vals
        self._calibration_freq    = freq

        if self.verbose >= 0:
            n_per_estimate = (2 * window + 1) * num_nodes
            print(
                f'✓ calibrate() complete | '
                f'window=±{window} weeks | '
                f'~{n_per_estimate} residuals per quantile estimate | '
                f'{len(quantiles)} quantiles × {horizon_size} horizons × '
                f'{len(unique_t_vals)} seasonal indices'
            )

        return self

    def set_model_hparams(self):
        """must be set by subclasses"""
        raise NotImplementedError("Subclass of DeepModel must implement set_model_hparams")      