from typing import Dict,  Union, Optional, Type, Any
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

from ....utils.helpers import get_project_utilities_env
from ....dataloading.dataloaders import DeepDataLoaderManager, GraphDataLoaderManager
from ..strategies.basestrategy import Strategy
from ...base.basemodel import BaseModel
    
class DeepModel(
    DeepModelInternalsMixin,
    DeepModelPresentationMixin,
    DeepModelTrainMixin,
    DeepModelForecastMixin,
    DeepModelGlobalhParamsMixin,
    DeepModelCheckpointMixin,
    BaseModel[Union[GraphDataLoaderManager, DeepDataLoaderManager]] # BaseModel comes last for hierarchy of methods-imported
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
                 dataloadermanager:     Union[GraphDataLoaderManager, DeepDataLoaderManager], 
                 strategy:              Strategy,
                 name:                  str,          
                 verbose:               int = -1):

        # DeepModel in itself may not be initted
        if self.__class__ is DeepModel:
            raise TypeError("DeepModel cannot be instantiated directly")

        super().__init__(dataloadermanager = dataloadermanager, name = name, verbose = verbose)        
    
        self.evaluation_datasets                            = {}

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
                   model_name:        str,
                   dataloadermanager: Union[GraphDataLoaderManager, DeepDataLoaderManager],
                   subdir:            Optional[str] = None,
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

        # build path — use class-level helper, not instance attribute
        base_dir = Path(os.path.join(get_project_utilities_env(), 'models'))
        base     = base_dir / subdir if subdir else base_dir
        
        # construct model path
        if model_name.endswith('.pt'):
            filepath = base / model_name
        else:
            filepath = base / f"{model_name}.pt"

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

        saved_test_start  = pd.Timestamp(save_dict['epiconfig_summary']['split_valtest'])
        new_train_end     = dataloadermanager.dataorchestrator.data_context.temporal_summary.split_valtest

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

        return instance

    def set_model_hparams(self):
        """must be set by subclasses"""
        raise NotImplementedError("Subclass of DeepModel must implement set_model_hparams")      