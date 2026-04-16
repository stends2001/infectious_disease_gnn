from typing import Dict,  Union, Optional, Type
import torch 
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path 
import os

from .internals import DeepModelInternalsMixin
from .logging import DeepModelLoggerMixin
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
    DeepModelLoggerMixin,
    DeepModelTrainMixin,
    DeepModelForecastMixin,
    DeepModelGlobalhParamsMixin,
    DeepModelCheckpointMixin,
    BaseModel[Union[GraphDataLoaderManager, DeepDataLoaderManager]],
):
    """ 
    # TODO
    """
    
    # This is run at runtime. dictionary of all models that inherit from here
    _childclasses: Dict[str, Type["DeepModel"]] = {}
    
    model:      torch.nn.Module 
    optimizer:  Optimizer
    scheduler:  _LRScheduler

    def __init__(self, 
                 dataloadermanager:     Union[GraphDataLoaderManager, DeepDataLoaderManager], 
                 strategy:              Strategy,
                 name:                  str,          
                 verbose:               int = -1):

        super().__init__(dataloadermanager = dataloadermanager, name = name, verbose = verbose)        
        
        self.monitoring_metrics                             = None
        self.evaluation_datasets                            = {}

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
                   dataloadermanager,
                   subdir:            Optional[str] = None,
                   ) -> 'DeepModel':

        # build path — use class-level helper, not instance attribute
        base_dir = Path(os.path.join(get_project_utilities_env(), 'models'))
        base     = base_dir / subdir if subdir else base_dir

        if model_name.endswith('.pt'):
            filepath = base / model_name
        else:
            filepath = base / f"{model_name}.pt"

        if not filepath.exists():
            raise FileNotFoundError(f"Model not found: {filepath}")

        save_dict = torch.load(filepath, map_location='cpu', weights_only=False)

        model_key = save_dict['model_class'].lower()
        if model_key not in cls._childclasses:
            raise ValueError(
                f"Unknown model class '{save_dict['model_class']}'. "
                f"Available: {list(cls._childclasses.keys())}"
            )

        child_cls = cls._childclasses[model_key]
        instance  = child_cls(
            name              = save_dict['name'],
            dataloadermanager = dataloadermanager,
        ) # type: ignore

        instance.set_model_hparams(**save_dict['model_hparams'])
        instance.set_global_hparams(**save_dict['global_hparams'])
        instance.model.load_state_dict(save_dict['model_state'])
        instance.model.to(instance.device)
        instance.monitoring_metrics           = save_dict.get('monitoring_metrics')
        instance.config_info['model_hparams'] = save_dict['model_hparams']
        instance.config_info['global_hparams']= save_dict['global_hparams']
        instance._update_status('trained')

        return instance

    def set_model_hparams(self):
        raise NotImplementedError("Subclass of DeepModel must implement set_model_hparams")

    def debug(self):
        if self.model is None:
            raise ValueError('Please initiate a model')

        self._check_status(['model_hparams_set'])

        train_sample = self.dataloadermanager.dataloader_train[0].to(self.device)
        y_hat , report = self.strategy.debug(self.model, train_sample)

        y_hat = y_hat.detach().cpu()
        y     = train_sample.y.detach().cpu()
        report.validate()
        # if y_hat.shape != y.shape:
        #     raise DeepModelDebuggingError(f"incompatible prediction shape: y_hat [{y_hat.shape}], y [{y.shape}]")
        
