from typing import Dict, Any, Union, Optional, List, TYPE_CHECKING
import torch 
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ...issues import InvalidOPtimizerError, InvalidSchedulerError
from ...utils.loss.losshandler import LossHandler
from ...base.statusmixin import ModelStatus

if TYPE_CHECKING:
    from ....dataloading.epiconfig import EpiConfig

class DeepModelGlobalhParamsMixin:
    """ 
    # TODO
    """    
    status_dict:        Dict[ModelStatus, bool]
    epiconfig:          'EpiConfig'
    config_info:        Dict[str, Any]    
    model:              torch.nn.Module
    n_params:           int

    # ======== STUBS ======= #
    def _check_status(self, required_states: Union[List[ModelStatus], ModelStatus]) -> None: ...
    def _update_status(self, status: ModelStatus) -> None: ...

    def set_global_hparams(self, 
                           lr:              float           = 0.001,
                           n_epochs:        int             = 5,
                           patience:        int             = 15,
                           min_delta:       float           = 1e-4,                            
                           optimizer:       str             = 'adam',
                           loss:            str             = 'mse',                           
                           scheduler:       Optional[str]   = 'step',
                           # kwargs
                           optimizer_kwargs:Optional[Dict[str, Any]] = None,                           
                           scheduler_kwargs:Optional[Dict[str, Any]] = None,
                           loss_kwargs:     Optional[Dict[str, Any]] = None                            
                           ):
        """
        Prepares model for training using global hyperparameters
        
        Parameters
        ---------

        """
        self._check_status(['model_hparams_set'])

        global_params_config = {
            'lr'                : lr,
            'n_epochs'          : n_epochs,
            'patience'          : patience,
            'min_delta'         : min_delta,                       
            'optimizer'         : optimizer,
            'loss'              : loss,
            'scheduler'         : scheduler,

            'optimizer_kwargs'  : optimizer_kwargs,
            'scheduler_kwargs'  : scheduler_kwargs,
            'loss_kwargs'       : loss_kwargs
        }
        
        # ==== CONSTANTS ===== #
        self.n_epochs           = n_epochs
        self.patience           = patience
        self.min_delta          = min_delta

        # ==== LOSS ==== #
        if loss in ['pinball','pinchpinball']:
            if loss_kwargs is None:
                loss_kwargs = {}

            if 'quantiles' not in loss_kwargs.keys():
                loss_kwargs['quantiles'] = self.epiconfig.quantiles

        self.loss       = LossHandler(loss, loss_kwargs = loss_kwargs)  

        # ==== OPTIMIZER ==== #
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer = self._get_optimizer(optimizer, lr, optimizer_kwargs)
        
        # ==== SCHEDULER ====== #
        if scheduler_kwargs is None:
            default_scheduler_kwargs = {
                'step':        {'step_size': 15, 'gamma': 0.8},
                'exponential': {'gamma': 0.95},
                'cosine':      {'T_max': 50},
                'cosine_warm': {'T_0': 10, 'T_mult': 2},                
                'plateau':     {'mode': 'min', 'factor': 0.5, 'patience': 10, 'verbose': True}
            }
            scheduler_kwargs = default_scheduler_kwargs.get(scheduler, {}) if scheduler else {}

        if scheduler:
            self.scheduler = self._get_scheduler(scheduler, self.optimizer, scheduler_kwargs)

        else:
            self.scheduler = None

        self._validate_global_hparams()
        self.config_info['global_hparams']  = global_params_config
        self._update_status('global_hparams_set')

    def _validate_global_hparams(self):        
        if self.epiconfig.quantiles:
            if self.loss.loss_name not in ['pinball','pinchpinball']:
                raise ValueError('quantiles given to DeepModel, yet loss is not pinball.')

    def _get_optimizer(self, 
                       optimizer_name:  str, 
                       lr:              float, 
                       optimizer_kwargs:Dict[str, Any]) -> Optimizer:
        """Factory method to create and return optimizer"""
        
        self._check_status(['model_hparams_set'])   

        # pylance struggles with torch typing?
        optimizer_map = {
            'adam':    optim.Adam,     # type: ignore
            'adamw':   optim.AdamW,    # type: ignore
            'sgd':     optim.SGD,      # type: ignore
            'rmsprop': optim.RMSprop,  # type: ignore
            'adagrad': optim.Adagrad,  # type: ignore
        }
        
        if optimizer_name.lower() not in optimizer_map:
            raise InvalidOPtimizerError(optimizer_name, list(optimizer_map.keys()))
        
        optimizer_class = optimizer_map[optimizer_name.lower()]

        return optimizer_class(self.model.parameters(), lr=lr, **optimizer_kwargs)

    def _get_scheduler(self, scheduler_name: str, optimizer: Optimizer, scheduler_kwargs: Dict[str, Any]) -> _LRScheduler:
        """Factory method to create and return scheduler"""
        
        self._check_status(['model_hparams_set'])
        
        scheduler_map = {
            'step':        torch.optim.lr_scheduler.StepLR,
            'exponential': torch.optim.lr_scheduler.ExponentialLR,
            'cosine':      torch.optim.lr_scheduler.CosineAnnealingLR,
            'cosine_warm': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,            
            'plateau':     torch.optim.lr_scheduler.ReduceLROnPlateau,
            'cyclic':      torch.optim.lr_scheduler.CyclicLR,
            'onecycle':    torch.optim.lr_scheduler.OneCycleLR,
            'multistep':   torch.optim.lr_scheduler.MultiStepLR,
            'lambda':      torch.optim.lr_scheduler.LambdaLR,
        }
        
        if scheduler_name.lower() not in scheduler_map:
            raise InvalidSchedulerError(scheduler_name, list(scheduler_map.keys()))
        
        scheduler_class = scheduler_map[scheduler_name.lower()]
        return scheduler_class(optimizer, **scheduler_kwargs)
