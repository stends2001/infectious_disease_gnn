from typing import Generic, Dict, Any

from .predictions_manager import PredictionManager
from .forecastdisplaymixin import ForecastDisplayMixin
from .appearancemixin import ModelAppearanceMixin
from .verbosemixin import ModelVerboseMixin
from .statusmixin import ModelStatusMixin

from ..issues import ModelInitError
from ...dataloading.dataloaders import DLM

class BaseModel(Generic[DLM], ModelStatusMixin, ModelVerboseMixin, ModelAppearanceMixin, ForecastDisplayMixin[DLM]):
    """ 
    Parent class of ALL models.
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
    BaseModel has subclasses:
    - BaseLineModel
    - DeepModel

    Each of which has further subclasses.
    """
    _expected_dataloadermanager:    str 
    config_info:                    Dict[str, Any]

    def __init__(self, 
                 dataloadermanager: DLM, 
                 name:              str,
                 verbose:           int             = -1):

        self.name = name
        self._set_dataloader_attributes(dataloadermanager)

        # static attributes
        self.verbose                    = verbose
        self.model_class                = self.__class__.__name__
        self.model_color                = self._get_model_color()
        self.clean_name                 = self._get_clean_name()
        # validate dataloadermanager-type vs model-type
        self._validate_dataloadermanager()
        
        # dynamic (changing) attributes
        self.predictions                = PredictionManager(self.dataloadermanager.dataorchestrator, self.column_registration, self.temporal_summary)
        self.weights_manager            = None
        
        # Configuration - info
        self.config_info                = {}        
        self.config_info                = {'name': self.name, 'model_class': self.model_class}

        self._init_status()
        self._update_status('model_initialized')
        self._print_status_update('model_initialized')

    # ======== HIDDEN METHODS ========= #
    def _set_dataloader_attributes(self, dataloadermanager: DLM):
        """An extention upon init: sets a range of easy to access attributes related to dataloadermanager"""
        self.dataloadermanager          = dataloadermanager
        self.epiconfig                  = self.dataloadermanager.dataorchestrator.config
        self.column_registration        = dataloadermanager.dataorchestrator.column_registration
        self.context_data               = dataloadermanager.dataorchestrator.data_context
        self.temporal_summary           = self.context_data.temporal_summary
        self.prediction_mode            = self.dataloadermanager.dataorchestrator.config.prediction_mode
        self.pred_cols                  = self._get_pred_cols()

    def _validate_dataloadermanager(self):
        """validate class of dataloadermanager"""
        if not hasattr(self, '_expected_dataloadermanager'):
            raise ModelInitError(f'attribute self._expected_dataloadermanager not set in model {self.name}')

        exp = self._expected_dataloadermanager
        got = self.dataloadermanager.__class__.__name__

        if exp != got:
            raise ModelInitError(f'{self.name} expected a dataloadermanager of class {exp} but got {got}')

    # ======== METHODS TO BE IMPLEMENTED BY SUBCLASSES ======== #
        # NOTE 
        # I'm using NotImplementedErrors over ABC-abstractmethods since 
        # some model-types need more arguments than other model-types.
    
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement train")

    def forecast(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement forecast")

    def set_global_hparams(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement set_global_hparams")

    def set_model_hparams(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement set_model_hparams")

    def save_model(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement save_model")
        