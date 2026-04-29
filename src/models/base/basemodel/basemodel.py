from typing import Generic, Dict, Any, Literal

from ..predictions import PredictionManager
from .forecastdisplaymixin import ForecastDisplayMixin
from .appearancemixin import ModelAppearanceMixin
from .presentationmixin import PresentationMixin
from .statusmixin import ModelStatusMixin

from ...issues import ModelInitError
from ....dataloading.dataloaders import DLM

class BaseModel(Generic[DLM], ModelStatusMixin, PresentationMixin, ModelAppearanceMixin, ForecastDisplayMixin[DLM]):
    """ 
    Parent class of ALL models.
    Upon init, all models must supply the following

    Parameters
    ----------
    dataloadermanager: Union[BaseLineDataLoaderManager, DeepDataLoaderManager, GraphDataLoaderManager]
        dataloadermanager that stores the data in a way required by the model type.
    name: str
        model name
    verbose: int = -1
        the extend of which output will be returned. The following levels for verbose (v) are distinguished:
        - v < 0:        no output           =>  no output
        - 0 <= v < 1    minimal output      =>  minimal status updates, # TODO
        - 1 <= v < 2    intermediate output =>  full status updates, # TODO
        - v >= 2        maximal output      =>  full status updates, # TODO

    Methods
    -------
    - show_forecasts
        main method to show a model's forecasts.
    
    Attributes
    ----------
    Important attributes include:
    - config_info: Dict[str, Any]
        stores configuration data essential for model-saving and model-loading.
    - status_dict: Dict[ModelStatus, bool]
        stores model status data (whether certain checkpoints are done). Handled by ModelStatusMixin.
    
    Examples
    --------
    As this model class may not be initiated by itself, instead, a subclass (of degree 2) must be initiated,
    there is no relevant example for this class.
    
    See Also
    --------
    
    #### Subclasses
    BaseModel has subclasses of degree 1 (model - families) and degree 2 (model - classes/architectures).
    The first degree of subclasses store general code, that is specifically tailored to its architectures,
    but different to the architectures of the other the other classes in degree 1. While the BaseLineModel
    is relatively short and easy to follow, DeepModel is much more extensive.
    
    - BaseLineModel
        - PersistenceModel
        - ClimateologyModel
        - ClimaScaleModel
        - ConstantModel
    - DeepModel
        - LSTMModel
        - ...

    #### Helperclasses
    This BaseModel uses the following mixin-classes: short libraries of methods that this class inherits.
    - ModelStatusMixin
        contains hidden methods that deal with `status_dict`.
    - PresentationMixin
        contains dunder and hidden methods that deal with (re)presentation.
    - ModelAppearanceMixin
        contains hidden methods used to set attributes in `self.__init__()`
    - ForecastDisplayMixin
        contains public method `show_forecasts()` and supportive hidden methods.

    Additionally, the PredictionManager is of importance. This is where predictions are stored and interacted with.
    """
    _expected_dataloadermanager:    Literal['BaseLineDataLoaderManager', 'DeepDataLoaderManager', 'GraphDataLoaderManager'] 
    config_info:                    Dict[str, Any]

    def __init__(self, 
                 dataloadermanager: DLM, 
                 name:              str,
                 verbose:           int  = -1):

        # BaseModel in itself may not be initted
        if self.__class__ is BaseModel:
            raise TypeError("BaseModel cannot be instantiated directly")

        self.name = name
        self._set_dataloader_attributes(dataloadermanager)

        # static attributes
        self.verbose                    = verbose
        self.model_class                = self.__class__.__name__
        self.model_color                = self._get_model_color()
        self.clean_name                 = self._get_clean_name()
        
        # validate dataloadermanager-type vs model-type
        # if unexpected -> Error is raised
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
        