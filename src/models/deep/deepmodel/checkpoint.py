from typing import Dict, Optional, Any, TYPE_CHECKING
import torch 
from torch import Tensor as Tensor
from pathlib import Path

from ....utils import PathNotFound

if TYPE_CHECKING:
    from ....dataloading.epiconfig import EpiConfig

import logging
logger = logging.getLogger(__name__)

class DeepModelCheckpointMixin:
    """ 
    Mixin class that deals with the saving of DeepModels.
    NOTE `load_model()` is a classmethod defined in DeepModel itself.
    """    
    model:              torch.nn.Module 
    clean_name:         str
    models_dir:         Path
    config_info:        Dict[str, Any]
    verbose:            int
    epiconfig:          'EpiConfig'

    def save_model(self, dir: Path):
        """
        save current model in subdirectory (optionally).
        could be named after an experiment.
        """
        if not hasattr(self, 'model'):
            raise ValueError('No model found!')
        
        # if parent of dir doesn't exist, error
        if not dir.parent.exists():
            raise PathNotFound(f"parent of dir {dir} does not exist")

        filepath = dir / f"{self.clean_name}.pt"

        save_dict: Dict[str, Any] = {
            'name':               self.clean_name,            
            'model_class':        self.__class__.__name__,
            'model_state':        self.model.state_dict(),
            'model_hparams':      self.config_info.get('model_hparams', {}),
            'global_hparams':     self.config_info.get('global_hparams', {}),
            'epiconfig_summary':  self.epiconfig.get_summary(level = 1),
            'monitoring_metrics': getattr(self, 'monitoring_metrics', None),
        }    

        torch.save(save_dict, filepath)
        logger.info("Model %s saved to %s", self.clean_name, dir)