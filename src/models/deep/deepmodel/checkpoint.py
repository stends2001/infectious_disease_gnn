from typing import Dict, Optional, Any, TYPE_CHECKING
import torch 
from torch import Tensor as Tensor
from pathlib import Path

if TYPE_CHECKING:
    from ....dataloading.epiconfig import EpiConfig

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

    def save_model(self, dir: Optional[Path] = None):
        """
        save current model in subdirectory (optionally).
        could be named after an experiment.
        """
        if not hasattr(self, 'model'):
            raise ValueError('No model found!')

        base_dir    = self.models_dir 
        sub_dir     = dir

        # if base dir doesn't exist, error
        if not base_dir.exists():
            raise FileNotFoundError(f".base_dir {base_dir} does not exist")

        if sub_dir is not None:
            full_sub_dir = base_dir / sub_dir
            # if subdir doesn't exist, make it
            full_sub_dir.mkdir(exist_ok=True)
            filepath = full_sub_dir / f"{self.clean_name}.pt"

        else:
            filepath = base_dir / f"{self.clean_name}.pt"

        save_dict: Dict[str, Any] = {
            'name':               self.clean_name,            
            'model_class':        self.__class__.__name__,
            'model_state':        self.model.state_dict(),
            'model_hparams':      self.config_info.get('model_hparams', {}),
            'global_hparams':     self.config_info.get('global_hparams', {}),
            'epiconfig':          self.epiconfig.get_summary(level = 1),
            'monitoring_metrics': getattr(self, 'monitoring_metrics', None),
        }    

        torch.save(save_dict, filepath)
        
        if self.verbose >= 0:
            local_path = str(filepath).split("/wissdaten/")[1]
            print(f"✓ Model saved: Wissdaten/{local_path}")
