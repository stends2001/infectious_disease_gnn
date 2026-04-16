from typing import Dict, Optional, Any
import torch 
from torch import Tensor as Tensor
from pathlib import Path
import pandas as pd 

class DeepModelCheckpointMixin:
    """ 
    # TODO
    """    
    model:              torch.nn.Module 
    clean_name:         str
    models_dir:         Path
    config_info:        Dict[str, Any]
    verbose:            int

    def save_model(self, dir: Optional[Path] = None):
        """
        # TODO
        """
        if not hasattr(self, 'model'):
            raise ValueError('No model found!')

        base_dir    = self.models_dir 
        sub_dir     = dir

        if not base_dir.exists():
            raise FileNotFoundError(f".base_dir {base_dir} does not exist")

        if sub_dir is not None:
            full_sub_dir = base_dir / sub_dir
            full_sub_dir.mkdir(exist_ok=True)  # creates if not exists, errors if base_dir missing
            filepath = full_sub_dir / f"{self.clean_name}.pt"

        else:
            filepath = base_dir / f"{self.clean_name}.pt"

        save_dict = {
            'name':               self.clean_name,            
            'model_class':        self.__class__.__name__,
            'model_state':        self.model.state_dict(),
            'model_hparams':      self.config_info.get('model_hparams', {}),
            'global_hparams':     self.config_info.get('global_hparams', {}),
            'monitoring_metrics': getattr(self, 'monitoring_metrics', None),
        }    

        torch.save(save_dict, filepath)
        
        if self.verbose >= 0:
            local_path = str(filepath).split("/wissdaten/")[1]
            print(f"✓ Model saved: Wissdaten/{local_path}")
