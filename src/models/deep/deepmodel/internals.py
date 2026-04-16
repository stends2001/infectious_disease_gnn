import torch 
from pathlib import Path 
import os 

from ..strategies.basestrategy import Strategy
from ...issues import DeviceWarning
from ....utils.helpers import get_project_utilities_env

class DeepModelInternalsMixin:
    """ 
    # TODO
    """
    def _set_strategy(self, strategy: Strategy):
        """Allow subclasses to specify their strategy"""
        self.strategy = strategy

    def _set_device(self):
        """sets attribute device"""
        self.device            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.device == 'cpu':
            w = DeviceWarning('device found is CPU')
            print(w)    

    def _set_models_directory(self):
        base_dir = Path(os.path.join(get_project_utilities_env(), 'models')) 
        base_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = base_dir  
