from typing import TYPE_CHECKING
import torch 
from pathlib import Path 
import os 

from ..strategies.basestrategy import Strategy

class DeepModelInternalsMixin:
    """ 
    Mixin class that deals with the setting of attributes.
    These methods are called by DeepModel's init function to set attributes.
    """
    def _set_strategy(self, strategy: Strategy):
        """sets strategy"""
        self.strategy = strategy

    def _set_device(self):
        """sets device"""
        self.device            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.device.type == 'cpu':
            print('device found is CPU')

    def _set_models_directory(self):
        """sets model directory"""
        base_dir = Path('data/experiment_outcomes')
        base_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = base_dir  
