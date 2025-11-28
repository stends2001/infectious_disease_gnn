
from typing import List, Dict, Optional, Union
import yaml
import os
import json
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..models._basemodel import BaseModel

from ._baseconfigmanager import ConfigManager

from ..utils.helpers import list_files, write_yaml_file, to_underscore_string,reorder_dict

class ModelConfigManager(ConfigManager):
    """
    Manages configurations of:

    - models (descriptions, not weights!)
    """

    def __init__(self, base_dir: str = "config/models/descriptions/"):
        self.base_dir               = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)   


        self.default_config_order   = ['name', 'id', 'child', 'model', 
                                       'global_hparams', 'model_hparams', 'task']


             
    
    # def register_model(self, model: 'BaseModel') -> str:
    #     """
    #     Register model configuration using its name.
        
    #     Returns:
    #     -------
    #     str : The assigned model ID
    #     """
    #     model_id = self._generate_model_id()
        
    #     config = model.config_info.copy()
    #     config['id'] = model_id
        
    #     modelname = config['name']
    #     if not modelname or modelname == 'unknown':
    #         raise ValueError("Model must have a valid name before registration")
        
    #     modelname_safe = to_underscore_string(modelname)
        
    #     if not self._validate_registration(modelname_safe):
    #         raise ValueError(
    #             f"Model '{modelname_safe}' already registered. "
    #             f"Use a different name or update the existing config."
    #         )
        
    #     # Reorder config for readability
    #     config_reordered = reorder_dict(config, self.default_config_order)
        
    #     # Save config
    #     write_yaml_file(config_reordered, self.config_models_dir, modelname_safe)
        
    #     print(f"✓ Config saved: {modelname} -> {modelname_safe}.yaml (ID: {model_id})")
        
    #     return model_id

    # def load_config(self, model_name: str) -> Dict:
    #     """Load a model configuration by name"""
    #     config_path = self.config_models_dir / f"{model_name}.yaml"
        
    #     if not config_path.exists():
    #         raise FileNotFoundError(f"Config not found: {config_path}")
        
    #     with open(config_path, 'r') as f:
    #         return yaml.safe_load(f)
    

    
        
