
from typing import List, Dict, Optional, Union
import yaml
import os
import json
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..models._basemodel import BaseModel

from ..utils.helpers import list_files, write_yaml_file, to_underscore_string,reorder_dict


class ConfigManager:

    """
    Parent class for the configmanagers.
    Inheritance into:
    - ExperimentConfigManager
    - ModelConfigManager
    
    """

    def __init__(self, base_dir: str = "config/"):
        self.base_dir               = Path(base_dir)

    def return_entries(self) -> List[str]:
        """Return list of model names registered"""
        return list_files(self.base_dir, extension='.yaml')

    def register_entry(self, config: dict) -> str:
        """
        Registers a configuration entry 
        
        Returns:
        -------
        str : The assigned model ID
        """
        try:
            entry_name = config['name']

        except KeyError:
            raise KeyError("The required key 'name' is missing in the config dictionary.")

        if not self.validate_entry(entry_name):
            raise ValueError(f'{entry_name} already saved in the registry')

        entry_id = self._generate_entry_id()
        
        config       = config.copy()
        config['id'] = entry_id
    
        
        entry_name_safe = to_underscore_string(entry_name)
        
        # Reorder config for readability => put name and id before anything else
        config_reordered = reorder_dict(config, ["name","id"])
        
        # Save config
        write_yaml_file(config_reordered, self.base_dir, entry_name_safe)

        print(f"✓ Config saved: {entry_name} -> {entry_name_safe}.yaml (ID: {entry_id})")
        return entry_id

    def _generate_entry_id(self) -> str:
        """Generate next sequential model ID"""
        num = len(self.return_entries()) + 1
        return f"{num:04d}"  # 0001, 0002, etc.        

    def load_entry(self, entry_name: str) -> Dict:
        """Load a model configuration by name"""
        config_path = self.base_dir / f"{entry_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
   
    def delete_entry(self, entry_name: str):
        """Delete a model configuration"""
        config_path = self.base_dir / f"{entry_name}.yaml"
        if config_path.exists():
            config_path.unlink()
            print(f"✓ Config deleted: {entry_name}")
    
    def validate_entry(self, name: str) -> bool:
        """Check if model name is available"""
        return name not in self.return_entries()

class ExperimentConfigManager(ConfigManager):
    """
    Manages configurations of:

    - experiments
    """

    def __init__(self, base_dir: str = "config/experiments/"):
        self.base_dir  = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)   

        # self.default_config_order   = ['name', 'id', 'child', 'model', 
                                    #    'global_hparams', 'model_hparams', 'task']        

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
    

    
        
