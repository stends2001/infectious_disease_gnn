
from typing import List, Dict, Optional, Union
import yaml
import os
import json
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..models.base.basemodel import BaseModel

from ..utils.helpers import list_files, write_yaml_file, to_underscore_string,reorder_dict


class ConfigManager:

    """
    Parent class for the configmanagers.

    Downstream
    ----------
    The following children inherit from this class
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
        str : The assigned ID
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
