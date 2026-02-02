from typing import Optional, List, Dict
from pathlib import Path
import os
import numpy as np
import yaml

def to_underscore_string(s: str) -> str:
    """Convert string to underscore format"""
    return s.replace(' ', '_').replace('-', '_').lower()

def list_files(dir: Path, extension: Optional[str] = None) -> List[str]:
    """List files in directory with optional extension filter"""
    files_list = []
    for filepath in dir.iterdir():
        if filepath.is_file():
            if extension is None or filepath.suffix == extension:
                files_list.append(filepath.stem)
    return files_list

def write_yaml_file(yaml_dictionary: Dict, dir: Path, filename: str):
    """Write dictionary to YAML file"""
    if not filename.endswith('.yaml'):
        filename = filename + '.yaml'
    
    filepath = dir / filename
    with open(filepath, 'w') as file:
        yaml.dump(yaml_dictionary, file, sort_keys=False, default_flow_style=False)

def reorder_dict(d: dict, elements: List[str]) -> dict:
    """Reorder dictionary keys"""
    reordered = {}
    for key in elements:
        if key in d:
            reordered[key] = d[key]
    for key in d:
        if key not in reordered:
            reordered[key] = d[key]
    return reordered

def get_wissdaten_env() -> str:
    tmpdir          = os.environ.get('TMPDIR', None)
    if tmpdir is None:
        raise ValueError('No temporary directory found!')
    
    wissdaten_dir           = os.path.join(tmpdir, 'wissdaten')
    personal_wissdaten_dir  = os.path.join(wissdaten_dir, 'ZKI-PH4/deschrijvers_wissdaten')
    return personal_wissdaten_dir

def get_outcomes_env() -> str:
    outcomes_env        = os.path.join(get_wissdaten_env(), 'outcomes')
    return outcomes_env

def get_project_utilities_env() -> str:
    proj_utils_env        = os.path.join(get_wissdaten_env(), 'project_utilities/infectious_disease_gnn')
    return proj_utils_env

def get_data_env() -> str:
    data_env        = os.path.join(get_wissdaten_env(), 'data')
    return data_env

def sum_preserve_nan(x):
    return x.sum() if x.notna().any() else np.nan