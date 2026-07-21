from typing import Optional, List, Dict, Union, Any
from pathlib import Path
import os
import numpy as np
import yaml
import pickle, json, copy

from .exceptions import PathNotFound

# ========== #
#    SETS    #
# ========== #
def compare_sets(set1: set, set2: set):

    if set1 != set2:
        missing = set(set1) - set(set2)
        leftover= set(set2) - set(set1)

        raise UnequalSetsError(f'missing from set 2 are: {missing}. Leftover in set 1 are: {leftover}')
        
def load_mapping_dict(path: Union[str, Path]) -> Dict[Any, Any]:

    if isinstance(path, str):
        path = Path(path)

    match path.suffix:
        case ".pkl":
            with open(path, "rb") as f:
                data = pickle.load(f)        
        case ".json":
            with open(path, "r") as f:
                data = json.load(f)
        case _:
            raise ValueError('the only type of files suppported in load_mapping_dict are ".pkl" and "json"')

    if type(data) != dict:
        raise ValueError(f'expected dictionary. got {type(data)}')
    return copy.deepcopy(data)

def save_mapping_dict(dictionary: Dict[Any,Any], path: Union[str, Path]):

    if isinstance(path, str):
        path = Path(path)

    if not path.parent.exists():
        raise PathNotFound(path.parent)

    match path.suffix:
        case ".pkl":
            raise ValueError('you should probably save this file as a .json instead!')
        case ".json":
            with open(path, "w") as f:
                json.dump(dictionary,f,indent = 4)
        case _:
            raise ValueError('the only type of files suppported in load_mapping_dict are ".pkl" and "json"')

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

# ========================= #
# DONT NEED TO BE FUNCTIONS #
# ========================= #

def to_underscore_string(s: str) -> str:
    """Convert string to underscore format"""
    return s.replace(' ', '_').replace('-', '_').lower()


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