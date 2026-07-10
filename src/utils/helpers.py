from typing import Optional, List, Dict, Union 
from pathlib import Path
import os
import numpy as np
import yaml
import pickle, json, copy
from functools import wraps
import inspect
from typing import Iterable, Any

from typing import Literal, Callable, List
import inspect

class MethodNotInRegistry(Exception):

    def __init__(self, method: str, available_methods: List[str]):
        msg = f'Unknown method {method}. Available methods are: {available_methods}'
        super().__init__(msg)


def registry_method(func: Callable) -> Callable:
    """
    Decorator that marks a method as a registered one.
    Used to create a selection of methods for graph-building
    and graph-normalization.
    """
    setattr(func, '_is_registered_method', True)
    return func


def get_registered_methods(cls: type) -> List[str]:
    """
    Returns the names of all methods decorated with @registry_method
    for a given class.
    """
    return [
        name for name, func in inspect.getmembers(cls, predicate=inspect.isfunction)
        if getattr(func, '_is_registered_method', False)
    ]

class UnequalSetsError(Exception):

    def __init__(self, message: str):
        super().__init__(message)

def compare_sets(set1: set, set2: set):

    if set1 != set2:
        missing = set(set1) - set(set2)
        leftover= set(set2) - set(set1)

        raise UnequalSetsError(f'missing from set 2 are: {missing}. Leftover in set 1 are: {leftover}')
        

class PathNotFound(Exception):

    def __init__(self, path: Union[str, Path]):
        super().__init__(f'Path was not found: {path}')

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


class InvalidDataSetError(Exception):
    def __init__(self, input: str):
        super().__init__(f"Invalid value for dataset: {input}. Expected one of ['train','val','test']")    

# function receives arguments for decorator
def check_dataset(allowed_values: Iterable =("train", "val", "test"), arg_name: str="dataset"):
    """
    Decorator that checks if the function argument `arg_name` is in `allowed`
    Here I specifically use it for datasets which must be either train/val/test
    """
    # decorator take functions as input
    def decorator(func):            
        # take argument names and argument defaults from the way the function is defined
        sig = inspect.signature(func)

        # temporarily we create a dummy copy of the original function
        @wraps(func)
        def wrapper(*args, **kwargs):
            # convert args and kwargs to dicitonary mapping: argument-name -> value
            bound = sig.bind(*args, **kwargs)
            # fills in default values for all unused arguments
            bound.apply_defaults()
            # check which value was fed in for argument_name (thus including defaults)
            value = bound.arguments.get(arg_name)

            if not isinstance(value, str):
                raise TypeError(f"Expected a string for {arg_name}, got {type(value).__name__}")
                    
            if value not in allowed_values:
                raise InvalidDataSetError(value)
            return func(*args, **kwargs)
        return wrapper
    return decorator

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