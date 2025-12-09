from typing import Optional
from dataclasses import dataclass

@dataclass
class StaticGraphConfig:
    """
    """
    method:         str
    self_connection:str
    scaling_method: Optional[str] = None
    kwargs:         Optional[dict]= None

@dataclass
class DynamicGraphConfig:
    """ 
    """
    method:         str
    self_connection:str
    scaling_method: Optional[str] = None
    temporal_window     = None
    temporal_frequency = None
    kwargs:         Optional[dict]= None


