from typing import Optional, List
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
    Configuration for dynamic graph generation
    """
    method: str
    time_window: List[str]  # ISO format strings
    frequency: str
    self_connection: str
    scaling_method: Optional[str]
    kwargs: dict
    mode: str = 'dynamic'
    
    def copy(self):
        """Return a dict copy of the config"""
        return {
            'method': self.method,
            'time_window': self.time_window.copy(),
            'frequency': self.frequency,
            'self_connection': self.self_connection,
            'scaling_method': self.scaling_method,
            'kwargs': self.kwargs.copy(),
            'mode': self.mode
        }    