from typing import Optional, List, Literal
from dataclasses import dataclass

@dataclass
class StaticGraphConfig:
    """
    Configuration dataclass for static (single) graphs, storing:

    Paramters
    ---------
    method: str
        name of method used
    self_connection: Literal['mean','0','max'] 
        name of method used to create self-loops
    normalization_method: Literal['minmax','log','zscore','symmetric','rowwise']
        name of method used to normalize edge-weights
    kwargs: Optional[dict] = None
        dictionary of kwargs
    """
    method:             str
    self_connection:    Literal['mean','0','max'] 
    normalization:      Optional[Literal['minmax','log','zscore','symmetric','rowwise']] = None
    kwargs:             Optional[dict]= None

    def __post_init__(self):
        self.mode = 'static'

    def copy(self):
        """Return a dict copy of the config"""
        return {
            'method':           self.method,
            'self_connection':  self.self_connection,
            'normalization':    self.normalization,
            'kwargs':           self.kwargs.copy(),
        }        

@dataclass
class DynamicGraphConfig:
    """
    Configuration for dynamic graph generation
    """
    method:             str
    time_window:        List[str]  # ISO format strings
    frequency:          str
    self_connection:    str
    normalization:      Optional[str]
    kwargs:             dict

    def __post_init__(self):
        self.mode = 'dynamic'

    def copy(self):
        """Return a dict copy of the config"""
        return {
            'method':           self.method,
            'time_window':      self.time_window.copy(),
            'frequency':        self.frequency,
            'self_connection':  self.self_connection,
            'normalization':    self.normalization,
            'kwargs':           self.kwargs.copy(),
        }    