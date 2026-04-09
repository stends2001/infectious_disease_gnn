from dataclasses import dataclass
from typing import Optional

@dataclass
class ZScoreParams:
    mean: float
    std:  float

@dataclass
class MinMaxParams:
    min: float
    max: float

@dataclass
class LogParams:
    shift: float = 1.0

@dataclass
class TransformationParams:
    """
    Holds all transformation parameters for a single column.
    Note that this currently supports only:
    - normalization methods:
        - ZScoreParams
        - MinMaxParams
    - transformation methods:
        - LogParams

    Properties
    ----------
    - `has_transformation()` -> whether or not a transformation is saved
    - `has_normalization()` -> whether or not a normalization is saved    
    """
    log:    Optional[LogParams]     = None
    zscore: Optional[ZScoreParams]  = None
    minmax: Optional[MinMaxParams]  = None

    @property
    def has_transformation(self) -> bool:
        return self.log is not None

    @property  
    def has_normalization(self) -> bool:
        return self.zscore is not None or self.minmax is not None