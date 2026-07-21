import pandas as pd 
from dataclasses import dataclass, field

from .exceptions import MissingPredictionsError

@dataclass
class PredictionCollection:
    """
    Stores predictions across horizons for a single datast (train/val/test)
    Predictions are stored under three variables:
    - horizon:              int
    - is_original:          bool
    - spatially_aggregated: bool
    
    All data is stored in attribute `_data` which is interacted with through
    methods
    - `add()`
    - `get()`
    """
    _data: dict[tuple[int, bool, bool], pd.DataFrame] = field(default_factory=dict)       # a new dictionary is created for each class' instance
    
    def add(self, 
            data:                   pd.DataFrame, 
            horizon:                int, 
            is_original:            bool, 
            spatially_aggregated:   bool):
        """
        Add predictions

        Parameters
        ----------
        data: pd.DataFrame
            the model's predictions for the specific combination of parameters.
            df has been validated in PredictionsManager
        horizon: int
            the index of the horizon (NOTE always starts at 0 independently of horizon_leadtime)
        is_original: bool
            if False, then transformed scale, if True then nontransformed        
        spatially_aggregated: bool
            whether the predictions are per node, or spatially aggregated (i.e. national).
        """
        self._data[(horizon, is_original, spatially_aggregated)] = data
    
    def get(self, 
            horizon:                int, 
            is_original:            bool, 
            spatially_aggregated:   bool) -> pd.DataFrame:
        """
        get predictions. The opposite of `.add()`

        Parameters
        ----------
        horizon: int
            the index of the horizon (NOTE always starts at 0 independently of horizon_leadtime)
        is_original: bool
            if False, then transformed scale, if True then nontransformed        
        spatially_aggregated: bool
            whether the predictions are per node, or spatially aggregated (i.e. national).
        """
        key = (horizon, is_original, spatially_aggregated)

        if key not in self._data:
            raise MissingPredictionsError(f"No predictions found for horizon={horizon}, is_original={is_original}, spatially_aggregated={spatially_aggregated}. Available: {list(self._data.keys())}")
        
        return self._data[key].copy()

    @property
    def horizons(self) -> list[int]:
        """return a list of horizon integers for which predictions are found"""
        return sorted(set(h for h, _, _ in self._data.keys()))
    
    def _contains_data(self) -> bool:
        """return bool for whether or not predictions exist"""
        return bool(self.horizons)
    
    def __repr__(self) -> str:
        if self._contains_data():
            internals =  f"predictions for horizons {self.horizons}"
        else:
            internals = "no predictions"
        
        return f"<{self.__class__.__name__}({internals})>"