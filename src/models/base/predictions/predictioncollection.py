import pandas as pd 
from dataclasses import dataclass, field

from ...issues import MissingPredictionsError

@dataclass
class PredictionCollection:
    """
    Stores predictions across horizons for one datast (train/val/test)
    Predictions are stored under three variables:
    - horizon:              int
    - is_original:          bool
    - spatially_aggregated: bool
    
    Accessible by:
    self.get_original(0)

    """
    _data: dict[tuple[int, bool, bool], pd.DataFrame] = field(default_factory=dict)       # a new dictionary is created for each class' instance
    
    def add(self, data: pd.DataFrame, horizon: int, is_original: bool = False, spatially_aggregated: bool = False):
        """
        Add predictions

        Parameters
        ----------
        data: pd.DataFrame

        horizon: int

        is_original: bool
            if False, then transformed scale, if True then nontransformed        
        """
        self._data[(horizon, is_original, spatially_aggregated)] = data
    
    def get(self, horizon: int, is_original: bool, spatially_aggregated: bool) -> pd.DataFrame:

        key = (horizon, is_original, spatially_aggregated)
        if key not in self._data:
            raise MissingPredictionsError(f"No predictions found for horizon={horizon}, is_original={is_original}, spatially_aggregated={spatially_aggregated}. Available: {list(self._data.keys())}")
        return self._data[key]

    @property
    def horizons(self) -> list[int]:
        """return a list of horizon integers for which predictions are found"""
        return sorted(set(h for h, _, _ in self._data.keys()))
    
    def _contains_data(self) -> bool:
        """return bool for whether or not predictions exist"""
        return bool(self.horizons)
    
    def __repr__(self) -> str:
        if self._contains_data():
            return f"<PredictionCollection(predictions for horizons {self.horizons})>"
        else:
            return f"<PredictionCollection(no predictions)>"