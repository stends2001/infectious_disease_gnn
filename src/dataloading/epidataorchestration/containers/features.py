import pandas as pd 
from dataclasses import dataclass

from ....utils.textformatting import checkmark


@dataclass
class FeatureEpiData:
    """
    Datacontainer for feature-engineered-data, and shifted target

    Parameters:
    ----------
    epidata: pd.DataFrame
    """        
    data:               pd.DataFrame

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(data {checkmark}")
        representation += ")>"
        return representation  