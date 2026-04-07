import pandas as pd 
from dataclasses import dataclass

from ....utils.textformatting import checkmark

@dataclass
class FinalizedEpiData:
    """
    Datacontainer for finalized data, both normalized and denormalized

    Parameters:
    ----------
    epidata: pd.DataFrame
    """     
    data:        pd.DataFrame
    data_denorm: pd.DataFrame

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(data {checkmark},"
                f"data_denorm {checkmark}"
                )
        representation += ")>"
        return representation   