from typing import List, TYPE_CHECKING
import re

from ...utils.modelcolors import model_colors

if TYPE_CHECKING:
    from ....dataloading.columnregistration import ColumnRegistry
    from ....dataloading.epiconfig import EpiConfig

class ModelAppearanceMixin:
    """ 
    Mixin class that deals with the model's appearance.
    These methods are called by BaseModel's init function to set attributes.
    """
    name:                str
    model_class:         str
    epiconfig:          'EpiConfig'
    column_registration:'ColumnRegistry'

    def _get_model_color(self) -> str:
        """returns model-color in string format based on the lookup in model_colors"""
        lookup_name = self.model_class.lower()
        
        if lookup_name not in model_colors:
            raise ValueError(f'no color set for model of class {lookup_name}')
        else:
            return model_colors[lookup_name]
        
    def _get_clean_name(self) -> str:
        """Return a filesystem-safe version of the model name."""
        name = self.name.lower()
        
        # Replace whitespace with underscore
        name = re.sub(r"\s+", "_", name)
        
        # Remove all characters except a-z, 0-9, and underscore
        name = re.sub(r"[^a-z0-9_-]", "", name)
        
        # Collapse multiple underscores
        name = re.sub(r"_+", "_", name)
        
        # Strip leading/trailing underscores
        return name.strip("_")

    def _get_pred_cols(self) -> List[str]:
        """ 
        return the names of the prediction - columns
        """
        if self.epiconfig._num_quantiles == 0:
            pred_cols= ['pred']
        else:
            pred_cols= [c for c in self.column_registration.pred_columns if c != 'pred']
        return pred_cols    