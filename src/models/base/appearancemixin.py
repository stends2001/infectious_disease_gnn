from typing import List, TYPE_CHECKING

from ..utils.modelcolors import model_colors

if TYPE_CHECKING:
    from ...dataloading.columnregistration import ColumnRegistry
    from ...dataloading.epiconfig import EpiConfig

class ModelAppearanceMixin:

    name:                str
    model_class:         str
    epiconfig:          'EpiConfig'
    column_registration:'ColumnRegistry'

    def _get_model_color(self) -> str:
        lookup_name = self.model_class.lower()
        
        if lookup_name not in model_colors:
            raise ValueError(f'no color set for model of class {lookup_name}')
        else:
            return model_colors[lookup_name]
        
    def _get_clean_name(self) -> str:
        return self.name.lower().replace(' ', '_')

    def _get_pred_cols(self) -> List[str]:
        """ 
        return the names of the prediction - columns
        """
        if self.epiconfig._num_quantiles == 0:
            pred_cols= ['pred']
        else:
            pred_cols= [c for c in self.column_registration.pred_columns if c != 'pred']
        return pred_cols    