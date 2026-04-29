from typing import Dict, Any, Union, Tuple, Optional, TYPE_CHECKING
from tqdm import tqdm 
import numpy as np

from ...base.basemodel.statusmixin import ModelStatus
from ....dataloading.dataloaders import DeepDataLoaderManager, GraphDataLoaderManager
from ....utils.textformatting import section, align

if TYPE_CHECKING:
    from ..strategies.basestrategy import Strategy   
    from ...base.predictions import PredictionManager 

class DeepModelPresentationMixin:
    """ 
    Mixin class that deals with the model's (re)presentation.
    similar to PresentationMixin from BaseModel, but a bit more
    extensive.
    Contains dunder-method `str()` as well as
    hidden methods for status-update printing.
    """    
    name:               str 
    model_class:        str
    status_dict:        Dict[ModelStatus, bool]
    config_info:        Dict[str, Any]
    strategy:           'Strategy'
    predictions:        'PredictionManager'
    verbose:            int
    n_epochs:           int
    dataloadermanager:  Union[DeepDataLoaderManager, GraphDataLoaderManager]

    def _return_verbose_iter(self) -> Tuple[list, Union[range, tqdm[int]]]:
        # print dataloader snapshot
        if self.verbose >= 2:
            print('Dataloader:' + str(self.dataloadermanager.dataloader_train))        

        # determine verbose - loops (which loops to return evaluation metric)
        if self.verbose >= 2:
            verbose_loops   = list(np.arange(1, self.n_epochs + 1))
            epoch_iter      = range(self.n_epochs)

        elif self.verbose >= 1:
            verbose_loops   = list(np.arange(1, self.n_epochs + 1, step=10))
            epoch_iter      = range(self.n_epochs)

        elif self.verbose < 0:
            verbose_loops   = []
            epoch_iter      = range(self.n_epochs)

        else:
            verbose_loops   = []
            epoch_iter      = tqdm(range(self.n_epochs), desc="Training epochs") # if no verbose, just a tqdm     

        return verbose_loops, epoch_iter   

    def _return_verbose_line(self, 
                             epoch:         Optional[int]  = None, 
                             train_loss:    Optional[float]= None, 
                             val_loss:      Optional[float]= None, 
                             new_best:      Optional[str]  = None, 
                             patience:      Optional[str]  = None, 
                             lr_updated:    Optional[bool] = None):
        """prints a single line in the training - table"""
        columns = ["epoch", "train loss", "val loss", "new best", "patience"]
        columns = [col.upper() for col in columns]

        widths = [5, 10, 10, 8, 9]
        alignments = ["^", "^", "^", "^", "^"]

        def fmt(value, width, align):
            return f"{value:{align}{width}}"

        def make_row(values):
            return "| " + " | ".join(
                fmt(v, w, a) for v, w, a in zip(values, widths, alignments)
            ) + " |"

        # total table width = pipes + spaces + column widths
        total_width = sum(widths) + 3 * len(widths) + 1
        separator = "─" * total_width

        if any(x is not None for x in (epoch, train_loss, val_loss, new_best, patience, lr_updated)):
            row_values = [
                f"{epoch:03d}" if epoch is not None else "",
                f"{train_loss:.4f}" if train_loss is not None else "",
                f"{val_loss:.4f}" if val_loss is not None else "",
                f"{new_best}" if new_best is not None else "",
                f"{patience}" if patience else "",
            ]
            line = make_row(row_values)
            if lr_updated:
                line += " *"
            print(line)
        else:
            print(separator)
            print(make_row(columns))            

    def __str__(self):
        # Calculate width
        all_keys = (
            ['model name', 'model class'] +
            list(self.status_dict.keys()) +
            list(self.config_info.get('model_hparams', {}).keys()) +
            list(self.config_info.get('global_hparams', {}).keys())
        )
        width = max(len(k) for k in all_keys) if all_keys else 20
        
        # Build output
        lines = ['<DeepModel(']
        lines.append(align('model name', self.name, width))
        lines.append(align('model class', self.model_class, width))
        lines.append('')
        
        # Status section
        status_items = {k: "✓" if v else "✗" for k, v in self.status_dict.items()}
        lines.extend(section('status', status_items, width))
        lines.append('')
        
        # Forecasts section
        lines.extend(section('forecasts', {'forecasted': str(self.predictions)}, width))
        lines.append('')
        
        # Model hparams
        model_hparams = dict(self.config_info.get('model_hparams', {}))
        model_hparams['strategy'] = self.strategy
        lines.extend(section('model hparams', model_hparams, width))
        lines.append('')
        
        # Global hparams
        lines.extend(section('global hparams', self.config_info.get('global_hparams', {}), width))
        
        lines.append(')>')
        
        return '\n'.join(lines)
