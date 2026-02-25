from typing import Literal, Self, Optional
import pandas as pd

from ...dataloading import BaseLineDataLoaderManager
from ...utils import check_dataset
from .baselinemodel import BaseLineModel 
from ...utils import align, section

class ConstantModel(BaseLineModel):
    """ 
    # TODO
    """
    def __init__(self, 
                 dataloadermanager: BaseLineDataLoaderManager, 
                 name:              Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = f'ConstantModel'

        super().__init__(dataloadermanager=dataloadermanager, name= name, verbose=verbose )

    def train(self):
        # compute residuals on training data
        train_df        = self.dataloadermanager.dataloader_main[self.dataloadermanager.dataloader_main['train']]
        residuals       = train_df['target'] - self.constant_value
        self._residuals = residuals
        self._update_status('trained')

    def set_model_hparams(self, constant_value: float):
        self.constant_value = constant_value
        self._update_status('model_hparams_set')
    
    @check_dataset()
    def forecast(self, dataset: Literal['train','val','test'] = 'test') -> Self:
        """
        Forecast for set dataset
        """
        self._check_state(['model_hparams_set','trained'])

        quantiles = self.dataloadermanager.dataorchestrator.config.quantiles
        
        for hh in range(self.dataloadermanager.dataorchestrator.config.horizon_size):
            evaluation_df               = self.dataloadermanager.dataloader_main[self.dataloadermanager.dataloader_main[dataset]]
            evaluation_df               = evaluation_df[[self.epiconfig.id_column,self.epiconfig.temporal_column,'target']]
            
            if quantiles:
                for i, q in enumerate(quantiles):
                    evaluation_df[f'pred_q{i}'] = self.constant_value + self._residuals.quantile(q)            
            else:
                evaluation_df['pred']       = self.constant_value
                        
            self.predictions.add_horizon_predictions(dataset, self._normalize(evaluation_df), hh)
            
        self._update_status('forecasted')   
        return self  
    
    def __str__(self) -> str:

        all_keys = (
            ['model name', 'model class'] + list(self._state.keys())
        )

        width = max(len(k) for k in all_keys) if all_keys else 20
        
        # Build output
        lines = [f'<{self.__class__.__name__}(']
        lines.append('')        
        general_items = {'name': self.name, 'model_class': self.model_class}
        lines.extend(section('generics', general_items, width))
        lines.append('')
        
        # Status section
        status_items = {k: "✓" if v else "✗" for k, v in self._state.items()}
        status_items['global_hparams_set'] = 'NA'
        lines.extend(section('status', status_items, width))
        lines.append('')
                
        # Global hparams
        lines.extend(section('global hparams', self.config_info.get('global_hparams', {}), width))
        lines.append(align('constant_value',  self.constant_value, width))
        
        lines.append(')>')
        
        return '\n'.join(lines)    