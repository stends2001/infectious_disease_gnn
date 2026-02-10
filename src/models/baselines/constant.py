from typing import Literal, Union, Optional
import pandas as pd
from ...utils.textformatting import section, align
from ..base import BaseModel, PredictionCollection
from ...dataloading import BaseLineDataLoaderManager
from ...utils import check_dataset
from .baselinemodel import BaseLineModel 

class ConstantModel(BaseLineModel):

    """
    """

    def __init__(self, 
                 dataloadermanager: BaseLineDataLoaderManager, 
                 name:              Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = f'ConstantModel'

        super().__init__(dataloadermanager=dataloadermanager, name= name, verbose=verbose )
        
        self.train_losses           = []
        self.val_losses             = []
        
    def train(self):
        print("This naive model doesn't train")

    def set_global_hparams(self):
        print("This naive model doesn't require global hparams")

    def set_model_hparams(self, constant_value: float):
        self.constant_value = constant_value
        self._update_status('model_hparams_set')
    
    @check_dataset()
    def forecast(self, dataset: Literal['train','val','test'] = 'test'):
        """
        Forecast for set dataset
        """

        self._check_state(['model_hparams_set'])
        
        for hh in range(self.dataloadermanager.dataorchestrator.config.horizon_size):
            evaluation_df               = self.dataloadermanager.dataloader_collections[self.dataloadermanager.dataloader_collections[dataset]]
            evaluation_df               = evaluation_df[[self.epiconfig.id_column,self.epiconfig.temporal_column,'target']]
            evaluation_df['pred']       = self.constant_value
                        
            self.predictions.add_horizon_predictions(dataset, self._normalize(evaluation_df), hh)
            
        self._update_status('forecasted')   
        return self  
        
    def __str__(self):
        # Calculate width
        all_keys = ['model name', 'constant value' 'forecasted']
        width = max(len(k) for k in all_keys)
        
        # Build output
        lines = [f'<{self.model_class}(']
        lines.append(align('model name', self.name, width))
        lines.append(align('constant value', self.constant_value, width))        
        lines.append('')
        
        # Status section
        lines.extend(section('status', {'forecasted': self.predictions}, width))
        
        lines.append(')>')
        
        return '\n'.join(lines)