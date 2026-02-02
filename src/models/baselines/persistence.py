from typing import TYPE_CHECKING, Optional, Literal
import pandas as pd 

from ...utils.textformatting import align, section
from ...dataloading import BaseLineDataLoaderManager 
from .baselinemodel import BaseLineModel 



class PersistenceModel(BaseLineModel):

    def __init__(self, 
                 dataloadermanager: BaseLineDataLoaderManager,                 
                 name:              Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = f'Persistence Model'

        super().__init__(dataloadermanager=dataloadermanager, name= name, verbose=verbose)

    def forecast(self, dataset: Literal['train','val','test'] = 'test'):
        """
        Forecast for set dataset
        """
        
        for hh in range(self.dataloadermanager.dataorchestrator.config.horizon_size):
            horizon_name            = f'horizon_{hh}'
            timeshift_num           = int(hh + self.dataloadermanager.dataorchestrator.config.horizon_leadtime)
            evaluation_df           = self.dataloadermanager.dataloader_collections.copy()
            evaluation_df['pred']   = evaluation_df.groupby(self.dataloadermanager.dataorchestrator.config.id_column)['target'].shift(timeshift_num).reset_index(drop = True)
           
            evaluation_dataset      = evaluation_df[evaluation_df[dataset]]

            df_normalized = self._normalize(evaluation_dataset)                               
            self.predictions.add_horizon_predictions(dataset, df_normalized, hh)

        self._update_status('forecasted')   
        return self  
  
    def __str__(self):
        # Calculate width
        all_keys = ['model name', 'model family', 'forecasted']
        width = max(len(k) for k in all_keys)
        
        # Build output
        lines = [f'<{self.model_class}(']
        lines.append(align('model name', self.name, width))
        lines.append(align('model family', 'BaseLineModel', width))
        lines.append('')
        
        # Status section
        lines.extend(section('status', {'forecasted': self.predictions}, width))
        
        lines.append(')>')
        
        return '\n'.join(lines)        