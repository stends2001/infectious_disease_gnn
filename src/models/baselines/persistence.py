from typing import TYPE_CHECKING, Optional, Literal
import pandas as pd 

from ...utils.textformatting import align, section
from ...dataloading import BaseLineDataLoaderManager 
from .baselinemodel import BaseLineModel 

from ...utils import check_dataset

class PersistenceModel(BaseLineModel):

    def __init__(self, 
                 dataloadermanager: BaseLineDataLoaderManager,                 
                 name:              Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = f'Persistence Model'

        super().__init__(dataloadermanager=dataloadermanager, name= name, verbose=verbose)

    @check_dataset()
    def forecast(self, dataset: Literal['train','val','test'] = 'test'):
        """
        Forecast for set dataset
        """
        
        for hh in range(self.dataloadermanager.dataorchestrator.config.horizon_size):
            horizon_name            = f'horizon_{hh}'
            timeshift_num           = int(hh + self.dataloadermanager.dataorchestrator.config.horizon_leadtime)
            evaluation_df = self.dataloadermanager.dataloader_collections
            # evaluation_df = evaluation_df[[self.epiconfig.id_column,'timestamp','target']]
            evaluation_df = evaluation_df.sort_values([self.epiconfig.id_column, self.epiconfig.temporal_column])

            timeshift_num           = int(hh + self.dataloadermanager.dataorchestrator.config.horizon_leadtime)
            evaluation_df['pred']   = evaluation_df.groupby(self.dataloadermanager.dataorchestrator.config.id_column)['target'].shift(timeshift_num)
            evaluation_df           = evaluation_df[evaluation_df[dataset]]

            # Now evaluation_df is already filtered
            evaluation_dataset = evaluation_df[[self.epiconfig.id_column, self.epiconfig.temporal_column,'target','pred']]

                          
            self.predictions.add_horizon_predictions(dataset, self._normalize(evaluation_dataset), hh)

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