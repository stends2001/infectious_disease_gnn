from typing import TYPE_CHECKING, Optional, Literal
import pandas as pd 

from ...utils.textformatting import align, section
from ...dataloading import BaseLineDataLoaderManager 

from .baselinemodel import BaseLineModel 

class ClimateologyModel(BaseLineModel):

    def __init__(self, 
                 dataloadermanager: BaseLineDataLoaderManager,                 
                 name:              Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = f'Climateology Mode'

        super().__init__(dataloadermanager=dataloadermanager, name= name, verbose=verbose)
        self.seasonal_averages = self._get_temporal_averages(dataloadermanager.dataloader_collections)

    def _get_temporal_averages(self, dataloader_main: pd.DataFrame) -> pd.DataFrame:
        """returns a pd with the averages per timepoint over the entire dataset"""
        dl_main = dataloader_main.copy()
        if self.dataloadermanager.dataorchestrator.config.temporal_frequency == 'w':
            dl_main['t_number'] = dl_main['timestamp'].dt.isocalendar().week
        elif self.dataloadermanager.dataorchestrator.config.temporal_frequency == 'd':
            dl_main['t_number'] = dl_main['timestamp'].dt.isocalendar().day            
        return dl_main.groupby(['node','t_number'])['target'].mean().reset_index().rename(columns = {'target':'pred'})

    def forecast(self, dataset: Literal['train','val','test'] = 'test'):
        """
        Forecast for set dataset
        """
        
        for hh in range(self.dataloadermanager.dataorchestrator.config.horizon_size):
            horizon_name            = f'horizon_{hh}'

            evaluation_df               = self.dataloadermanager.dataloader_collections[self.dataloadermanager.dataloader_collections[dataset]]
            evaluation_df               = evaluation_df[['node','timestamp','target']]
            
            if self.dataloadermanager.dataorchestrator.config.temporal_frequency == 'w':
                evaluation_df['t_number']= evaluation_df['timestamp'].dt.isocalendar().week
            elif self.dataloadermanager.dataorchestrator.config.temporal_frequency == 'd':   
                evaluation_df['t_number']= evaluation_df['timestamp'].dt.isocalendar().day    

            evaluation_dataset               = pd.merge(evaluation_df, self.seasonal_averages, on = ['node','t_number']).drop(columns = ['t_number'])
            df_normalized                   = self._normalize(evaluation_dataset)    
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