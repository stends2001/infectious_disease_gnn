from typing import Literal, Union, Optional
import pandas as pd
from ...utils.textformatting import section, align
from ..base import BaseModel
from ...dataloading import ShallowDataLoaderManager

class ClimateologyModel(BaseModel):

    """
    Seasonal model predicts seasonal-persistence

    Training is therefore not necessary

    Examples:
    --------
    >>> seasonal_baseline = SeasonalModel(shallowdata)
    >>> seasonal_baseline.forecast('test')    
    >>> seasonal_baseline.show_forecast('test', 26) 
    """

    def __init__(self, 
                 dataloadermanager: ShallowDataLoaderManager, 
                 name:              Optional[str] = None):
        
        super().__init__(dataloadermanager, name)
        
        if not self.name:
            self.name = f'Climateology Model'
        
        self._state = {
            'model_initialized' : False,
            'trained'           : False,
            'forecasted'        : False,
        }
        
        self.train_losses           = []
        self.val_losses             = []


    def train(self):
        print("This naive model doesn't train")

    def forecast(self, dataset: Literal['train','val','test'] = 'test'):
        """
        Forecast for set dataset
        """
        
        for hh in range(self.dataloadermanager.dataorchestrator.config.horizon_size):
            horizon_name            = f'horizon_{hh}'
            timeshift               = f"{int(hh + self.dataloadermanager.dataorchestrator.config.horizon_leadtime)}W"
            dataloader_collection   = self.dataloadermanager.dataloader_collections[horizon_name]

            dataloader_main_all     = dataloader_collection.main[['timestamp','node','target','train','val','test']]
            

            if dataset == 'train':
                main_dataloader = dataloader_main_all[dataloader_main_all['train']]
            elif dataset == 'val':
                main_dataloader = dataloader_main_all[dataloader_main_all['train']]               
            elif dataset == 'test':
                main_dataloader = dataloader_main_all[(dataloader_main_all['train'] | dataloader_main_all['val'])]   
            else:
                raise ValueError(f'dataset must be "train", "val" or "test"')             

            weekly_averages = self._get_weekly_averages(main_dataloader).rename(columns = {'target':'pred'})

            evaluation_df               = dataloader_main_all[dataloader_main_all[dataset]]
            evaluation_df               = evaluation_df[['node','timestamp','target']]
            evaluation_df['week_number']= evaluation_df['timestamp'].dt.isocalendar().week
            evaluation_df               = pd.merge(evaluation_df, weekly_averages, on = ['node','week_number']).drop(columns = ['week_number'])
            self.predictions.add_horizon_predictions(dataset, evaluation_df, hh)
        self._state['forecasted'] = True
        return self

    def _get_weekly_averages(self, dataloader_main):
        dataloader_main['week_number'] = dataloader_main['timestamp'].dt.isocalendar().week
        return dataloader_main.groupby(['node','week_number'])['target'].mean().reset_index()
        
    def __str__(self):
        # Calculate width
        all_keys = ['model name', 'model class', 'prediction column', 'forecasted']
        width = max(len(k) for k in all_keys)
        
        # Build output
        lines = ['<PersistenceModel(']
        lines.append(align('model name', self.name, width))
        lines.append(align('model class', self.model_class, width))
        lines.append('')
        
        # Status section
        lines.extend(section('status', {'forecasted': self.predictions}, width))
        
        lines.append(')>')
        
        return '\n'.join(lines)