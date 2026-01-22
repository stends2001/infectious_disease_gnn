from typing import Literal, Union, Optional
import pandas as pd
from ...utils.textformatting import section, align
from ..base import BaseModel
from ...dataloading import ShallowDataLoaderManager

class ClimateologyModel(BaseModel):

    """
    Seasonal baseline model

    Predictions are made according to:
    $$\hat{y}_{i}^{(t+h)} = \frac{1}{|\mathcal{T}_{\text{train}}(t_N)|} \sum_{t' \in \mathcal{T}_{\text{train}}(t_N)} y_i^{(t')}$$

    where $\mathcal{T}_{\text{train}}(w)$ denotes all time points in the training set corresponding to position in the seasonal cycle $t_N$.

    Examples
    --------
    >>> seasonal_baseline = ClimateologyModel(shallowdata)
    >>> seasonal_baseline.forecast('test')    
    >>> seasonal_baseline.show_forecast('test', 26) 

    Note
    ----
    training is not necessary
    """

    def __init__(self, 
                 dataloadermanager: ShallowDataLoaderManager, 
                 name:              Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1
                 ):

        if not name:
            name = f'Climateology Model'         

        super().__init__(dataloadermanager=dataloadermanager, name= name, model_color="#707070", verbose=verbose )
        
        self.train_losses           = []
        self.val_losses             = []

    def train(self):
        print("This naive model doesn't train")

    def set_global_hparams(self):
        print("This naive model doesn't require global hparams")

    def set_model_hparams(self):
        print("This naive model doesn't require model hparams")        

    def forecast(self, dataset: Literal['train','val','test'] = 'test'):
        """
        Forecast for set dataset
        """
        
        for hh in range(self.dataloadermanager.dataorchestrator.config.horizon_size):
            horizon_name            = f'horizon_{hh}'
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

            weekly_averages = self._get_temporal_averages(main_dataloader).rename(columns = {'target':'pred'})

            evaluation_df               = dataloader_main_all[dataloader_main_all[dataset]]
            evaluation_df               = evaluation_df[['node','timestamp','target']]
            
            if self.dataloadermanager.dataorchestrator.config.temporal_frequency == 'w':
                evaluation_df['t_number']= evaluation_df['timestamp'].dt.isocalendar().week
            elif self.dataloadermanager.dataorchestrator.config.temporal_frequency == 'd':   
                evaluation_df['t_number']= evaluation_df['timestamp'].dt.isocalendar().day    

            evaluation_df               = pd.merge(evaluation_df, weekly_averages, on = ['node','t_number']).drop(columns = ['t_number'])
            self.predictions.add_horizon_predictions(dataset, evaluation_df, hh)
        self._update_status('forecasted')
        return self

    def _get_temporal_averages(self, dataloader_main: pd.DataFrame) -> pd.DataFrame:
        dl_main = dataloader_main.copy()
        if self.dataloadermanager.dataorchestrator.config.temporal_frequency == 'w':
            dl_main['t_number'] = dl_main['timestamp'].dt.isocalendar().week
        elif self.dataloadermanager.dataorchestrator.config.temporal_frequency == 'd':
            dl_main['t_number'] = dl_main['timestamp'].dt.isocalendar().day            
        return dl_main.groupby(['node','t_number'])['target'].mean().reset_index()
        
    def __str__(self):
        # Calculate width
        all_keys = ['model name', 'forecasted']
        width = max(len(k) for k in all_keys)
        
        # Build output
        lines = [f'<{self.model_class}(']
        lines.append(align('model name', self.name, width))
        lines.append('')
        
        # Status section
        lines.extend(section('status', {'forecasted': self.predictions}, width))
        
        lines.append(')>')
        
        return '\n'.join(lines)