from typing import TYPE_CHECKING, Optional, Literal
import pandas as pd 

from .baselinemodel import BaseLineModel 

from ...utils.textformatting import align, section
from ...dataloading import BaseLineDataLoaderManager 

class ClimaScaleModel(BaseLineModel):

    def __init__(self, 
                 dataloadermanager: BaseLineDataLoaderManager,                 
                 name:              Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = f'ClimaScale Model'

        super().__init__(dataloadermanager=dataloadermanager, name= name, verbose=verbose)

        self.horizon_leadtime = self.dataloadermanager.dataorchestrator.config.horizon_leadtime
               
    def forecast(self, dataset: Literal['train','val','test'] = 'test'):
        """
        Forecast for set dataset
        """
        dl: pd.DataFrame = self.dataloadermanager.dataloader_collections.copy()

        for hh in range(self.dataloadermanager.dataorchestrator.config.horizon_size):
            horizon_name            = f'horizon_{hh}'

            if dataset == 'train':
                data_seen = dl[dl['train']]
            elif dataset == 'val':
                data_seen = dl[dl['train']]
            elif dataset == 'test':
                data_seen = dl[dl['train'] | dl['val']]   
            else:
                raise ValueError(f'invalid dataset {dataset}')
        
            evaluation_df   = dl[dl[dataset]]

            seasonal_averages = self._get_seasonal_means(data_seen)
            dataloader_main   = self._get_seasonal_indexes(evaluation_df)

            merged_df   = pd.merge(dataloader_main, seasonal_averages, on = ['node','t_idx']).sort_values(by = ['timestamp','node'])
            
            evaluation_dataset = merged_df.groupby('node').apply(lambda g: self.compute_pred(g, self.horizon_leadtime))
            evaluation_dataset = evaluation_dataset.reset_index(drop = True)
            df_normalized                   = self._normalize(evaluation_dataset[['node','timestamp','target','pred']])    
            self.predictions.add_horizon_predictions(dataset, df_normalized, hh)          

        self._update_status('forecasted')   
        return self  
  
    def _get_seasonal_means(self, dataloader_main: pd.DataFrame) -> pd.DataFrame:
        """returns a pd with the averages per timepoint over the entire dataset"""
        dl_main         = self._get_seasonal_indexes(dataloader_main)
        seasonal_means  = dl_main.groupby(['node','t_idx'])['target'].mean().reset_index().rename(columns = {'target':'seasonal_mean'})
        return seasonal_means
    
    def _get_seasonal_indexes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        returns the same df but with a new column t_ind;
        relating timestamp to seasonal time index
        """
        dfc = df.copy()
        temporal_frequency = self.dataloadermanager.dataorchestrator.config.temporal_frequency

        if temporal_frequency== 'w':
            dfc['t_idx'] = dfc['timestamp'].dt.isocalendar().week
        
        elif temporal_frequency == 'd':
            dfc['t_idx'] = dfc['timestamp'].dt.isocalendar().day          
        
        elif temporal_frequency == "m":
           dfc['t_idx'] = dfc["timestamp"].dt.month          

        return dfc
    
    def compute_pred(self, group, h):
        # shift target and seasonal_mean by h within each node
        shifted_target  = group['target'].shift(h)
        shifted_seasonal = group['seasonal_mean'].shift(h)
        
        # avoid division by zero
        scaling_factor = shifted_target / shifted_seasonal.replace(0, float('nan'))
        
        group['pred'] = scaling_factor * group['seasonal_mean']
        
        group['pred'] = group['pred'].fillna(0)
        return group   

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