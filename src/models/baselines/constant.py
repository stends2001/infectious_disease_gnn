from typing import Literal, Union, Optional
import pandas as pd
from ...utils.textformatting import section, align
from ..base import BaseModel, PredictionCollection
from ...dataloading import ShallowDataLoaderManager

class ConstantModel(BaseModel):

    """
    """

    def __init__(self, 
                 dataloadermanager: ShallowDataLoaderManager, 
                 name:              Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = f'ConstantModel'

        super().__init__(dataloadermanager, name, verbose)
        
        self.train_losses           = []
        self.val_losses             = []
        
    def train(self):
        print("This naive model doesn't train")

    def set_global_hparams(self):
        print("This naive model doesn't require global hparams")

    def set_model_hparams(self, constant_value: float):
        self.constant_value = constant_value

    def forecast(self, dataset: Literal['train','val','test'] = 'test'):
        """
        Forecast for set dataset
        """
        
        for hh in range(self.dataloadermanager.dataorchestrator.config.horizon_size):
            horizon_name            = f'horizon_{hh}'
            timeshift_num           = int(hh + self.dataloadermanager.dataorchestrator.config.horizon_leadtime)
            timeshift_str           = f"{timeshift_num}W"
            dataloader_collection   = self.dataloadermanager.dataloader_collections[horizon_name]

            if dataset == 'train':
                X, y = dataloader_collection.train.X, dataloader_collection.train.y

            elif dataset == 'val':
                X, y = dataloader_collection.val.X, dataloader_collection.val.y         

            elif dataset == 'test':
                X, y = dataloader_collection.test.X, dataloader_collection.test.y      
            else:
                raise ValueError('please provide a valid dataset: "train"/"val"/"test"')             

            
            # Then filter to get the evaluation dataset
            evaluation_df = dataloader_collection.main[dataloader_collection.main[dataset]].reset_index(drop=True)                          
            evaluation_df['pred'] = self.constant_value
            evaluation_df = evaluation_df[['node','timestamp','target','pred']]

            # Get the n largest timestamps
            if hh > 0:
                largest_timestamps = list(evaluation_df['timestamp'].unique())[-hh:]
            
                # Filter out the rows that have these largest timestamps
                evaluation_df = evaluation_df[~evaluation_df['timestamp'].isin(largest_timestamps)]   
                         
            self.predictions.add_horizon_predictions(dataset, evaluation_df, hh)
            
        self._update_status('forecasted')   
        return self  
        
    def __str__(self):
        # Calculate width
        all_keys = ['model name', 'model class', 'prediction column', 'forecasted']
        width = max(len(k) for k in all_keys)
        
        # Build output
        lines = [f'<ConstantModel({self.constant_value}']
        lines.append(align('model name', self.name, width))
        lines.append(align('model class', self.model_class, width))
        lines.append(align('prediction column', self.prediction_col, width))
        lines.append('')
        
        # Status section
        lines.extend(section('status', {'forecasted': self.predictions}, width))
        
        lines.append(')>')
        
        return '\n'.join(lines)