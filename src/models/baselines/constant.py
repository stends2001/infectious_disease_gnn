from typing import Literal, Self

from ...dataloading.dataloaders import BaseLineDataLoaderManager
from ...utils import check_dataset
from .baselinemodel import BaseLineModel 

class ConstantModel(BaseLineModel):
    """ 
    # TODO
    """
    def __init__(self, 
                 dataloadermanager: BaseLineDataLoaderManager, 
                 name:              str = "constant_model",
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        super().__init__(dataloadermanager=dataloadermanager, name= name, verbose=verbose )
        self.status_dict.pop('model_hparams_set')
        
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
        self._check_status(['model_hparams_set','trained'])

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