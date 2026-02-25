from typing import Optional, Literal
import pandas as pd 
import numpy as np

from ..base.issues import ModelError
from ...utils.textformatting import align, section
from ...dataloading import BaseLineDataLoaderManager 
from ...utils import check_dataset

from .baselinemodel import BaseLineModel 

class ClimateologyModel(BaseLineModel):
    """ 
    # TODO
    """
    def __init__(self, 
                 dataloadermanager: BaseLineDataLoaderManager,                 
                 name:              Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = f'Climateology Model'

        super().__init__(dataloadermanager=dataloadermanager, name=name, verbose=verbose)
        self.seasonal_averages = self._get_temporal_averages(dataloadermanager.dataloader_main)

    def train(self):
        """
        Compute per-seasonal-timepoint residual quantiles on training data.
        Uncertainty scales with the season rather than being globally flat.
        """
        quantiles = self.dataloadermanager.dataorchestrator.config.quantiles

        if quantiles:
            train_df = self.dataloadermanager.dataloader_main
            train_df = train_df[train_df['train']]
            train_df = self._get_seasonal_indexes(train_df)
            merged   = pd.merge(train_df, self.seasonal_averages, on=[self.epiconfig.id_column, 't_idx'])

            residuals= merged['target'] - merged['seasonal_mean']

            # per seasonal timepoint quantiles
            self._residual_quantiles = (
                residuals.groupby(merged['t_idx'])
                         .quantile(np.array(quantiles))
                         .unstack()
            )

        self._update_status('trained')

    @check_dataset()
    def forecast(self, dataset: Literal['train','val','test'] = 'test'):
        """
        Forecast for set dataset
        """
        quantiles = self.dataloadermanager.dataorchestrator.config.quantiles

        for hh in range(self.dataloadermanager.dataorchestrator.config.horizon_size):
            evaluation_df = self.dataloadermanager.dataloader_main[self.dataloadermanager.dataloader_main[dataset]]
            evaluation_df = evaluation_df[[self.epiconfig.id_column, self.epiconfig.temporal_column, 'target']]
            evaluation_df = self._get_seasonal_indexes(evaluation_df)
            evaluation_df = pd.merge(evaluation_df, self.seasonal_averages, on=[self.epiconfig.id_column, 't_idx'])

            if quantiles:
                for i, q in enumerate(quantiles):
                    offset = evaluation_df['t_idx'].map(self._residual_quantiles[q])
                    evaluation_df[f'pred_q{i}'] = (evaluation_df['seasonal_mean'] + offset).clip(lower=0)
                pred_cols     = [f'pred_q{i}' for i in range(len(quantiles))]
                evaluation_df = evaluation_df.drop(columns=['seasonal_mean', 't_idx'])
            else:
                evaluation_df = evaluation_df.rename(columns={'seasonal_mean': 'pred'}).drop(columns=['t_idx'])
                pred_cols     = ['pred']

            evaluation_dataset = evaluation_df[[self.epiconfig.id_column, self.epiconfig.temporal_column, 'target'] + pred_cols]
            self.predictions.add_horizon_predictions(dataset, self._normalize(evaluation_dataset), hh)            
           
        self._update_status('forecasted')   
        return self  
    
    def _get_seasonal_indexes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds t_idx column based on temporal frequency"""
        
        dfc         = df.copy()
        freq        = self.dataloadermanager.dataorchestrator.config.temporal_frequency

        timestamp: pd.Series[pd.Timestamp]  = dfc[self.epiconfig.temporal_column]   

        if freq == 'w':
            dfc['t_idx'] = timestamp.dt.isocalendar().week.astype(int)
        elif freq == 'd':
            dfc['t_idx'] = timestamp.dt.isocalendar().day.astype(int)
        elif freq == 'm':
            dfc['t_idx'] = timestamp.dt.month
        else:
            raise ModelError(f'Invalid temporal frequency found for ClimaScale model: {freq}')
        
        return dfc
    
    def _get_temporal_averages(self, dataloader_main: pd.DataFrame) -> pd.DataFrame:
        """Returns a df with the average target per (node, seasonal timepoint) over training data"""
        dl_main = self._get_seasonal_indexes(dataloader_main)
        return (dl_main.groupby([self.epiconfig.id_column, 't_idx'])['target']
                       .mean()
                       .reset_index()
                       .rename(columns={'target': 'seasonal_mean'}))
