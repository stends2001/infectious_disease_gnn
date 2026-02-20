from typing import TYPE_CHECKING, Optional, Literal
import pandas as pd 

from ...utils.textformatting import align, section
from ...dataloading import BaseLineDataLoaderManager 
from ...utils import check_dataset

from .baselinemodel import BaseLineModel 

class ClimateologyModel(BaseLineModel):

    def __init__(self, 
                 dataloadermanager: BaseLineDataLoaderManager,                 
                 name:              Optional[str] = None,
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        if not name:
            name = f'Climateology Model'

        super().__init__(dataloadermanager=dataloadermanager, name=name, verbose=verbose)
        self.seasonal_averages = self._get_temporal_averages(dataloadermanager.dataloader_main)

    def _add_seasonal_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds t_number column based on temporal frequency"""
        dfc  = df.copy()
        freq = self.dataloadermanager.dataorchestrator.config.temporal_frequency
        if freq == 'w':
            dfc['t_number'] = dfc[self.epiconfig.temporal_column].dt.isocalendar().week.astype(int)
        elif freq == 'd':
            dfc['t_number'] = dfc[self.epiconfig.temporal_column].dt.isocalendar().day.astype(int)
        elif freq == 'm':
            dfc['t_number'] = dfc[self.epiconfig.temporal_column].dt.month
        return dfc

    def _get_temporal_averages(self, dataloader_main: pd.DataFrame) -> pd.DataFrame:
        """Returns a df with the average target per (node, seasonal timepoint) over training data"""
        dl_main = self._add_seasonal_index(dataloader_main)
        return (dl_main.groupby([self.epiconfig.id_column, 't_number'])['target']
                       .mean()
                       .reset_index()
                       .rename(columns={'target': 'seasonal_mean'}))

    def train(self):
        """
        Compute per-seasonal-timepoint residual quantiles on training data.
        Uncertainty scales with the season rather than being globally flat.
        """
        quantiles = self.dataloadermanager.dataorchestrator.config.quantiles

        if quantiles:
            train_df = self.dataloadermanager.dataloader_main
            train_df = train_df[train_df['train']]
            train_df = self._add_seasonal_index(train_df)
            merged   = pd.merge(train_df, self.seasonal_averages, on=[self.epiconfig.id_column, 't_number'])

            residuals = merged['target'] - merged['seasonal_mean']

            # per seasonal timepoint quantiles
            self._residual_quantiles = (
                residuals.groupby(merged['t_number'])
                         .quantile(quantiles)
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
            evaluation_df = self._add_seasonal_index(evaluation_df)
            evaluation_df = pd.merge(evaluation_df, self.seasonal_averages, on=[self.epiconfig.id_column, 't_number'])

            if quantiles:
                for i, q in enumerate(quantiles):
                    offset = evaluation_df['t_number'].map(self._residual_quantiles[q])
                    evaluation_df[f'pred_q{i}'] = (evaluation_df['seasonal_mean'] + offset).clip(lower=0)
                pred_cols     = [f'pred_q{i}' for i in range(len(quantiles))]
                evaluation_df = evaluation_df.drop(columns=['seasonal_mean', 't_number'])
            else:
                evaluation_df = evaluation_df.rename(columns={'seasonal_mean': 'pred'}).drop(columns=['t_number'])
                pred_cols     = ['pred']

            evaluation_dataset = evaluation_df[[self.epiconfig.id_column, self.epiconfig.temporal_column, 'target'] + pred_cols]
            self.predictions.add_horizon_predictions(dataset, self._normalize(evaluation_dataset), hh)            
           
        self._update_status('forecasted')   
        return self  
  
    def __str__(self):
        all_keys = ['model name', 'model family', 'forecasted']
        width = max(len(k) for k in all_keys)
        
        lines = [f'<{self.model_class}(']
        lines.append(align('model name', self.name, width))
        lines.append(align('model family', 'BaseLineModel', width))        
        lines.append('')
        lines.extend(section('status', {'forecasted': self.predictions}, width))
        lines.append(')>')
        
        return '\n'.join(lines)