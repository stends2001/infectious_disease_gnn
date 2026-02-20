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

        super().__init__(dataloadermanager=dataloadermanager, name=name, verbose=verbose)

    def train(self):
        """
        Compute per-seasonal-timepoint residual quantiles on training data.
        Uncertainty scales with the season rather than being globally flat.
        """
        quantiles = self.dataloadermanager.dataorchestrator.config.quantiles

        if quantiles:
            train_df        = self.dataloadermanager.dataloader_collections
            train_df        = train_df[train_df['train']].sort_values([self.epiconfig.id_column, self.epiconfig.temporal_column]).copy()
            timeshift_num   = self.dataloadermanager.dataorchestrator.config.horizon_leadtime
            persistence_pred= train_df.groupby(self.epiconfig.id_column)['target'].shift(timeshift_num)

            residuals       = train_df['target'] - persistence_pred
            t_idx           = self._get_seasonal_index(train_df)

            # per seasonal timepoint quantiles
            self._residual_quantiles = (
                residuals.groupby(t_idx)
                         .quantile(quantiles)
                         .unstack()
            )

        self._update_status('trained')

    def _get_seasonal_index(self, df: pd.DataFrame) -> pd.Series:
        """Returns seasonal index series based on temporal frequency"""
        freq = self.dataloadermanager.dataorchestrator.config.temporal_frequency
        if freq == 'w':
            return df[self.epiconfig.temporal_column].dt.isocalendar().week.astype(int)
        elif freq == 'd':
            return df[self.epiconfig.temporal_column].dt.isocalendar().day.astype(int)
        elif freq == 'm':
            return df[self.epiconfig.temporal_column].dt.month

    @check_dataset()
    def forecast(self, dataset: Literal['train','val','test'] = 'test'):
        """
        Forecast for set dataset
        """
        quantiles = self.dataloadermanager.dataorchestrator.config.quantiles

        for hh in range(self.dataloadermanager.dataorchestrator.config.horizon_size):
            timeshift_num   = int(hh + self.dataloadermanager.dataorchestrator.config.horizon_leadtime)
            evaluation_df   = self.dataloadermanager.dataloader_collections
            evaluation_df   = evaluation_df.sort_values([self.epiconfig.id_column, self.epiconfig.temporal_column]).copy()

            persistence_pred = evaluation_df.groupby(self.epiconfig.id_column)['target'].shift(timeshift_num)

            if quantiles:
                t_idx = self._get_seasonal_index(evaluation_df)
                for i, q in enumerate(quantiles):
                    offset = t_idx.map(self._residual_quantiles[q])
                    evaluation_df[f'pred_q{i}'] = (persistence_pred + offset).clip(lower=0)
                pred_cols = [f'pred_q{i}' for i in range(len(quantiles))]
            else:
                evaluation_df['pred'] = persistence_pred
                pred_cols = ['pred']

            evaluation_df       = evaluation_df[evaluation_df[dataset]]
            evaluation_dataset  = evaluation_df[[self.epiconfig.id_column, self.epiconfig.temporal_column, 'target'] + pred_cols]

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