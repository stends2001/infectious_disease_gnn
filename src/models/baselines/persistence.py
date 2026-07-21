from typing import Literal
import pandas as pd 
import numpy as np

from ..issues import ModelError
from ...dataloading.databuilders import BaseLineDataBuilder 
from .baselinemodel import BaseLineModel 

from ...utils import DataSetSplit

class PersistenceModel(BaseLineModel):
    """ 
    # TODO
    """
    def __init__(self, 
                 dataloadermanager: BaseLineDataBuilder,                 
                 name:              str = 'persistence_model',
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        super().__init__(dataloadermanager=dataloadermanager, name=name, verbose=verbose)
        self.status_dict.pop('model_hparams_set')
        self.status_dict.pop('global_hparams_set')

    def train(self):
        quantiles = self.dataloadermanager.dataorchestrator.config.quantiles

        if quantiles:
            self._residual_quantiles = self._compute_residual_quantiles('train')

        self._update_status('trained')

    def calibrate(self):
        """Refit residual quantiles on val data (conformal-style calibration)."""
        quantiles = self.dataloadermanager.dataorchestrator.config.quantiles

        if quantiles:
            self._residual_quantiles = self._compute_residual_quantiles('val')

    def _compute_residual_quantiles(self, split: str) -> pd.DataFrame:
        quantiles       = self.dataloadermanager.dataorchestrator.config.quantiles
        timeshift_num   = self.dataloadermanager.dataorchestrator.config.horizon_leadtime
        df              = self.dataloadermanager.dataloader_main
        df              = df[df[split]].sort_values([self.epiconfig.id_column, self.epiconfig.temporal_column]).copy()

        persistence_pred = df.groupby(self.epiconfig.id_column)['target'].shift(timeshift_num)
        residuals        = df['target'] - persistence_pred
        t_idx            = self._get_seasonal_index(df)

        return (
            residuals.groupby(t_idx)
                    .quantile(np.array(quantiles))
                    .unstack()
        )

    def forecast(self, dataset: DataSetSplit = 'test') -> None:
        """
        Forecast for set dataset
        """
        quantiles = self.dataloadermanager.dataorchestrator.config.quantiles

        for hh in range(self.dataloadermanager.dataorchestrator.config.horizon_size):
            timeshift_num   = int(hh + self.dataloadermanager.dataorchestrator.config.horizon_leadtime)
            evaluation_df   = self.dataloadermanager.dataloader_main
            evaluation_df   = evaluation_df.sort_values([self.epiconfig.id_column, self.epiconfig.temporal_column]).copy()

            persistence_pred = evaluation_df.groupby(self.epiconfig.id_column)['target'].shift(timeshift_num)

            if quantiles:
                # t_idx = self._get_seasonal_index(evaluation_df)
                for i, q in enumerate(quantiles):
                #     offset = t_idx.map(self._residual_quantiles[q])
                    evaluation_df[f'pred_q{i}'] = persistence_pred * (1 + (q - 0.5))
                pred_cols = [f'pred_q{i}' for i in range(len(quantiles))]
            else:
                evaluation_df['pred'] = persistence_pred
                pred_cols = ['pred']

            evaluation_df       = evaluation_df[evaluation_df[dataset]]
            evaluation_dataset  = evaluation_df[[self.epiconfig.id_column, self.epiconfig.temporal_column, 'target'] + pred_cols]

            self.predictions.add_horizon_predictions(dataset, self._normalize(evaluation_dataset), hh)

        self._update_status('forecasted')   
    
    def _get_seasonal_index(self, df: pd.DataFrame) -> pd.Series:
        """Returns seasonal index series based on temporal frequency"""
        freq = self.dataloadermanager.dataorchestrator.config.temporal_frequency
        if freq == 'w':
            return df[self.epiconfig.temporal_column].dt.isocalendar().week.astype(int)
        elif freq == 'd':
            return df[self.epiconfig.temporal_column].dt.isocalendar().day.astype(int)
        elif freq == 'm':
            return df[self.epiconfig.temporal_column].dt.month
        else:
            raise ModelError(f'Invalid temporal frequency found for ClimaScale model: {freq}')        