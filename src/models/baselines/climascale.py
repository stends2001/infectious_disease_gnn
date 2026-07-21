from typing import Literal, Self
import pandas as pd 
import numpy as np

from .baselinemodel import BaseLineModel 
from ..issues import ModelError
from ...utils import DataSetSplit
from ...dataloading.databuilders import BaseLineDataBuilder 

class ClimaScaleModel(BaseLineModel):
    """ 
    # TODO
    """
    def __init__(self, 
                 dataloadermanager: BaseLineDataBuilder,                 
                 name:              str = 'climascale_model',
                 verbose:           Literal[-1, 0, 1, 2] = -1):
        
        super().__init__(dataloadermanager=dataloadermanager, name=name, verbose=verbose)
        self.status_dict.pop('model_hparams_set')
        self.status_dict.pop('global_hparams_set')        

    def train(self) -> None:
        """
        Compute per-seasonal-timepoint residual quantiles on training data.
        Uncertainty scales with the season rather than being globally flat.
        """
        quantiles = self.dataloadermanager.dataorchestrator.config.quantiles

        if quantiles:
            dl          = self.dataloadermanager.dataloader_main.copy()
            data_seen   = dl[dl['train']]
            train_df    = dl[dl['train']]

            seasonal_averages = self._get_seasonal_means(data_seen)
            train_indexed     = self._get_seasonal_indexes(train_df)
            merged            = pd.merge(train_indexed, seasonal_averages, on=[self.epiconfig.id_column, 't_idx']).sort_values(
                by=[self.epiconfig.temporal_column, self.epiconfig.id_column])
            
            merged            = merged.groupby(self.epiconfig.id_column).apply(
                lambda g: self._compute_pred(g, self.epiconfig.horizon_leadtime)
                ).reset_index(drop=True)

            residuals = merged['target'] - merged['pred']

            # per seasonal timepoint quantiles
            self._residual_quantiles = (
                residuals.groupby(merged['t_idx'])
                         .quantile(np.array(quantiles))
                         .unstack()
            )

        self._update_status('trained')

    def forecast(self, dataset: DataSetSplit = 'test') -> None:
        """
        Forecast for set dataset
        """
        quantiles = self.dataloadermanager.dataorchestrator.config.quantiles
        dl        = self.dataloadermanager.dataloader_main.copy()

        for hh in range(self.dataloadermanager.dataorchestrator.config.horizon_size):

            if dataset == 'train':
                data_seen = dl[dl['train']]
            elif dataset == 'val':
                data_seen = dl[dl['train']]
            elif dataset == 'test':
                data_seen = dl[dl['train'] | dl['val']]
        
            evaluation_df       = dl[dl[dataset]]
            seasonal_averages   = self._get_seasonal_means(data_seen)
            evaluation_indexed  = self._get_seasonal_indexes(evaluation_df)

            merged_df = pd.merge(evaluation_indexed, seasonal_averages, on=[self.epiconfig.id_column, 't_idx']).sort_values(
                            by=[self.epiconfig.temporal_column, self.epiconfig.id_column])
            merged_df = merged_df.groupby(self.epiconfig.id_column).apply(
                            lambda g: self._compute_pred(g, self.epiconfig.horizon_leadtime)).reset_index(drop=True)

            if quantiles:
                for i, q in enumerate(quantiles):
                    offset = merged_df['t_idx'].map(self._residual_quantiles[q])
                    merged_df[f'pred_q{i}'] = (merged_df['pred'] + offset).clip(lower=0)
                pred_cols = [f'pred_q{i}' for i in range(len(quantiles))]
                merged_df = merged_df.drop(columns=['pred', 't_idx', 'seasonal_mean'])
            else:
                pred_cols = ['pred']
                merged_df = merged_df.drop(columns=['t_idx', 'seasonal_mean'])

            evaluation_dataset = merged_df[[self.epiconfig.id_column, self.epiconfig.temporal_column, 'target'] + pred_cols]
            self.predictions.add_horizon_predictions(dataset, self._normalize(evaluation_dataset), hh)          

        self._update_status('forecasted')   

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

    def _get_seasonal_means(self, dataloader_main: pd.DataFrame) -> pd.DataFrame:
        """Returns a df with the average target per (node, seasonal timepoint)"""
        dl_main = self._get_seasonal_indexes(dataloader_main)
        return (dl_main.groupby([self.epiconfig.id_column, 't_idx'])['target']
                       .mean()
                       .reset_index()
                       .rename(columns={'target': 'seasonal_mean'}))
    
    def _compute_pred(self, group, h):
        """Compute scaled seasonal prediction per node group"""
        shifted_target   = group['target'].shift(h)
        shifted_seasonal = group['seasonal_mean'].shift(h)
        scaling_factor   = shifted_target / shifted_seasonal.replace(0, float('nan'))
        group['pred']    = (scaling_factor * group['seasonal_mean']).fillna(0)
        return group