from typing import Union, List, TYPE_CHECKING, Literal, assert_never, Dict, Tuple
import pandas as pd
import torch 
from tqdm import tqdm
from torch import Tensor as Tensor
import numpy as np

from ....types import DataSetSplit
from ..issues import UnexpectedDataShape
from ...base.basemodel.statusmixin import ModelStatus
from ....dataloading.dataloaders import DeepDataLoaderManager, GraphDataLoaderManager

if TYPE_CHECKING:
    from ...base.predictions import PredictionManager
    from ....dataloading.epiconfig import EpiConfig
    from ..strategies.basestrategy import Strategy 
    from ...utils.loss.losshandler import LossHandler  
    from ....dataloading.epidataorchestration.containers import ContextEpiData

class DeepModelForecastMixin:
    """ 
    Mixin class that deals with the forecasting of DeepModels.
    NOTE we have two stubs here, defined in ModelStatusMixin.
    These stubs follow the actual functions' signatures but
    are not called. Simply here for typing.

    Main method is `forecast()` with its helper method `_format_forecast_results()`.
    """  
    model:              torch.nn.Module
    dataloadermanager:  Union[DeepDataLoaderManager, GraphDataLoaderManager]
    strategy:           'Strategy'
    verbose:            int
    epiconfig:          'EpiConfig'
    device:             torch.device
    loss:               'LossHandler'
    predictions:        'PredictionManager'
    context_data:       'ContextEpiData'
    _residual_quantiles: Dict[Tuple[int, int], Dict[int, float]]  

    def forecast(self, dataset: DataSetSplit = 'test'):
        """forecast the given dataset"""
        raw_predictions: List[Tensor]   = []
        raw_targets: List[Tensor]       = []

        # check the required states
        self._check_status(['model_hparams_set', 'global_hparams_set', 'trained'])

        # set model in evaluation mode
        self.model.eval()

        match dataset:
            case 'train':
                dataloader = self.dataloadermanager.dataloader_train
            case 'val':
                dataloader = self.dataloadermanager.dataloader_val 
            case 'test':
                dataloader = self.dataloadermanager.dataloader_test
            case _:
                assert_never(dataset)
        
        self.strategy.reset_state_dataset()

        # define iterator: whether or not to use tqdm
        iterator            = tqdm(dataloader, desc=f"Forecasting {dataset}") if self.verbose >= 0 else dataloader
        total_loss          = 0
        
        # setup expected predictions-shape [num_nodes, horizon_size, num_quantiles]
        num_nodes           = self.context_data.num_nodes
        _out_quantiles      = 1 if (hasattr(self, '_residual_quantiles') or self.epiconfig._num_quantiles == 0) else self.epiconfig._num_quantiles
        expected_shape_yhat = [num_nodes,
                            self.epiconfig.horizon_size,
                            max(1, _out_quantiles)
                            ]

        # turn off gradient tracking
        with torch.no_grad():

            # for each snapshot, forecast
            for idx, snapshot in enumerate(iterator):
                snapshot = snapshot.to(self.device)
                y_hat, loss_val = self.strategy.forecast_step(
                    model   = self.model, 
                    snapshot= snapshot, 
                    loss_fn = self.loss
                )
                total_loss += loss_val

                # validate predictions-shape only the first snapshot
                if idx == 0:
                    if list(y_hat.shape) != expected_shape_yhat:
                        raise UnexpectedDataShape(f'{list(y_hat.shape)}', f'{expected_shape_yhat}', "stacked yhat forecasting snapshot 0")

                raw_predictions.append(y_hat.detach().cpu())
                raw_targets.append(snapshot.y.detach().cpu())

        avg_loss = total_loss / len(dataloader)
        setattr(self, f'{dataset}_loss', avg_loss)

        # =========== SHAPE CHECK 1 ============= #
        # at this point, raw_predictions is a List of len [timestamps].
        # at each idx, there is a Tensor with shape [num_nodes, horizon_size, quantiles].
        # Since quantiles don't play a role in the target, those do not have that final dim
        # We're now removing the list-ness and stack that to a new dimension. The tensors therefore
        # get 3 (target) and 4 (predictions) dimensions.

        predictions_tensor  = torch.stack(raw_predictions)
        targets_tensor      = torch.stack(raw_targets)

        expected_shape_predictions = [len(dataloader), num_nodes, self.epiconfig.horizon_size, max(1, _out_quantiles)]
        expected_shape_targets      = [len(dataloader), num_nodes, self.epiconfig.horizon_size]        

        received_shape_predictions  = list(predictions_tensor.shape)
        received_shape_targets      = list(targets_tensor.shape)

        if expected_shape_predictions != received_shape_predictions:
            raise UnexpectedDataShape(f'{received_shape_predictions}', f'{expected_shape_predictions}', "stacked raw predictions")

        if expected_shape_targets != received_shape_targets:
            raise UnexpectedDataShape(f'{received_shape_targets}', f'{expected_shape_targets}', "stacked raw targets")

        num_timesteps, num_nodes, horizon_size, num_quantiles = predictions_tensor.shape

        # =========== CALIBRATION ============= #
        # If calibrate() has been called on a point-loss model, expand the
        # single pred dim into Q quantile columns using val residuals.
        if len(self._residual_quantiles) >0 and self.loss.loss_name not in ['pinball', 'pinchpinball']:
            predictions_tensor = self._apply_calibration(predictions_tensor, dataset)
            num_quantiles      = predictions_tensor.shape[-1]  # update: 1 → Q

        # Get the quantile column names from the column registry
        if self.epiconfig._num_quantiles == 0:
            pred_col_names = ['pred']                  
        else:
            pred_col_names = [c for c in self.predictions.column_registration.pred_columns if c != 'pred'] 

        # Get the quantile column names from the column registry
        # These are the names PredictionManager expects, e.g. ['q_0.1', 'q_0.5', 'q_0.9']
        # For num_quantiles == 0 (point forecast), this degenerates to ['pred']
        if self.epiconfig._num_quantiles == 0:
            pred_col_names = ['pred']                  
        else:
            pred_col_names = [c for c in self.predictions.column_registration.pred_columns if c != 'pred']    
        
        results = self._format_forecast_results(
            predictions     = predictions_tensor,
            targets         = targets_tensor,
            dataset         = dataset,
            num_timesteps   = num_timesteps,
            num_nodes       = num_nodes,
            horizon_size    = horizon_size,
            pred_col_names  = pred_col_names,
        )

        for hh in range(horizon_size):
            # Select the columns for this horizon: timestamp, id, all pred_cols, target
            horizon_cols = (
                [self.epiconfig.temporal_column, self.epiconfig.id_column]
                + [f'{col}_{hh}' for col in pred_col_names]
                + [f'target_{hh}']
            )
            horizon_data = results[horizon_cols].rename(
                columns={
                    **{f'{col}_{hh}': col for col in pred_col_names},
                    f'target_{hh}': 'target',
                }
            )

            self.predictions.add_horizon_predictions(dataset, horizon_data, hh)

        if self.verbose > 1:
            print(f"{dataset.capitalize()} loss: {avg_loss:.4f}")

        self._update_status('forecasted')

    def _format_forecast_results(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        dataset: Literal['train','val','test'],
        num_timesteps: int,
        num_nodes: int,
        horizon_size: int,
        pred_col_names: list[str],
        ) -> pd.DataFrame:
        """
        Formats predictions into a flat DataFrame aligned with correct timestamps.
        Handles both point forecasts (num_quantiles=1) and quantile forecasts.

        predictions shape: [num_timesteps, num_nodes, horizon_size, num_quantiles]
        targets shape:     [num_timesteps, num_nodes, horizon_size]
        """
        num_quantiles = len(pred_col_names)

        # Reshape: [num_sequences * num_nodes, horizon_size, num_quantiles]
        pred_reshaped = (
            predictions
            .view(num_timesteps * num_nodes, horizon_size, num_quantiles)
            .numpy()
        )
        # Reshape: [num_sequences * num_nodes, horizon_size]
        target_reshaped = targets.view(num_timesteps * num_nodes, horizon_size).numpy()

        # Index arrays — np.repeat/tile is correct here, no issue
        sequence_idx = np.repeat(np.arange(num_timesteps), num_nodes)
        node_idx     = np.tile(np.arange(num_nodes), num_timesteps)

        global_indices = self.dataloadermanager.time_splits[
            self.dataloadermanager.time_splits[dataset]
        ].index

        offset = (self.dataloadermanager.dataorchestrator.config.sequence_length - 1) if dataset == 'train' else 0

        timestamps = self.dataloadermanager.time_splits.loc[
            global_indices[sequence_idx + offset], self.epiconfig.temporal_column
        ].values
        
        results = pd.DataFrame({
            self.epiconfig.temporal_column: timestamps,
            self.epiconfig.id_column: node_idx,
        })

        # Sanity check: first and last timestamp should match expected range
        expected = self.predictions.temporal_summary.get_daterange_dataset(dataset, reference='t0')
        assert pd.Timestamp(timestamps[0]) == pd.Timestamp(expected[0]), \
            f"First timestamp mismatch: got {timestamps[0]}, expected {expected[0]}"
        assert pd.Timestamp(timestamps[-num_nodes]) == pd.Timestamp(expected[1]), \
            f"Last timestamp mismatch: got {timestamps[-num_nodes]}, expected {expected[1]}"

        # One column per horizon per quantile: e.g. q_0.1_0, q_0.5_0, q_0.9_0, ...
        for hh in range(horizon_size):
            for qq, col_name in enumerate(pred_col_names):
                results[f'{col_name}_{hh}'] = pred_reshaped[:, hh, qq]
            results[f'target_{hh}'] = target_reshaped[:, hh]

        return results

    def _apply_calibration(
        self,
        predictions_tensor: torch.Tensor,
        dataset: Literal['train', 'val', 'test'],
    ) -> torch.Tensor:
        """
        If calibration has been run, replace the single point-forecast quantile dim
        with Q calibrated quantile offsets applied to the median prediction.
        predictions_tensor: [num_timesteps, num_nodes, horizon_size, 1]
        returns:            [num_timesteps, num_nodes, horizon_size, Q]
        """
        import numpy as np

        quantiles   = self.epiconfig.quantiles
        time_splits = self.dataloadermanager.time_splits
        freq        = self.dataloadermanager.dataorchestrator.config.temporal_frequency

        timestamps  = pd.to_datetime(
            time_splits[time_splits[dataset]][self.epiconfig.temporal_column].values
        )

        if freq == 'w':
            t_idx = timestamps.isocalendar().week.astype(int).values
        elif freq == 'm':
            t_idx = timestamps.month.values
        elif freq == 'd':
            t_idx = timestamps.isocalendar().day.astype(int).values
        else:
            raise ValueError(f'Unknown temporal frequency: {freq}')

        num_timesteps, num_nodes, horizon_size, _ = predictions_tensor.shape
        point_preds = predictions_tensor.squeeze(-1).numpy()  # [T, N, H]
        q_preds     = np.zeros((num_timesteps, num_nodes, horizon_size, len(quantiles)))

        for hh in range(horizon_size):
            for q_idx in range(len(quantiles)):
                offsets = np.array([
                    self._residual_quantiles[(hh, q_idx)].get(t, 0.0)
                    for t in t_idx
                ])  # [T]
                # broadcast over nodes: [T, 1] + [T, N]
                q_preds[:, :, hh, q_idx] = point_preds[:, :, hh] + offsets[:, None]

        return torch.tensor(q_preds, dtype=torch.float32)


    # ========== STUBS ========== #
    def _check_status(self, required_states: Union[List[ModelStatus], ModelStatus]) -> None: ...
    def _update_status(self, status: ModelStatus) -> None: ...