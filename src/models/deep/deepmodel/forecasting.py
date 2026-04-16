from typing import Union, List, TYPE_CHECKING, Literal
import pandas as pd
import torch 
from tqdm import tqdm
from torch import Tensor as Tensor
import numpy as np

from ....types import DataSetSplit
from ..issues import UnexpectedDataShape
from ...issues import ModelStatusError
from ...base.statusmixin import ModelStatus
from ....dataloading.dataloaders import DeepDataLoaderManager, GraphDataLoaderManager

if TYPE_CHECKING:
    from ...base.predictions_manager import PredictionManager
    from ....dataloading.epiconfig import EpiConfig
    from ..strategies.basestrategy import Strategy 
    from ...utils.loss.losshandler import LossHandler  

class DeepModelForecastMixin:
    """ 
    # TODO
    """  
    model:              torch.nn.Module
    dataloadermanager:  Union[DeepDataLoaderManager, GraphDataLoaderManager]
    strategy:           'Strategy'
    verbose:            int
    epiconfig:          'EpiConfig'
    device:             torch.device
    loss:               'LossHandler'
    predictions:        'PredictionManager'
    # ========== STUBS ========== #
    def _check_status(self, required_states: Union[List[ModelStatus], ModelStatus]) -> None: ...
    def _update_status(self, status: ModelStatus) -> None: ...

    def forecast(self, dataset: DataSetSplit = 'test'):
        
        self._check_status(['model_hparams_set', 'global_hparams_set', 'trained'])

        if self.model is None:
            raise ModelStatusError(f'attribute model is None. Model was not correctly initiated')   

        self.model.eval()

        # stupdly keep pylance happy
        if dataset == 'train':
            dataloader = self.dataloadermanager.dataloader_train
        elif dataset == 'val':
            dataloader = self.dataloadermanager.dataloader_val 
        elif dataset == 'test':
            dataloader = self.dataloadermanager.dataloader_test
        else:
            # will never be raised given the check_dataset decorator
            raise ValueError(f'cant find dataset {dataset}')

        self.strategy.reset_state_dataset()
        iterator = tqdm(dataloader, desc=f"Forecasting {dataset}") if self.verbose >= 0 else dataloader

        raw_predictions: List[Tensor]   = []
        raw_targets: List[Tensor]       = []
        
        total_loss = 0

        num_nodes = len(self.dataloadermanager.dataorchestrator.data_context.local_shapedata)

        expected_shape_yhat = [num_nodes, self.epiconfig.horizon_size, max(1,self.epiconfig._num_quantiles)]

        with torch.no_grad():
            for idx, snapshot in enumerate(iterator):
                snapshot = snapshot.to(self.device)
                y_hat, loss_val = self.strategy.forecast_step(
                    model=self.model, 
                    snapshot=snapshot, 
                    loss_fn=self.loss
                )
                total_loss += loss_val

                if not idx:
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

        expected_shape_predictions  = [len(dataloader), num_nodes, self.epiconfig.horizon_size, max(1,self.epiconfig._num_quantiles)]
        expected_shape_targets      = [len(dataloader), num_nodes, self.epiconfig.horizon_size]        

        received_shape_predictions  = list(predictions_tensor.shape)
        received_shape_targets      = list(targets_tensor.shape)

        if expected_shape_predictions != received_shape_predictions:
            raise UnexpectedDataShape(f'{received_shape_predictions}', f'{expected_shape_predictions}', "stacked raw predictions")

        if expected_shape_targets != received_shape_targets:
            raise UnexpectedDataShape(f'{received_shape_targets}', f'{expected_shape_targets}', "stacked raw targets")

        num_timesteps, num_nodes, horizon_size, num_quantiles = predictions_tensor.shape

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
        Format predictions into a flat DataFrame aligned with correct timestamps.
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
        