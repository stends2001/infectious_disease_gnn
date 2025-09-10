import torch
from torch_geometric.data import Data, DataLoader
import pandas as pd
import os
from typing import Dict, List, Literal, Any, Optional, Tuple
import numpy as np
from .epidataloader import EpiDataLoader


class GNNDataLoader:
    """
    creates an instance with attributes
    .dataset_train
    .dataset_val
    .dataset_test

    which have an X, y, edge_index and edge_weight (the latter only when applicable!)

    X has shape: [N, F, periods] (nodes/features/periods)
    """

    def __init__(self,
                 epidataloader: EpiDataLoader):
        
        # copy metadata from EpiDataLoader
        self.feature_columns = epidataloader.feature_columns
        self.id_column       = epidataloader.id_column
        self.temporal_column = epidataloader.temporal_column
        self.target_column   = epidataloader.target_column

        self.train_df        = epidataloader.XYt_train
        self.val_df          = epidataloader.XYt_val
        self.test_df         = epidataloader.XYt_test           
        self.XYt_train       = epidataloader.XYt_train
        self.XYt_val         = epidataloader.XYt_val
        self.XYt_test        = epidataloader.XYt_test  

        self.norm_params     = epidataloader.norm_params['params']
        self.edge_index      = None
        self.edge_weight     = None

    def construct_dataloaders(self, periods: int, prediction_horizon: int = 1):
        """
        Creates the actual dataloaders

        Parameters:
        -----------
        periods : int
            Length of the temporal window (lookback period). Number of consecutive 
            timesteps used as input to predict the next timestep. For example:
            - periods=4 uses weeks t-1, t-2, t-3, t-4 to predict week t
            - periods=8 uses 8 weeks of history to predict the next week
            Higher values capture longer temporal dependencies but reduce training samples.

        prediction_horizon : int
            Number of timesteps ahead to predict. For example:
            - prediction_horizon=1 predicts next week (t+1)

        Attributes set:
        ---------------
        dataset_train & dataset_val & dataset_test : List[torch_Geometric.data.Data]
            each of which has the followign attributes
            - x             => [periods, node, feature]
            - y             => [node]
            - edge_index    => [2, edge_number]
            - edge_weight   => [edge_number]
        """

        self.periods = periods

        # Separate X,y
        X_train, y_train   = self._separate_Xy(self.train_df)
        self.dataset_train = self._create_temporal_dataset(X_train, y_train, periods = periods)

        # For train set
        X_val, y_val = self._separate_Xy(self.val_df)
        self.dataset_val = self._create_temporal_dataset(X_val, y_val, periods = periods)

        # For train set
        X_test, y_test = self._separate_Xy(self.test_df)
        self.dataset_test = self._create_temporal_dataset(X_test, y_test, periods = periods)   
        return self     

    def _create_temporal_dataset(self,X: torch.tensor, y: torch.tensor, periods: int, prediction_horizon: int = 1) -> List[Data]:

        """
        Creates a dataloader based on X, y and periods
        """
        dataset = []
        T = X.shape[0]  # Total number of timesteps
        
        # Calculate maximum valid start position
        # Need: start + periods + prediction_horizon - 1 < T
        max_start = T - periods - prediction_horizon + 1
        
        if max_start <= 0:
            raise ValueError(f"Not enough data: T={T}, periods={periods}, prediction_horizon={prediction_horizon}. "
                            f"Need at least {periods + prediction_horizon} timesteps.")
        
        for start in range(max_start):
            # Input window: periods consecutive timesteps
            x_seq = X[start : start + periods]  # shape [periods, nodes, features]
            
            # Target: prediction_horizon steps after the input window
            target_timestep = start + periods + prediction_horizon - 1
            y_target = y[target_timestep]  # shape [nodes]
            
            data = Data(
                x = x_seq.clone().detach().float().permute(1, 2, 0),  # (nodes, features, periods)
                y = y_target.clone().detach().float(),                # (nodes,)
                edge_index = self.edge_index,
                edge_weight = self.edge_weight
            )
            dataset.append(data)
            
        return dataset   

    def retrieve_graph(self, graphname: str, graphdirectory:str ='data/graphs'):
        """
        Loads graphstructure into dataloaders

        sets the following attributes in the class
        - edge_index 
        - edge_weight
        """
    
        graphpath  = os.path.join(graphdirectory, graphname)
        
        try:
            edge_index = torch.load(graphpath + '_edge_index.pt', weights_only = False)

        except Exception as e:
            raise RuntimeError(f'graph by the name of {graphname} not found')


        # Try loading edge_weight, fallback to ones if file not found
        try:
            edge_weight = torch.load(graphpath + '_edge_weight.pt', weights_only = False)

        except FileNotFoundError:
            num_edges   = edge_index.shape[1]
            edge_weight = torch.ones(num_edges, dtype=torch.float)

        self.edge_index  = edge_index
        self.edge_weight = edge_weight

        return self
    
    def _separate_Xy(self,df: pd.DataFrame) -> Tuple[torch.tensor, torch.tensor]:
        """
        separates X and y from an XYt dataframe
        """
        df[self.id_column] = df[self.id_column].astype(int)       

        timestamps = sorted(df[self.temporal_column].unique())
        node_ids   = sorted(df[self.id_column].unique())

        feature_arrays = []
        for feat in self.feature_columns:
            # Pivot from long to wide: rows=time, columns=nodes, values=feature
            pivoted = df.pivot(index=self.temporal_column, columns=self.id_column, values=feat)

            # set missing nodes to zero
            pivoted = pivoted.reindex(index=timestamps, columns=node_ids)

            # convert to numeric
            pivoted = pivoted.apply(pd.to_numeric, errors='coerce')
            
            # Convert to numpy float array, replace NaNs with 0
            arr = pivoted.values
            arr = np.where(pd.isna(arr), 0.0, arr)  # replace NaN with 0.0
            arr = arr.astype(np.float32)             # force float32 dtype
            feature_arrays.append(arr)

        # Stack feature arrays along new dimension: (time, nodes, features)
        X_np = np.stack(feature_arrays, axis=-1)

        # Process target column using same approach
        y_pivoted = df.pivot(index=self.temporal_column, columns=self.id_column, values=self.target_column)
        y_pivoted = y_pivoted.reindex(index=timestamps, columns=node_ids)
        y_pivoted = y_pivoted.apply(pd.to_numeric, errors='coerce')
        y_arr     = y_pivoted.values
        y_arr     = np.where(pd.isna(y_arr), 0.0, y_arr)
        y_arr     = y_arr.astype(np.float32)
        y_np      = y_arr[..., None]

        X = torch.tensor(X_np, dtype=torch.float)
        y = torch.tensor(y_np, dtype=torch.float)

        # ensure y shape is [time, node]
        if y.dim() > 2 and y.size(2) == 1:
            y = y.squeeze(2)

        return X, y