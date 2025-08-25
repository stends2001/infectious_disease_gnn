import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import os
import torch
from torch_geometric.data import Data, DataLoader
import pandas as pd
import os
import numpy as np

from .epidataloader import EpiDataLoader

class GNNSequenceDataLoader:
    def __init__(self, epidataloader):
        self.feature_columns = epidataloader.feature_columns
        self.id_column = epidataloader.id_column
        self.temporal_column = epidataloader.temporal_column
        self.target_column = epidataloader.target_column

        self.train_df = epidataloader.XYt_train
        self.val_df = epidataloader.XYt_val
        self.test_df = epidataloader.XYt_test

        self.norm_params = epidataloader.norm_params['params']

    def retrieve_graph(self, graphname, graphdirectory='src/dataloading/graphs'):
        graphpath = os.path.join(graphdirectory, graphname)

        self.edge_index = torch.load(graphpath + '_edge_index.pt')

        try:
            self.edge_weight = torch.load(graphpath + '_edge_weight.pt')
        except FileNotFoundError:
            num_edges = self.edge_index.shape[1]
            self.edge_weight = torch.ones(num_edges, dtype=torch.float)

        return self

    def construct_dataloaders(self, periods, seq_len):
        self.periods = periods
        self.seq_len = seq_len

        X_train, y_train = self._separate_Xy(self.train_df)
        self.dataset_train = self._create_sequence_dataset(X_train, y_train, self.edge_index, self.edge_weight, periods, seq_len)

        X_val, y_val = self._separate_Xy(self.val_df)
        self.dataset_val = self._create_sequence_dataset(X_val, y_val, self.edge_index, self.edge_weight, periods, seq_len)

        X_test, y_test = self._separate_Xy(self.test_df)
        self.dataset_test = self._create_sequence_dataset(X_test, y_test, self.edge_index, self.edge_weight, periods, seq_len)

        return self

    def _create_temporal_dataset(self, X, y, edge_index, edge_weight, periods):
        dataset = []
        T = X.shape[0]
        for start in range(T - periods):
            x_seq = X[start: start + periods]  # [periods, num_nodes, features]
            y_target = y[start + periods]      # [num_nodes]

            data = Data(
                x=torch.tensor(x_seq, dtype=torch.float).permute(1, 2, 0),  # [num_nodes, features, periods]
                y=torch.tensor(y_target, dtype=torch.float),                # [num_nodes]
                edge_index=edge_index,
                edge_attr=edge_weight
            )
            dataset.append(data)
        return dataset

    def _create_sequence_dataset(self, X, y, edge_index, edge_weight, periods, seq_len):
        # First create temporal snapshots
        snapshots = self._create_temporal_dataset(X, y, edge_index, edge_weight, periods)
        dataset = []

        # Slide over snapshots to create sequences
        for start in range(len(snapshots) - seq_len):
            seq_snapshots = snapshots[start: start + seq_len]

            # Stack x: each x is [num_nodes, features, periods]
            x_seq = torch.stack([snap.x for snap in seq_snapshots], dim=0)  # [seq_len, num_nodes, features, periods]

            # Target: y of last snapshot in sequence
            y_target = seq_snapshots[-1].y  # [num_nodes]

            data = Data(
                x=x_seq,                # [seq_len, num_nodes, features, periods]
                y=y_target,
                edge_index=edge_index,
                edge_attr=edge_weight
            )
            dataset.append(data)

        return dataset

    def _separate_Xy(self, df):
        df[self.id_column] = df[self.id_column].astype(int)

        timestamps = sorted(df[self.temporal_column].unique())
        node_ids = sorted(df[self.id_column].unique())

        feature_arrays = []
        for feat in self.feature_columns:
            pivoted = df.pivot(index=self.temporal_column, columns=self.id_column, values=feat)
            pivoted = pivoted.reindex(index=timestamps, columns=node_ids)
            pivoted = pivoted.apply(pd.to_numeric, errors='coerce')

            arr = pivoted.values
            arr = np.where(pd.isna(arr), 0.0, arr)
            arr = arr.astype(np.float32)
            feature_arrays.append(arr)

        X_np = np.stack(feature_arrays, axis=-1)  # [T, N, F]

        y_pivoted = df.pivot(index=self.temporal_column, columns=self.id_column, values=self.target_column)
        y_pivoted = y_pivoted.reindex(index=timestamps, columns=node_ids)
        y_pivoted = y_pivoted.apply(pd.to_numeric, errors='coerce')

        y_arr = y_pivoted.values
        y_arr = np.where(pd.isna(y_arr), 0.0, y_arr)
        y_arr = y_arr.astype(np.float32)
        y_np = y_arr  # [T, N]

        X = torch.tensor(X_np, dtype=torch.float)
        y = torch.tensor(y_np, dtype=torch.float)

        return X, y
