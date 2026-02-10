import torch
import pandas as pd
import numpy as np 
from typing import List, Tuple

from ...epidataorchestration import EpiDataOrchestrator


class DeepBaseDataLoaderManager:
    """
    Base class for creating dataloaders from EpiDataOrchestrator.
    Handles X/y construction and temporal splitting.
    """
    def __init__(self, dataorchestrator: EpiDataOrchestrator):
        self.dataorchestrator       = dataorchestrator
        self.column_registration    = dataorchestrator.column_registration
    
    def _split_Xyt(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs torch.tensor objects from long-format dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Long-format data with columns: timestamp, id_column, features, targets, splits
        
        Returns
        -------
        X : torch.Tensor
            Input data of shape [num_timestamps, num_nodes, num_features]
        y : torch.Tensor
            Target data of shape [num_timestamps, num_nodes, num_targets]

        Also creates

        time_splits : pd.DataFrame
            DataFrame with columns [timestamp, train, val, test] and t_idx as index
        """
        
        # Get column groups
        feature_cols    = self.column_registration.get_by_type('feature')
        split_cols      = self.column_registration.get_by_type('split')
        target_cols     = self.column_registration.get_by_type('target')
        
        temporal_col    = self.dataorchestrator.config.temporal_column
        id_col          = self.dataorchestrator.config.id_column
        
        # Store time splits for later use with t_idx as index
        time_splits = (
            df[[temporal_col] + split_cols]
            .drop_duplicates()
            .sort_values(temporal_col)
            .reset_index(drop=True)
        )
        time_splits.index.name = 't_idx'
        self.time_splits = time_splits
        
        # Helper function to pivot and convert columns
        def pivot_columns(columns: List[str]) -> np.ndarray:
            """Pivot columns from long to wide format and stack into 3D array"""
            arrays = []
            
            for col in columns:
                # Pivot: rows=time, cols=nodes, values=column
                wide = df.pivot(
                    index=temporal_col,
                    columns=id_col,
                    values=col
                )
                
                # Convert to numeric, then to numpy array
                arr = wide.apply(pd.to_numeric, errors='coerce').values
                
                # Preserve int types, otherwise use float32
                if df[col].dtype.kind == 'i':
                    arr = arr.astype(np.int32)
                else:
                    arr = arr.astype(np.float32)
                
                arrays.append(arr)
            
            # Stack: [time, nodes, features]
            return np.stack(arrays, axis=-1)
        
        # Build X from feature columns
        X_np = pivot_columns(feature_cols)
        
        # Build y from target columns (any column containing target names but not in features)
        target_column_names = [
            col for col in df.columns
            if any(target in col for target in target_cols)
            and col not in feature_cols
        ]
        
        y_np = pivot_columns(target_column_names)
        
        # Convert to tensors
        X = torch.tensor(X_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)
        
        return X, y
    
    def _build_sequences(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[List, List, List, List]:
        """
        Creates sequences from X, y tensors.
        Returns lists of data objects (not wrapped in container yet).
        """
        dataset_main = []
        dataset_train = []
        dataset_val = []
        dataset_test = []

        T = X.shape[0]
        min_end = self.dataorchestrator.config.sequence_length
        max_end = T + 1

        if min_end > T:
            raise ValueError(...)

        for x_end in range(min_end, max_end):
            x_start = x_end - self.dataorchestrator.config.sequence_length
            x_seq = X[x_start : x_end]
            y_seq = y[x_end - 1]
            target_idx = x_end - 1
            
            # Create data object (to be overridden by subclasses)
            data = self._create_data_object(x_seq, y_seq)
            
            # Split assignment
            if self.time_splits.loc[target_idx, 'train']:
                dataset_train.append(data)
            elif self.time_splits.loc[target_idx, 'val']:
                dataset_val.append(data)
            elif self.time_splits.loc[target_idx, 'test']:
                dataset_test.append(data)
            
            dataset_main.append(data)

        return dataset_main, dataset_train, dataset_val, dataset_test
    
    def _create_data_object(self, x_seq: torch.Tensor, y_seq: torch.Tensor):
        """Override in subclasses to create appropriate data objects"""
        raise NotImplementedError("Subclasses must implement _create_data_object")
    
    def build(self):
        """Main orchestration method - override in subclasses"""
        raise NotImplementedError("Subclasses must implement construct_dataloaders")
