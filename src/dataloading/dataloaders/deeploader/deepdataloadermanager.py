import torch
import pandas as pd
import numpy as np 
import os
from typing import List, Tuple, Optional, Union, Literal
from dataclasses import dataclass

from tqdm import tqdm

from ....graphconstruction.containers import GraphStructure, DynamicGraphStructure
from ...dataorchestration.dataorchestrator import DataOrchestrator

from ....utils.textformatting import error_emoji

class GraphStructureError(Exception):
    pass

class DeepData:
    """
    An alternative to the Pytorch Data class for dataentries of X, y

    Parameters
    ----------
    x: torch.Tensor
        input data of shape ...
    y: torch.Tensor
        target data of shape ...
    """
    def __init__(self, 
                 x:             torch.Tensor, 
                 y:             torch.Tensor):
        
        self.x          = x
        self.y          = y

    def to(self, device: torch.device) -> 'DeepData':
        """Move all tensors to the specified device (GPU)"""
        return DeepData(
            x=self.x.to(device),
            y=self.y.to(device),
        )        
    
    def __repr__(self):
        cls = self.__class__.__name__
        info = (
            f"x={tuple(self.x.shape)}, "
            f"y={tuple(self.y.shape)}"
        )
        return f"{cls}({info})"
    
class DeepDataLoader:
    """
    An alternative to the Pytorch DataLoader class
    Basically a list of GraphData entries
    """    
    def __init__(self, data_list: List[DeepData]):
        self.data_list = data_list

    def __iter__(self):
        return iter(self.data_list)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> DeepData:
        return self.data_list[idx]
    
    def __repr__(self):
        cls = self.__class__.__name__
        if not self.data_list:
            return f"{cls}(empty)"
        snapshot = self.data_list[0]
        info = (
            f"{len(self.data_list)} datapoints, "
            f"sample: x={tuple(snapshot.x.shape)}, "
            f"y={tuple(snapshot.y.shape)}"
        )
        return f"{cls}({info})"      
    
@dataclass
class DeepDataLoaderCollection:
    """ 
    """
    train:              'DeepDataLoader'
    val:                'DeepDataLoader'   
    test:               'DeepDataLoader'
    main:               'DeepDataLoader'

    def __repr__(self):
        return (f"<DeepDataLoaderCollection(train, val, test, main)>") 

class DeepDataLoaderManager:
    """
    """
    def __init__(self, 
                 dataorchestrator: DataOrchestrator):
        
        self.dataorchestrator       = dataorchestrator
        self.column_registration    = dataorchestrator.column_registration

    def construct_dataloaders(self):
        """
        creates the actual dataloaders
        """
        X,y,t                             = self._construct_Xy(self.dataorchestrator.data_final.data)
        main_dataloader                   = self._construct_main_dataloader(X = X, y = y, t= t)
        self.dataloader_collection        = self._split_dataloader(main_dataloader = main_dataloader)
        return self     
    
    def _construct_Xy(self, 
                      df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, List[Union[pd.Timestamp, str]]]:
        """
        constructs torch.tensor objects from df, which should be the dataorchestrator's final data object.
        
        Returns:
        -------
        X: torch.Tensor
            Input data of shape [num_timestamps, num_nodes, num_features]
        y: torch.Tensor
            Target data of shape [num_timestamps, num_nodes, horizon_size]          
        """

        dfc                 = df.copy()
        feature_arrays      = []
        target_arrays       = []
        t                   = dfc['timestamp'].unique().tolist()

        feature_cols = self.column_registration.get_by_type('feature')
        split_cols   = self.column_registration.get_by_type('split')
        target_cols  = self.column_registration.get_by_type('target')          

        time_splits         = dfc[['timestamp'] + split_cols].drop_duplicates().reset_index(drop = True)
        self.time_splits    = time_splits

        for feat in feature_cols:
            dtype = dfc[feat].dtype
            # Pivot from long to wide: rows=time, columns=nodes, values=feature
            pivoted = dfc.pivot(index=['timestamp'], columns='node', values=feat).reset_index(drop = True)

            # set missing nodes to zero
            # pivoted = pivoted.reindex(index=timestamps, columns=node_ids)

            # # convert to numeric
            pivoted = pivoted.apply(pd.to_numeric, errors='coerce')
            
            # # Convert to numpy float array, replace NaNs with 0
            arr = pivoted.values
            if str(dtype).startswith('int'):
                arr = arr.astype(np.int32)
            else:
                arr = arr.astype(np.float32)             # force float32 dtype

            feature_arrays.append(arr)

        X_np = np.stack(feature_arrays, axis=-1)

        for target in target_cols:
            for dfc_col in dfc.columns:
                
                if target in dfc_col and dfc_col not in feature_cols:
                    dtype = dfc[dfc_col].dtype
                    # Pivot from long to wide: rows=time, columns=nodes, values=feature
                    pivoted = dfc.pivot(index=['timestamp'], columns='node', values=dfc_col).reset_index(drop = True)

                    # set missing nodes to zero
                    # pivoted = pivoted.reindex(index=timestamps, columns=node_ids)

                    # # convert to numeric
                    pivoted = pivoted.apply(pd.to_numeric, errors='coerce')
                    
                    # # Convert to numpy float array, replace NaNs with 0
                    arr = pivoted.values
                    if str(dtype).startswith('int'):
                        arr = arr.astype(np.int32)
                    else:
                        arr = arr.astype(np.float32)   
                    target_arrays.append(arr)

        # Process target column using same approach
        y_np = np.stack(target_arrays, axis=-1)

        X = torch.tensor(X_np, dtype=torch.float)
        y = torch.tensor(y_np,dtype=torch.float)
        return X, y, t

    def _construct_main_dataloader(self, 
                                   X: torch.Tensor,
                                   y: torch.Tensor,
                                   t: List[str]) -> DeepDataLoader:
        """
        """


        dataset = []
        T       = X.shape[0]  # Total number of timesteps

        # Calculate maximum valid start position
        # Need: start + periods + prediction_horizon - 1 < T
        max_start       = T - self.dataorchestrator.config.horizon_leadtime - (self.dataorchestrator.config.horizon_size - 1) - (self.dataorchestrator.config.sequence_length - 1)
        self.max_start  = max_start

        if max_start <= 0:
            raise ValueError(f"Not enough data: T={T}, periods={self.dataorchestrator.config.sequence_length}"
                            f"Need at least {self.dataorchestrator.config.sequence_length} timesteps.")

        for t_idx, start in enumerate(range(max_start)):
            # Input window: periods consecutive timesteps
            x_seq = X[start : start + self.dataorchestrator.config.sequence_length]  # shape [periods, nodes, features]
            y_seq = y[start + self.dataorchestrator.config.sequence_length - 1]
            
            data = DeepData(
                x = x_seq.clone().detach().float().permute(1, 2, 0),  # (nodes, features, periods)
                y = y_seq.clone().detach().float(),                   # (nodes, horizon)
            )
            dataset.append(data)

        return DeepDataLoader(dataset)

    def _split_dataloader(self, 
                          main_dataloader: DeepDataLoader) -> DeepDataLoaderCollection:
        """
        Splits main dataloader into those for train/val/test
        """        

        train_idx = list(self.time_splits[self.time_splits['train']].index)
        val_idx   = list(self.time_splits[self.time_splits['val']].index)
        test_idx  = list(self.time_splits[self.time_splits['test']].index)

        dataloader_train = DeepDataLoader([main_dataloader[tt] for tt in train_idx])
        dataloader_val   = DeepDataLoader([main_dataloader[tt] for tt in val_idx])
        dataloader_test  = DeepDataLoader([main_dataloader[tt] for tt in test_idx if tt < self.max_start])        
        
        return DeepDataLoaderCollection(
            train = dataloader_train,
            val   = dataloader_val,
            test  = dataloader_test,
            main  = main_dataloader
        )

    def __repr__(self) -> str:

        representation = f'<DeepDataLoaderManager(dataloaders at .dataloader_collection)>'          
        return representation