import torch
import pandas as pd
import numpy as np 
import os
from typing import List, Tuple, Optional, Union, Literal
from dataclasses import dataclass

from tqdm import tqdm

from ...epidataorchestration import EpiDataOrchestrator

from ....graphconstruction.containers import GraphStructure, DynamicGraphStructure
from ....utils.textformatting import error_emoji

class GraphStructureError(Exception):
    pass

class DeepData:
    """
    An alternative to the Pytorch Data class for dataentries of X, y

    Parameters
    ----------
    x: torch.Tensor
        input data of shape [num_nodes, sequence_length, num_features]
    y: torch.Tensor
        target data of shape [num_nodes, horizon_size]
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
                 dataorchestrator: EpiDataOrchestrator):
        
        self.dataorchestrator       = dataorchestrator
        self.column_registration    = dataorchestrator.column_registration

    def construct_dataloaders(self):
        """
        creates the actual dataloaders
        """
        X,y                               = self._construct_Xy(self.dataorchestrator.data_final.data)
        main_dataloader                   = self._construct_main_dataloader(X = X, y = y)
        self.dataloader_collection        = self._split_dataloader(main_dataloader = main_dataloader)
        return self     
    
    def _construct_Xy(self, 
                      df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
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
        t                   = dfc[self.dataorchestrator.config.temporal_column].unique().tolist()

        feature_cols = self.column_registration.get_by_type('feature')
        split_cols   = self.column_registration.get_by_type('split')
        target_cols  = self.column_registration.get_by_type('target')          

        time_splits         = dfc[[self.dataorchestrator.config.temporal_column] + split_cols].drop_duplicates().reset_index(drop = True)
        self.time_splits    = time_splits


        # create feature_arrays:
        # a list of np.arrays. Each array is of shape [num_timestamp, num_nodes]
        # which are merged together into a list of an array per feature.
        for feat in feature_cols:
            dtype = dfc[feat].dtype
            # Pivot from long to wide: rows=time, columns=nodes, values=feature
            pivoted = dfc.pivot(index=[self.dataorchestrator.config.temporal_column], columns=self.dataorchestrator.config.id_column, values=feat).reset_index(drop = True)

            # convert to numeric
            pivoted = pivoted.apply(pd.to_numeric, errors='coerce')

            # Convert to numpy float array, replace NaNs with 0
            arr = pivoted.values
            if str(dtype).startswith('int'):
                arr = arr.astype(np.int32)
            else:
                arr = arr.astype(np.float32)             # force float32 dtype

            feature_arrays.append(arr)

        
        # create target_arrays using the same approach:
        # a list of np.arrays. Each array is of shape [num_timestamp, num_nodes]
        # which are merged together into a list of an array per target column (per prediction horizon).
        for target in target_cols:
            for dfc_col in dfc.columns:
                
                if target in dfc_col and dfc_col not in feature_cols:
                    dtype = dfc[dfc_col].dtype
                    # Pivot from long to wide: rows=time, columns=nodes, values=feature
                    pivoted = dfc.pivot(index=[self.dataorchestrator.config.temporal_column], columns=self.dataorchestrator.config.id_column, values=dfc_col).reset_index(drop = True)

                    # convert to numeric
                    pivoted = pivoted.apply(pd.to_numeric, errors='coerce')
                    
                    # Convert to numpy float array, replace NaNs with 0
                    arr = pivoted.values
                    if str(dtype).startswith('int'):
                        arr = arr.astype(np.int32)
                    else:
                        arr = arr.astype(np.float32)   
                    target_arrays.append(arr)
        
        # now reshape the lists of arrays into 3d arrays
        X_np = np.stack(feature_arrays, axis=-1) # [num_timestamps, num_nodes, num_features]
        y_np = np.stack(target_arrays,  axis=-1) # [num_timestamps, num_nodes, horizon_size]

        # convert 3d arrays to tensors
        X_tensor = torch.tensor(X_np, dtype=torch.float) 
        y_tensor = torch.tensor(y_np, dtype=torch.float)
        return X_tensor, y_tensor
      
    def _construct_main_dataloader(self, 
                                   X: torch.Tensor,
                                   y: torch.Tensor) -> DeepDataLoader:
        """

        Parameters:
        ----------
        X: torch.Tensor
            Input data of shape [num_timestamps, num_nodes, num_features]
        y: torch.Tensor
            Target data of shape [num_timestamps, num_nodes, horizon_size]    
        t: List[pd.Timestamp]      
            List of timestamps associacted with each datapoint of length [num_timestamps]
        """

        dataset = []
        num_timesteps, num_nodes, num_features       = X.shape 
        _, _, num_targets                            = y.shape

        # Calculate maximum valid start position
        # Need: start + periods + prediction_horizon - 1 < num_timesteps
        max_start       = num_timesteps - self.dataorchestrator.config.horizon_leadtime - (self.dataorchestrator.config.horizon_size - 1) - (self.dataorchestrator.config.sequence_length - 1)
        self.max_start  = max_start

        if max_start <= 0:
            raise ValueError(
                f"Not enough data for windowing: T={num_timesteps}, "
                f"sequence_length={self.dataorchestrator.config.sequence_length}, "
                f"horizon_leadtime={self.dataorchestrator.config.horizon_leadtime}, "
                f"horizon_size={self.dataorchestrator.config.horizon_size}. "
                f"Need at least {self.dataorchestrator.config.sequence_length + self.dataorchestrator.config.horizon_leadtime + self.dataorchestrator.config.horizon_size} timesteps."
            )

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


# import torch
# import pandas as pd
# import numpy as np 
# import os
# from typing import List, Tuple, Optional, Union, Literal
# from dataclasses import dataclass

# from tqdm import tqdm

# from ....graphconstruction.containers import GraphStructure, DynamicGraphStructure
# from ...dataorchestration.dataorchestrator import DataOrchestrator

# from ....utils.textformatting import error_emoji

# class GraphStructureError(Exception):
#     pass

# class DeepData:
#     """
#     An alternative to the Pytorch Data class for dataentries of X, y

#     Parameters
#     ----------
#     x: torch.Tensor
#         input data of shape ...
#     y: torch.Tensor
#         target data of shape ...
#     """
#     def __init__(self, 
#                  x:             torch.Tensor, 
#                  y:             torch.Tensor):
        
#         self.x          = x
#         self.y          = y

#     def to(self, device: torch.device) -> 'DeepData':
#         """Move all tensors to the specified device (GPU)"""
#         return DeepData(
#             x=self.x.to(device),
#             y=self.y.to(device),
#         )        
    
#     def __repr__(self):
#         cls = self.__class__.__name__
#         info = (
#             f"x={tuple(self.x.shape)}, "
#             f"y={tuple(self.y.shape)}"
#         )
#         return f"{cls}({info})"
    
# class DeepDataLoader:
#     """
#     An alternative to the Pytorch DataLoader class
#     Basically a list of GraphData entries
#     """    
#     def __init__(self, data_list: List[DeepData]):
#         self.data_list = data_list

#     def __iter__(self):
#         return iter(self.data_list)
    
#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx: int) -> DeepData:
#         return self.data_list[idx]
    
#     def __repr__(self):
#         cls = self.__class__.__name__
#         if not self.data_list:
#             return f"{cls}(empty)"
#         snapshot = self.data_list[0]
#         info = (
#             f"{len(self.data_list)} datapoints, "
#             f"sample: x={tuple(snapshot.x.shape)}, "
#             f"y={tuple(snapshot.y.shape)}"
#         )
#         return f"{cls}({info})"      
  
   
# @dataclass
# class DeepDataLoaderCollection:
#     """ 
#     """
#     train:              'DeepDataLoader'
#     val:                'DeepDataLoader'   
#     test:               'DeepDataLoader'
#     main:               'DeepDataLoader'

#     def __repr__(self):
#         return (f"<DeepDataLoaderCollection(train, val, test, main)>") 

# class DeepDataLoaderManager:
#     """
#     """
#     def __init__(self, 
#                  dataorchestrator: DataOrchestrator):
        
#         self.dataorchestrator       = dataorchestrator
#         self.column_registration    = dataorchestrator.column_registration

#     def construct_dataloaders(self):
#         """
#         creates the actual dataloaders
#         """
#         X,y,t                             = self._construct_Xy(self.dataorchestrator.data_final.data)
#         main_dataloader                   = self._construct_main_dataloader(X = X, y = y, t= t)
#         self.dataloader_collection        = self._split_dataloader(main_dataloader = main_dataloader)
#         return self     
    
#     def _construct_Xy(self, 
#                       df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, List[pd.Timestamp]]:
#         """
#         constructs torch.tensor objects from df, which should be the dataorchestrator's final data object.
        
#         Returns:
#         -------
#         X: torch.Tensor
#             Input data of shape [num_timestamps, num_nodes, num_features]
#         y: torch.Tensor
#             Target data of shape [num_timestamps, num_nodes, horizon_size]    
#         t: List[pd.Timestamp]      
#             List of timestamps associacted with each datapoint of length [num_timestamps]
#         """

#         dfc                 = df.copy()
#         feature_arrays      = []
#         target_arrays       = []
#         t                   = dfc[self.dataorchestrator.config.temporal_column].unique().tolist()

#         feature_cols = self.column_registration.get_by_type('feature')
#         split_cols   = self.column_registration.get_by_type('split')
#         target_cols  = self.column_registration.get_by_type('target')          

#         time_splits         = dfc[[self.dataorchestrator.config.temporal_column] + split_cols].drop_duplicates().reset_index(drop = True)
#         self.time_splits    = time_splits


#         # create feature_arrays:
#         # a list of np.arrays. Each array is of shape [num_timestamp, num_nodes]
#         # which are merged together into a list of an array per feature.
#         for feat in feature_cols:
#             dtype = dfc[feat].dtype
#             # Pivot from long to wide: rows=time, columns=nodes, values=feature
#             pivoted = dfc.pivot(index=[self.dataorchestrator.config.temporal_column], columns=self.dataorchestrator.config.id_column, values=feat).reset_index(drop = True)

#             # convert to numeric
#             pivoted = pivoted.apply(pd.to_numeric, errors='coerce')

#             # Convert to numpy float array, replace NaNs with 0
#             arr = pivoted.values
#             if str(dtype).startswith('int'):
#                 arr = arr.astype(np.int32)
#             else:
#                 arr = arr.astype(np.float32)             # force float32 dtype

#             feature_arrays.append(arr)

        
#         # create target_arrays using the same approach:
#         # a list of np.arrays. Each array is of shape [num_timestamp, num_nodes]
#         # which are merged together into a list of an array per target column (per prediction horizon).
#         for target in target_cols:
#             for dfc_col in dfc.columns:
                
#                 if target in dfc_col and dfc_col not in feature_cols:
#                     dtype = dfc[dfc_col].dtype
#                     # Pivot from long to wide: rows=time, columns=nodes, values=feature
#                     pivoted = dfc.pivot(index=[self.dataorchestrator.config.temporal_column], columns=self.dataorchestrator.config.id_column, values=dfc_col).reset_index(drop = True)

#                     # convert to numeric
#                     pivoted = pivoted.apply(pd.to_numeric, errors='coerce')
                    
#                     # Convert to numpy float array, replace NaNs with 0
#                     arr = pivoted.values
#                     if str(dtype).startswith('int'):
#                         arr = arr.astype(np.int32)
#                     else:
#                         arr = arr.astype(np.float32)   
#                     target_arrays.append(arr)
        
#         # now reshape the lists of arrays into 3d arrays
#         X_np = np.stack(feature_arrays, axis=-1) # [num_timestamps, num_nodes, num_features]
#         y_np = np.stack(target_arrays,  axis=-1) # [num_timestamps, num_nodes, horizon_size]

#         # convert 3d arrays to tensors
#         X_tensor = torch.tensor(X_np, dtype=torch.float) 
#         y_tensor = torch.tensor(y_np, dtype=torch.float)
#         return X_tensor, y_tensor, t
      
#     def _construct_main_dataloader(self, 
#                                    X: torch.Tensor,
#                                    y: torch.Tensor,
#                                    t: List[str]) -> DeepDataLoader:
#         """

#         Parameters:
#         ----------
#         X: torch.Tensor
#             Input data of shape [num_timestamps, num_nodes, num_features]
#         y: torch.Tensor
#             Target data of shape [num_timestamps, num_nodes, horizon_size]    
#         t: List[pd.Timestamp]      
#             List of timestamps associacted with each datapoint of length [num_timestamps]
#         """

#         dataset = []
#         num_timesteps, num_nodes, num_features       = X.shape 
#         _, _, num_targets                            = y.shape

#         # Calculate maximum valid start position
#         # Need: start + periods + prediction_horizon - 1 < num_timesteps
#         max_start       = num_timesteps - self.dataorchestrator.config.horizon_leadtime - (self.dataorchestrator.config.horizon_size - 1) - (self.dataorchestrator.config.sequence_length - 1)
#         self.max_start  = max_start

#         if max_start <= 0:
#             raise ValueError(
#                 f"Not enough data for windowing: T={num_timesteps}, "
#                 f"sequence_length={self.dataorchestrator.config.sequence_length}, "
#                 f"horizon_leadtime={self.dataorchestrator.config.horizon_leadtime}, "
#                 f"horizon_size={self.dataorchestrator.config.horizon_size}. "
#                 f"Need at least {self.dataorchestrator.config.sequence_length + self.dataorchestrator.config.horizon_leadtime + self.dataorchestrator.config.horizon_size} timesteps."
#             )

#         for t_idx, start in enumerate(range(max_start)):
#             # Input window: periods consecutive timesteps
#             x_seq = X[start : start + self.dataorchestrator.config.sequence_length]  # shape [periods, nodes, features]
#             y_seq = y[start + self.dataorchestrator.config.sequence_length - 1]
            
#             data = DeepData(
#                 x = x_seq.clone().detach().float().permute(1, 2, 0),  # (nodes, features, periods)
#                 y = y_seq.clone().detach().float(),                   # (nodes, horizon)
#             )
#             dataset.append(data)       

#         return DeepDataLoader(dataset)

#     def _split_dataloader(self, 
#                           main_dataloader: DeepDataLoader) -> DeepDataLoaderCollection:
#         """
#         Splits main dataloader into those for train/val/test
#         """        

#         train_idx = list(self.time_splits[self.time_splits['train']].index)
#         val_idx   = list(self.time_splits[self.time_splits['val']].index)
#         test_idx  = list(self.time_splits[self.time_splits['test']].index)

#         dataloader_train = DeepDataLoader([main_dataloader[tt] for tt in train_idx])
#         dataloader_val   = DeepDataLoader([main_dataloader[tt] for tt in val_idx])
#         dataloader_test  = DeepDataLoader([main_dataloader[tt] for tt in test_idx if tt < self.max_start])        
        
#         return DeepDataLoaderCollection(
#             train = dataloader_train,
#             val   = dataloader_val,
#             test  = dataloader_test,
#             main  = main_dataloader
#         )

#     def __repr__(self) -> str:

#         representation = f'<DeepDataLoaderManager(dataloaders at .dataloader_collection)>'          
#         return representation