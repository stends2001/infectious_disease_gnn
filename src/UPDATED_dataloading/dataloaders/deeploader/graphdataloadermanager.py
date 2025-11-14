from typing import List
import torch
import pandas as pd
import pandas as pd
from typing import Literal, Union, List, Dict
from dataclasses import dataclass

from ...dataorchestration.dataorchestrator import DataOrchestrator

class GraphData:
    """
    An alternative to the Pytorch Data class
    """
    def __init__(self, 
                 x:             torch.Tensor, 
                 y:             torch.Tensor,
                 edge_index:    torch.Tensor,
                 edge_weight:   torch.Tensor):
        
        self.x          = x
        self.y          = y
        self.edge_index = edge_index 
        self.edge_weight= edge_weight

    def to(self, device: torch.device) -> 'GraphData':
        """Move all tensors to the specified device."""
        return GraphData(
            x=self.x.to(device),
            y=self.y.to(device),
            edge_index=self.edge_index.to(device),
            edge_weight=self.edge_weight.to(device)
        )        
    
    def __repr__(self):
        cls = self.__class__.__name__
        info = (
            f"x={tuple(self.x.shape)}, "
            f"y={tuple(self.y.shape)}, "
            f"edge_index={tuple(self.edge_index.shape)}, "
            f"edge_weight={tuple(self.edge_weight.shape)}"
        )
        return f"{cls}({info})"
    
class GraphDataLoader:
    """
    An alternative to the Pytorch DataLoader class
    """    
    def __init__(self, data_list: List[GraphData]):
        self.data_list = data_list

    def __iter__(self):
        return iter(self.data_list)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> GraphData:
        return self.data_list[idx]
    
    def __repr__(self):
        cls = self.__class__.__name__
        if not self.data_list:
            return f"{cls}(empty)"
        
        snapshot = self.data_list[0]

        info = (
            f"{len(self.data_list)} datapoints, "
            f"sample: x={tuple(snapshot.x.shape)}, "
            f"y={tuple(snapshot.y.shape)}, "
            f"edge_index={tuple(snapshot.edge_index.shape)}, "
            f"edge_weight={tuple(snapshot.edge_weight.shape)}"
        )
        return f"{cls}({info})"      
    
@dataclass
class GraphDataLoaderCollection:
    """ 
    Stores all DataLoaders for shallow models

    See Also:
    --------
    ShallowDataLoader
    """
    train:              'GraphDataLoader'
    val:                'GraphDataLoader'   
    test:               'GraphDataLoader'
    main:               'GraphDataLoader'

    def __repr__(self):
        return (f"<GraphDataLoaderCollection(train, val, test, main)>") 

from typing import Tuple
import torch
import numpy as np 
import os

class GraphDataLoaderManager:
    """
    Examples:
    --------
    graphdataloader = GraphDataLoaderManager(epidata_loader_basis).retrieve_graph('identity_graph').construct_dataloaders()    
    """
    def __init__(self, 
                 dataorchestrator: DataOrchestrator):
        
        self.dataorchestrator       = dataorchestrator
        self.column_registration    = dataorchestrator.column_registration

    def retrieve_graph(self, 
                       graphname:     str, 
                       graphdirectory:str = 'data/graphs') -> 'GraphDataLoaderManager':
        """
        """
    
        graphpath  = os.path.join(graphdirectory, self.dataorchestrator.config.nuts_level, graphname)
        
        try:
            self.edge_index  = torch.load(graphpath + '_edge_index.pt', weights_only = False)
            self.edge_weight = torch.load(graphpath + '_edge_weight.pt', weights_only = False)

        except Exception as e:
            raise RuntimeError(f'graph by the name of {graphname}s not found')
        
        return self

    def randomize_edge_weights(self) -> 'GraphDataLoaderManager':
        """
        """

        w_min = self.edge_weight.min().item()
        w_max = self.edge_weight.max().item()

        randomized_weights = torch.rand_like(self.edge_weight) * (w_max - w_min) + w_min 
        self.edge_weight   = randomized_weights
        return self  

    def randomize_edges(self) -> 'GraphDataLoaderManager':

        num_nodes = int(self.edge_index.max().item()) + 1
        num_edges = self.edge_index.shape[1]

        edges_from = torch.randint(low = 0, high = num_nodes, size = (num_edges,))
        edges_to   = torch.randint(low = 0, high = num_nodes, size = (num_edges,))

        self.edge_index = torch.stack([edges_from, edges_to], dim = 0)
        return self
        
    def construct_dataloaders(self):
        """
        Creates the actual dataloaders
        """

        X,y                             = self._construct_Xy(self.dataorchestrator.data_final.data)
        main_dataloader                 = self._construct_main_dataloader(X = X, y = y)
        self.dataloader_collection      = self._split_dataloader(main_dataloader = main_dataloader)
        return self     
    
    def _construct_Xy(self, 
                      df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:

        dfc                 = df.copy()
        feature_arrays      = []
        target_arrays       = []
        timestamps          = list(dfc['timestamp'].unique())

        feature_cols = self.column_registration.get_by_type('feature')
        split_cols   = self.column_registration.get_by_type('split')
        target_cols  = self.column_registration.get_by_type('target')          

        time_splits         = dfc[['timestamp'] + split_cols].drop_duplicates().reset_index(drop = True)
        self.time_splits    = time_splits

        for feat in feature_cols:
            
            # Pivot from long to wide: rows=time, columns=nodes, values=feature
            pivoted = dfc.pivot(index=['timestamp'], columns='node', values=feat).reset_index(drop = True)

            # set missing nodes to zero
            # pivoted = pivoted.reindex(index=timestamps, columns=node_ids)

            # # convert to numeric
            pivoted = pivoted.apply(pd.to_numeric, errors='coerce')
            
            # # Convert to numpy float array, replace NaNs with 0
            arr = pivoted.values
            arr = arr.astype(np.float32)             # force float32 dtype
            feature_arrays.append(arr)

        X_np = np.stack(feature_arrays, axis=-1)

        for target in target_cols:

            for dfc_col in dfc.columns:

                if target in dfc_col and dfc_col not in feature_cols:
                    # Pivot from long to wide: rows=time, columns=nodes, values=feature
                    pivoted = dfc.pivot(index=['timestamp'], columns='node', values=dfc_col).reset_index(drop = True)

                    # set missing nodes to zero
                    # pivoted = pivoted.reindex(index=timestamps, columns=node_ids)

                    # # convert to numeric
                    pivoted = pivoted.apply(pd.to_numeric, errors='coerce')
                    
                    # # Convert to numpy float array, replace NaNs with 0
                    arr = pivoted.values
                    arr = arr.astype(np.float32)             # force float32 dtype
                    target_arrays.append(arr)

        # Process target column using same approach
        y_np = np.stack(target_arrays, axis=-1)

        X = torch.tensor(X_np, dtype=torch.float)
        y = torch.tensor(y_np,dtype=torch.float)
        return X, y

    def _construct_main_dataloader(self, 
                                   X: torch.Tensor,
                                   y: torch.Tensor) -> GraphDataLoader:
        edge_index  = self.edge_index 
        edge_weight = self.edge_weight

        dataset = []
        T       = X.shape[0]  # Total number of timesteps

        # Calculate maximum valid start position
        # Need: start + periods + prediction_horizon - 1 < T
        max_start       = T - self.dataorchestrator.config.horizon_leadtime - (self.dataorchestrator.config.horizon_size - 1) - (self.dataorchestrator.config.sequence_length - 1)
        self.max_start  = max_start

        if max_start <= 0:
            raise ValueError(f"Not enough data: T={T}, periods={self.dataorchestrator.config.sequence_length}"
                            f"Need at least {self.dataorchestrator.config.sequence_length} timesteps.")

        for start in range(max_start):
            # Input window: periods consecutive timesteps
            x_seq = X[start : start + self.dataorchestrator.config.sequence_length]  # shape [periods, nodes, features]
            y_seq = y[start + self.dataorchestrator.config.sequence_length - 1]
            
            data = GraphData(
                x = x_seq.clone().detach().float().permute(1, 2, 0),  # (nodes, features, periods)
                y = y_seq.clone().detach().float(),                   # (nodes, horizon)
                edge_index = edge_index,
                edge_weight =edge_weight
            )
            dataset.append(data)

        return GraphDataLoader(dataset)

    def _split_dataloader(self, 
                          main_dataloader: GraphDataLoader) -> GraphDataLoaderCollection:

        train_idx = list(self.time_splits[self.time_splits['train']].index)
        val_idx   = list(self.time_splits[self.time_splits['val']].index)
        test_idx  = list(self.time_splits[self.time_splits['test']].index)

        dataloader_train = GraphDataLoader([main_dataloader[tt] for tt in train_idx])
        dataloader_val   = GraphDataLoader([main_dataloader[tt] for tt in val_idx])
        dataloader_test  = GraphDataLoader([main_dataloader[tt] for tt in test_idx if tt < self.max_start])        
        
        return GraphDataLoaderCollection(
            train = dataloader_train,
            val   = dataloader_val,
            test  = dataloader_test,
            main  = main_dataloader
        )
