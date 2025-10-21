from .epidataloader import EpiDataLoader, _reorder_df
import pandas as pd
import numpy as np
import torch
from typing import Literal, Optional, Union, List, Tuple
from typing import cast
from .dataobjects import GraphDataLoaderEntry, GraphDataLoader

import os
import matplotlib.pyplot as plt
import copy

class DeepDataLoader(EpiDataLoader):
    """
    creates an instance with attributes
    .dataset_train
    .dataset_val
    .dataset_test

    which have an X, y, edge_index and edge_weight (the latter only when applicable!)

    X has shape: [N, F, periods] (nodes/features/periods)
    """
    def __init__(self, 
                 disease_name: str,
                 data_env_dir: str,
                 min_date:     str     = '2001-01-01',
                 max_date:     str     = '2025-01-01',
                 nuts_level:   Literal['nuts1','nuts2','nuts3'] = 'nuts3',
                 include_population: bool = False,
                 horizon_size: int     = 1,
                 horizon_leadtime:int  = 1,
                 sequence_length: int  = 1,
                 split_berlin: bool    = True,
                 verbose: bool         = True):
        self.task_config = {}

        super().__init__(disease_name, data_env_dir, min_date, max_date, nuts_level, include_population, horizon_size, horizon_leadtime, sequence_length, split_berlin, verbose)
         
        self.edge_index:  Optional[torch.Tensor] = None
        self.edge_weight: Optional[torch.Tensor] = None
        self.dataloader_train, self.dataloader_val, self.dataloader_test = None, None, None

    def construct_dataloaders(self):
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

        X,y             = self._construct_Xy(self.data['final'])

        main_dataloader = self._construct_main_dataloader(X = X, y = y, sequence_length = self.sequence_length)
        dataloaders     = self._split_dataloader(main_dataloader = main_dataloader)
        self.dataloader_main = main_dataloader
        self.dataloader_train, self.dataloader_val, self.dataloader_test = dataloaders
        return self     
    
    def _construct_main_dataloader(self, 
                              X: torch.Tensor,
                              y: torch.Tensor, 
                              sequence_length: int) -> GraphDataLoader:
        edge_index  = self.edge_index 
        edge_weight = self.edge_weight

        if edge_index is None:
            raise ValueError('no edge index found')
        if edge_weight is None:
            raise ValueError('no edge weight found')

        dataset = []
        T       = X.shape[0]  # Total number of timesteps

        # Calculate maximum valid start position
        # Need: start + periods + prediction_horizon - 1 < T
        max_start = T - sequence_length - self.horizon_leadtime - self.horizon_size + 1
        self.max_start = max_start
        if max_start <= 0:
            raise ValueError(f"Not enough data: T={T}, periods={sequence_length}"
                            f"Need at least {sequence_length} timesteps.")

        for start in range(max_start):
            # Input window: periods consecutive timesteps
            x_seq = X[start : start + sequence_length]  # shape [periods, nodes, features]
            y_seq = y[start + sequence_length -1]
            
            data = GraphDataLoaderEntry(
                x = x_seq.clone().detach().float().permute(1, 2, 0),  # (nodes, features, periods)
                y = y_seq.clone().detach().float(),                   # (nodes, horizon)
                edge_index = edge_index,
                edge_weight =edge_weight
            )
            dataset.append(data)

        return GraphDataLoader(dataset)

    def _split_dataloader(self, 
                          main_dataloader: GraphDataLoader) -> Tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]:

        train_idx = list(self.time_splits[self.time_splits['train']].index)
        val_idx   = list(self.time_splits[self.time_splits['val']].index)
        test_idx  = list(self.time_splits[self.time_splits['test']].index)

        dataloader_train = GraphDataLoader([main_dataloader[tt] for tt in train_idx])
        dataloader_val   = GraphDataLoader([main_dataloader[tt] for tt in val_idx])
        dataloader_test  = GraphDataLoader([main_dataloader[tt] for tt in test_idx if tt < self.max_start])        
        
        return dataloader_train, dataloader_val, dataloader_test

    def _construct_Xy(self, 
                      df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:

        dfc            = df.copy()
        feature_arrays = []
        target_arrays  = []
        timestamps     = list(dfc[self.temporal_column].unique())
        time_splits    = dfc[[self.temporal_column] + self.split_columns].drop_duplicates().reset_index(drop = True)

        self.time_splits = time_splits

        for feat in self.feature_columns:
            
            # Pivot from long to wide: rows=time, columns=nodes, values=feature
            pivoted = dfc.pivot(index=['timestamp'], columns=self.id_column, values=feat).reset_index(drop = True)

            # set missing nodes to zero
            # pivoted = pivoted.reindex(index=timestamps, columns=node_ids)

            # # convert to numeric
            pivoted = pivoted.apply(pd.to_numeric, errors='coerce')
            
            # # Convert to numpy float array, replace NaNs with 0
            arr = pivoted.values
            arr = arr.astype(np.float32)             # force float32 dtype
            feature_arrays.append(arr)

        X_np = np.stack(feature_arrays, axis=-1)

        for target in self.target_horizons:

            # Pivot from long to wide: rows=time, columns=nodes, values=feature
            pivoted = dfc.pivot(index=['timestamp'], columns=self.id_column, values=target).reset_index(drop = True)

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

    def retrieve_graph(self, 
                       graphname:     str, 
                       graphdirectory:str ='data/graphs'):
        """
        Loads graphstructure into dataloaders

        sets the following attributes in the class
        - edge_index 
        - edge_weight
        """
    
        graphpath  = os.path.join(graphdirectory, self.nuts_level, graphname)
        
        try:
            edge_index = torch.load(graphpath + '_edge_index.pt', weights_only = False)

        except Exception as e:
            raise RuntimeError(f'graph by the name of {graphname}s not found')


        # Try loading edge_weight, fallback to ones if file not found
        try:
            edge_weight = torch.load(graphpath + '_edge_weight.pt', weights_only = False)

        # else weights are uniformly 1
        except FileNotFoundError:
            num_edges   = edge_index.shape[1]
            edge_weight = torch.ones(num_edges, dtype=torch.float)

        self.edge_index  = edge_index
        self.edge_weight = edge_weight

        # self.task_config['graph'] = {'graphname': graphname,
        #                              'graphdirectory' : graphdirectory}
        return self
   
    def preview_dataloader(self, 
                        node_idx: int, 
                        timepoint: int = 22, 
                        dataset: Literal['train','val','test'] = 'test') -> 'DeepDataLoader':

        if dataset == 'train':
            df = self.dataloader_train
        elif dataset == 'val':
            df = self.dataloader_val 
        elif dataset == 'test':
            df = self.dataloader_test
        else:
            raise ValueError(f'{dataset} is an invalid dataset')
        
        if df is None:
            raise ValueError(f'no dataloader found under {dataset}')
        
        assert all(entry.x is not None for entry in df), "Some Data objects are missing 'x'"


        lags = self.lags

        lag_cols = []
        lag_cols_idx = []
        for idx, cc in enumerate(self.feature_columns):
            if 'lag' in cc:
                lag_cols.append(cc)
                lag_cols_idx.append(idx)

        if self.sequence_length > 1:
            
            dataX         = torch.stack([entry.x[node_idx, lag_cols_idx, :] for entry in df]).cpu().numpy()  # all inputs [tt, features, periods]
            last_elements = dataX[:, 0, 1:]
            to_append     = last_elements[:, ::-1]
            dataX         = np.concatenate((to_append,dataX[:,:,0]), axis = 1)


        else:
            dataX = torch.stack([entry.x[node_idx, lag_cols_idx, 0] for entry in df]).cpu().numpy()  # all inputs [tt, features, periods]


        dataY = torch.stack([entry.y[node_idx, :] for entry in df]).cpu().numpy()                # all targets of the pred horizon [tt, horizon]

        input = dataX[timepoint,:]

        input = input[::-1]

        target= dataY[timepoint]

        fig, ax = plt.subplots(figsize = (14,6))
        ax.plot(dataY[:,0], '-o' ,markersize = 5, label=f'entire timeseries for node {node_idx}')
        ax.plot(np.arange(timepoint-len(input)-lags[0]+1-self.horizon_leadtime,timepoint-lags[0]+1-self.horizon_leadtime),input, label='input for selected point', color = "#1b9e77", marker='s', markersize=10)
        ax.plot(np.arange(timepoint,timepoint+self.horizon_size),target, marker='o', markersize=10, color='#d94e4e', label='Target last point')
        ax.set_title(f'Input vs Target of node {node_idx}')
        ax.legend()
        ax.grid()
        return self

    def copy(self, deep: bool = True) -> 'DeepDataLoader':
        """
        Create a copy of the DeepDataLoader instance.
        
        Parameters:
        -----------
        deep : bool, default True
            If True, creates a deep copy (independent copy of all data).
            If False, creates a shallow copy (references to same data objects).
            
        Returns:
        --------
        DeepDataLoader
            A copy of the current instance
        """
        # Create new instance with same initialization parameters
        new_instance = DeepDataLoader(
            disease_name       = self.disease,
            data_env_dir       = self.data_env_dir,
            min_date           = self.og_min_date if isinstance(self.og_min_date, str) else self.og_min_date.strftime('%Y-%m-%d'),
            max_date           = self.max_date if isinstance(self.max_date, str) else self.max_date.strftime('%Y-%m-%d'),
            nuts_level         = cast(Literal['nuts1', 'nuts2', 'nuts3'], self.nuts_level),
            include_population = self.include_population,
            horizon_size       = self.horizon_size,
            horizon_leadtime   = self.horizon_leadtime,
            sequence_length    = self.sequence_length,
            split_berlin       = self.split_berlin,
            verbose            = False
        )
        
        
        # Copy all attributes from parent class (EpiDataLoader)
        if deep:
            new_instance.target_column = copy.deepcopy(self.target_column)

            # Deep copy data structures
            if hasattr(self, 'data') and self.data:
                new_instance.data = copy.deepcopy(self.data)
            
            if hasattr(self, 'split_berlin'):
                new_instance.split_berlin = self.split_berlin
            # Copy split information
            if hasattr(self, 'time_splits'):
                new_instance.time_splits = self.time_splits.copy()
            
            if hasattr(self, 'split_summary'):
                new_instance.split_summary = self.split_summary

            if hasattr(self, 'transform_params'):
                new_instance.transform_params = self.transform_params                
                
            # Copy column definitions
            for attr in ['feature_columns', 'split_columns','target_horizons']:
                if hasattr(self, attr):
                    setattr(new_instance, attr, copy.deepcopy(getattr(self, attr)))
            
            # Copy graph structures
            if self.edge_index is not None:
                new_instance.edge_index = self.edge_index.clone()
            if self.edge_weight is not None:
                new_instance.edge_weight = self.edge_weight.clone()
                
            # Copy dataloaders (if they exist)
            for attr in ['dataloader_main', 'dataloader_train', 'dataloader_val', 'dataloader_test']:
                if hasattr(self, attr):
                    original_loader = getattr(self, attr)
                    if original_loader:
                        # Deep copy each Data object in the dataloader
                        if isinstance(original_loader, list):
                            new_loader = []
                            for data_obj in original_loader:
                                new_data = copy.deepcopy(data_obj)
                                new_loader.append(new_data)
                            setattr(new_instance, attr, new_loader)
                        else:
                            setattr(new_instance, attr, copy.deepcopy(original_loader))
        else:
            # Shallow copy - reference same objects
            for attr_name in dir(self):
                if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                    try:
                        setattr(new_instance, attr_name, getattr(self, attr_name))
                    except AttributeError:
                        # Skip read-only attributes
                        pass
        
        # Copy other important attributes
        for attr in ['max_start', 'lags']:
            if hasattr(self, attr):
                setattr(new_instance, attr, getattr(self, attr))
        
        return new_instance

    def randomize_edge_weights(self) -> 'DeepDataLoader':
        """
        Sanity check by giving each existing edge a random weights within the same min/max range as origional

        First retrieve a graph, then use this method.

        Optionally, use the `randomize_edges` in addition.
        """

        w_min = self.edge_weight.min().item()
        w_max = self.edge_weight.max().item()

        randomized_weights = torch.rand_like(self.edge_weight) * (w_max - w_min) + w_min 
        self.edge_weight   = randomized_weights
        return self  

    def randomize_edges(self) -> 'DeepDataLoader':

        num_nodes = int(self.edge_index.max().item()) + 1
        num_edges = self.edge_index.shape[1]

        edges_from = torch.randint(low = 0, high = num_nodes, size = (num_edges,))
        edges_to   = torch.randint(low = 0, high = num_nodes, size = (num_edges,))

        self.edge_index = torch.stack([edges_from, edges_to], dim = 0)
        return self
        


def add_horizon_shifts(df: pd.DataFrame, group_column: str, target_column: str, horizons: int):
    """
    For each node, create horizon-shifted columns for the given column.
    E.g., incidence_h1, incidence_h2, ..., incidence_hN
    
    Parameters:
    - df: pandas DataFrame with columns 'node' and the target column
    - column: str, the column to shift (default 'incidence')
    - horizons: int, number of horizons/shifts to create
    
    Returns:
    - df with new columns added
    """
    df = df.copy()

    for h in range(horizons):
        col_name = f"{target_column}_h{h}"
        # Shift backward to get future values (h steps ahead)
        df[col_name] = df.groupby(group_column)[target_column].shift(-h)
    df.drop(labels = [target_column], axis = 1, inplace = True)
    
    return df.dropna()
