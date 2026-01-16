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

class GraphData:
    """
    An alternative to the Pytorch Data class for dataentries of X, y, edge_index and edge_weight

    Parameters
    ----------
    x: torch.Tensor
        input data of shape ...
    y: torch.Tensor
        target data of shape ...
    edge_index: torch.Tensor
        edge_index of a graphstructure
    edge_weight: torch.Tensor
        edge_weight of a graphstructure
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
        """Move all tensors to the specified device (GPU)"""
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
    Basically a list of GraphData entries
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
    Stores all DataLoaders for GNN - dataloaders in attributes train/val/test/main

    Parameters
    ----------
    train: 'GraphDataLoader

    val: 'GraphDataLoader

    test: 'GraphDataLoader

    main: 'GraphDataLoader

    Downstream
    --------
    GraphDataLoaderManager: class that manages and stores GraphDataLoaderCollection
    """
    train:              'GraphDataLoader'
    val:                'GraphDataLoader'   
    test:               'GraphDataLoader'
    main:               'GraphDataLoader'

    def __repr__(self):
        return (f"<GraphDataLoaderCollection(train, val, test, main)>") 

class GraphDataLoaderManager:
    """
    DataLoader manager for GNNs. 
    An updated version of the DeepLoader class in the previous version.
    Uses the previously constructed dataorchestrator.

    Parameters
    ----------
    dataorchestrator: DataOrchestrator
        built dataorchestrator - object

    Examples
    --------
    >>> dataorchestrator= (DataOrchestrator(config).build())
    >>> graphdataloader = (GraphDataLoaderManager(dataorchestrator)
                            .retrieve_graph('identity_graph')
                            .construct_dataloaders()
                            )
    """
    def __init__(self, 
                 dataorchestrator: DataOrchestrator):
        
        self.dataorchestrator       = dataorchestrator
        self.column_registration    = dataorchestrator.column_registration
        self._graphmode: Optional[Literal['static','dynamic']] = None

    def retrieve_dynamic_graph(self, graphname:str , timesteps: Optional[List[str]] = None, frequency: str = 'yearly', graphdirectory:str = 'data/graphs'):
        """
        retrieves graphs in the folder and sets them as an instance of DynamicGraphStructure into self.graphs
        
        #TODO when timesteps = None => loop over files and import them all extracting timestep from filename
        """
        
        if isinstance(timesteps, range):
            timesteps = list(timesteps)
            timesteps = [str(t) for t in timesteps]

        graphpath  = os.path.join(graphdirectory, self.dataorchestrator.config.nuts_level, graphname)

        if frequency.lower() != 'yearly':
            raise ValueError(f'frequency of {frequency} is unsupported. Currently only yearly frequency is valid.')
        
        graphs = []
        for year in tqdm(timesteps, total = len(timesteps), desc = 'loading graphs'):

            name_t = f'{year}'

            try:
                i       = torch.load(graphpath + "/" + name_t + '_edge_index.pt', weights_only = False)
                w       = torch.load(graphpath + "/" + name_t  + '_edge_weight.pt', weights_only = False)
                self._validate_graphstructure(i, w, name_t)
                graph_t = GraphStructure(i, w)
                graphs.append(graph_t)

            except Exception as e:
                raise RuntimeError(f'graph by the name of {name_t} not found in {graphpath}')
            
        self.graph      = DynamicGraphStructure(timesteps, graphs)
        self._graphmode = 'dynamic'
        
        return self
            
    def retrieve_static_graph(self, graphname: str, graphdirectory: str = 'data/graphs') -> 'GraphDataLoaderManager':
        """
        load a graph object.

        Parameters
        ----------
        graphname: str
            The name of the file. Should correspond to the filename: f'{graphdirectory}/{graphname}/_edge_index.pt' and edge_weight.
        graphdirectory: str = 'data/graphs'
        """
    
        graphpath  = os.path.join(graphdirectory, self.dataorchestrator.config.nuts_level, graphname, graphname)
        
        try:
            i           = torch.load(graphpath + '_edge_index.pt', weights_only = False)
            w           = torch.load(graphpath + '_edge_weight.pt', weights_only = False)
            graph       = GraphStructure(i,w)
           
        except Exception as e:
            raise RuntimeError(f'graph by the name of {graphname} not found')
    
        self._validate_graphstructure(i, w, graphname)

        self.graph      = graph
        self._graphmode = 'static'
        
        return self

    def randomize_edge_weights(self) -> 'GraphDataLoaderManager':
        """
        randomizes edge weights to inspect whether a difference is observed in predictions.
        Only the case if edge_weights are intriniscally taken into account (no attention!)
        """

        w_min = self.edge_weight.min().item()
        w_max = self.edge_weight.max().item()

        randomized_weights = torch.rand_like(self.edge_weight) * (w_max - w_min) + w_min 
        self.edge_weight   = randomized_weights
        return self  

    def randomize_edges(self) -> 'GraphDataLoaderManager':
        """
        Randomly shuffles edge_index. edge_weights are unchanged.
        """
        num_nodes = int(self.edge_index.max().item()) + 1
        num_edges = self.edge_index.shape[1]

        edges_from = torch.randint(low = 0, high = num_nodes, size = (num_edges,))
        edges_to   = torch.randint(low = 0, high = num_nodes, size = (num_edges,))

        self.edge_index = torch.stack([edges_from, edges_to], dim = 0)
        return self
        
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
                                   t: List[str]) -> GraphDataLoader:
        """
        construct main dataloader based on X, y, edge_index and edge_weight

        Returns
        -------
        GraphDataLoader
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

            if self._graphmode == 'dynamic':
                year            = t[t_idx].year
                graphstructure_t = self.graph.get_snapshot(str(year))

            else:
                graphstructure_t = self.graph
            
            data = GraphData(
                x = x_seq.clone().detach().float().permute(1, 2, 0),  # (nodes, features, periods)
                y = y_seq.clone().detach().float(),                   # (nodes, horizon)
                edge_index = graphstructure_t.edge_index,
                edge_weight =graphstructure_t.edge_weight
            )
            dataset.append(data)

        return GraphDataLoader(dataset)

    def _split_dataloader(self, 
                          main_dataloader: GraphDataLoader) -> GraphDataLoaderCollection:
        """
        Splits main dataloader into those for train/val/test
        """        

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

    def _validate_graphstructure(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, structure_name: str):
        """validates edge index and edge weight; num_nodes is compared to each other and to data_orchestrator"""
        num_nodes_graph = len(edge_index.unique())
        
        if len(edge_index[0]) != len(edge_weight):
            raise GraphStructureError(f'{error_emoji} {structure_name}: edge_index and edge_weight have a different length')
        
        if num_nodes_graph != self.dataorchestrator.data_context.num_nodes:
            raise GraphStructureError(f'{error_emoji} {structure_name}: has {num_nodes_graph} nodes while data_orchestrator has {self.dataorchestrator.data_context.num_nodes} nodes')
        
    def __repr__(self) -> str:

        representation = f'<GraphDataLoaderManager(dataloaders at .dataloader_collection)>'          
        return representation