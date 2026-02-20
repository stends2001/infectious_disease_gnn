from torch import Tensor as Tensor
import torch
from typing import List
from dataclasses import dataclass

from ..issues import DataEntryError
from ....issues import IssueReport

class DeepData:
    """
    An alternative to the Pytorch Data class for data-entries of X, y

    Parameters
    ----------
    x: Tensor
        input data of shape [num_nodes, num_features, seq_len]
    y: Tensor
        target data of shape [num_nodes, horizon_size]
    """
    def __init__(self, 
                 x:             Tensor, 
                 y:             Tensor):
        
        self.x          = x
        self.y          = y

        self.__post_init_validate()

    def __post_init_validate(self) -> None:
        """validate entry against expected shapes"""
        data_entry_validation_errors = []
        
        if self.x.dim() != 3:
            data_entry_validation_errors.append(DataEntryError(f"x must be 3D [num_nodes, seq_len, num_features], got {self.x.dim()}D"))
        
        if self.y.dim() != 2:
            data_entry_validation_errors.append(DataEntryError(f"y must be 2D [num_nodes, horizon_size], got {self.y.dim()}D"))

        if len(data_entry_validation_errors):
            raise IssueReport(data_entry_validation_errors, f'Data Entry {self.__class__.__name__} couldnt be created')        

    def to(self, device: torch.device) -> 'DeepData':
        """Move all tensors to the specified device (GPU)"""
        return DeepData(
            x=self.x.to(device),
            y=self.y.to(device)
        )        
    
    def __repr__(self):
        cls = self.__class__.__name__
        info = (
            f"x={tuple(self.x.shape)}, "
            f"y={tuple(self.y.shape)}"
        )
        return f"{cls}({info})"
    
class DeepDataList:
    """
    An alternative to the Pytorch DataLoader class
    Basically a list of DeepData entries
    """    
    def __init__(self, data_list: List[DeepData]):
        self.data_list = data_list

        self.__post_init_validate()

    def __post_init_validate(self) -> None:
        """validate entry against expected shapes"""
        data_entry_validation_errors = []
        
        if len(self.data_list) == 0:
            data_entry_validation_errors.append(DataEntryError(f"got empty data_list"))

        elif not isinstance(self.data_list[0], DeepData):
            data_entry_validation_errors.append(DataEntryError(f"got unexpected type {self.data_list[0].__class__.__name__}"))            

        if len(data_entry_validation_errors):
            raise IssueReport(data_entry_validation_errors, f'Data Entry {self.__class__.__name__} couldnt be created')        
       
    def __iter__(self):
        return iter(self.data_list)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> DeepData:
        return self.data_list[idx]
    
    def __repr__(self):
        cls = self.__class__.__name__
        snapshot = self.data_list[0]
        info = (
            f"{len(self.data_list)} datapoints, "
            f"sample: x={tuple(snapshot.x.shape)}, "
            f"y={tuple(snapshot.y.shape)}"
        )
        return f"{cls}({info})"      

@dataclass 
class GraphStructure:
    """
    edge_index:     torch.Tensor 
    edge_weight:    torch.Tensor   
    """
    edge_index:     torch.Tensor 
    edge_weight:    torch.Tensor    

    def __post_init__(self):
        if self.edge_index.shape[1] != len(self.edge_weight):
            raise ValueError(
                f"Incompatiable shapes of edge_index (len = {self.edge_index.shape[1]}) and edge_weight (len = {len(self.edge_weight)}) in GraphStructure"
            )
        
        self.num_nodes      = len(self.edge_index[0].unique())
        self.num_edges      = len(self.edge_index)
    
    def __repr__(self) -> str:
        representation = f'<GraphStructure(num_nodes = {self.num_nodes}, num_edges = {len(self.edge_weight)})>'
        return representation 

class GraphData(DeepData):
    """
    An alternative to the Pytorch Data class for data-entries of X, y, edge_index and edge_weight

    Extension to DeepData, but with a graphstructure

    Parameters
    ----------
    x: Tensor
        input data of shape [num_nodes, num_features, seq_len]
    y: Tensor
        target data of shape [num_nodes, horizon_size]
    edge_index: Tensor
        edge_index of a graphstructure of shape [2, num_edges]
    edge_weight: Tensor
        edge_weight of a graphstructure of shape [num_edges]
    """
    def __init__(self, 
                 x:             Tensor, 
                 y:             Tensor,
                 edge_index:    Tensor,
                 edge_weight:   Tensor):
        
        self.edge_index  = edge_index
        self.edge_weight = edge_weight

        super().__init__(x, y)      # runs DeepData validation for x and y
        self.__post_init_validate() # only edge-specific checks here

    def __post_init_validate(self) -> None:
        """validate entry against expected shapes"""
        data_entry_validation_errors = []
        
        if self.edge_index.shape[0] != 2:
            data_entry_validation_errors.append(DataEntryError(f"edge_index must have shape [2, num_edges], got {tuple(self.edge_index.shape)}"))
        
        if self.edge_index.shape[1] != self.edge_weight.shape[0]:
            data_entry_validation_errors.append(DataEntryError(f"edge_index and edge_weight must have same number of edges got {self.edge_index.shape[1]} and {self.edge_weight.shape[0]}"))
           
        if len(data_entry_validation_errors):
            raise IssueReport(data_entry_validation_errors, f'Data Entry {self.__class__.__name__} couldnt be created')

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
    
class GraphDataList:
    """
    An alternative to the Pytorch DataLoader class
    Basically a list of GraphData entries

    Extension to DeepData, but with a graphstructure    
    """    
    def __init__(self, data_list: List[GraphData]):
        self.data_list = data_list

        self.__post_init_validate()

    def __post_init_validate(self) -> None:
        """validate entry against expected shapes"""
        data_entry_validation_errors = []
        
        if len(self.data_list) == 0:
            data_entry_validation_errors.append(DataEntryError(f"got empty data_list"))

        elif not isinstance(self.data_list[0], GraphData):
            data_entry_validation_errors.append(DataEntryError(f"got unexpected type {self.data_list[0].__class__.__name__}"))            

        if len(data_entry_validation_errors):
            raise IssueReport(data_entry_validation_errors, f'Data Entry {self.__class__.__name__} couldnt be created')        

    def __iter__(self):
        return iter(self.data_list)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> GraphData:
        return self.data_list[idx]
    
    def __repr__(self):
        cls = self.__class__.__name__
        snapshot = self.data_list[0]
        info = (
            f"{len(self.data_list)} datapoints, "
            f"sample: x={tuple(snapshot.x.shape)}, "
            f"y={tuple(snapshot.y.shape)}, "
            f"edge_index={tuple(snapshot.edge_index.shape)}, "
            f"edge_weight={tuple(snapshot.edge_weight.shape)}"
        )
        return f"{cls}({info})"      
