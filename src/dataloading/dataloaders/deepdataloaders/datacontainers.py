import torch
from typing import List

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
    
class GraphDataList:
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

class DeepData:
    """
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
