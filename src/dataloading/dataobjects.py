from typing import List
import torch

class GraphDataLoaderEntry:
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

    def to(self, device: torch.device) -> 'GraphDataLoaderEntry':
        """Move all tensors to the specified device."""
        return GraphDataLoaderEntry(
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
    def __init__(self, data_list: List[GraphDataLoaderEntry]):
        self.data_list = data_list

    def __iter__(self):
        return iter(self.data_list)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> GraphDataLoaderEntry:
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