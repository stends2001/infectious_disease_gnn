from typing import List
import torch

class GraphDataLoaderEntry:
    def __init__(self, 
                 x: torch.Tensor, 
                 y: torch.Tensor,
                 edge_index: torch.Tensor,
                 edge_weight: torch.Tensor):
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

class GraphDataLoader:
    def __init__(self, data_list: List[GraphDataLoaderEntry]):
        self.data_list = data_list

    def __iter__(self):
        return iter(self.data_list)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> GraphDataLoaderEntry:
        return self.data_list[idx]