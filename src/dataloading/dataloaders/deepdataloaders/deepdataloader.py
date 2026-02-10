import torch

from .baseloader import DeepBaseDataLoaderManager
from .datacontainers import DeepData 

class DeepDataLoaderManager(DeepBaseDataLoaderManager):
    """DataLoader manager for non-graph models (LSTMs, MLPs, etc.)"""
    
    def _create_data_object(self, x_seq: torch.Tensor, y_seq: torch.Tensor):
        """Simple data container without graph structure"""
        return DeepData(
            x_seq.clone().detach().float().permute(1, 2, 0),  # (nodes, features, seq_len)
            y_seq.clone().detach().float()  # (nodes, horizon_size)
        )
    
    def build(self):
        X, y     = self._split_Xyt(self.dataorchestrator.data_final.data)
        datasets = self._build_sequences(X, y)
        
        # Wrap in simple lists or custom container
        self.dataloader_main = datasets[0]
        self.dataloader_train = datasets[1]
        self.dataloader_val = datasets[2]
        self.dataloader_test = datasets[3]
        return self
    
    def __repr__(self):
        return '<DeepDataLoaderManager(dataloaders at .dataloader_main, .dataloader_train, .dataloader_val, .dataloader_test)>'
