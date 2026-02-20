from torch import Tensor as Tensor
from typing import Tuple, List, cast 
from .baseloader import DeepBaseDataLoaderManager
from .datacontainers import DeepData, DeepDataList

class DeepDataLoaderManager(DeepBaseDataLoaderManager):
    """DataLoader manager for non-graph deep-models"""
    
    def _create_data_object(self, x_seq: Tensor, y_seq: Tensor) -> 'DeepData':
        """Simple data container without graph structure"""
        return DeepData(
            x_seq.clone().detach().float().permute(1, 2, 0),  # (nodes, features, seq_len)
            y_seq.clone().detach().float()  # (nodes, horizon_size)
        )
    
    def build(self) -> 'DeepDataLoaderManager':
        """
        Orchestrates the entire DeepDataLoaderManager - creation. 
        """        
        X, y     = self._split_Xyt(self.dataorchestrator.data_final.data)

        main, train, val, test = cast(
            tuple[list[DeepData], list[DeepData], list[DeepData], list[DeepData]],
            self._build_sequences(X, y)
        )
                
        # Wrap in simple lists or custom container
        self.dataloader_main    = DeepDataList(main)
        self.dataloader_train   = DeepDataList(train)
        self.dataloader_val     = DeepDataList(val)
        self.dataloader_test    = DeepDataList(test)
        return self