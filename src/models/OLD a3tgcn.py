from tqdm import tqdm
from torch_geometric_temporal.nn.recurrent import A3TGCN
import torch.nn.functional as F
import torch.nn as nn
import torch
from .modelcore import DeepLearningModelCore
from ..dataloading.gnndataloader import GNNDataLoader
from typing import Optional


class A3TGCNArchitecture(nn.Module):
    def __init__(self, node_features, a3tgcn_dim, periods):
        super(A3TGCNArchitecture, self).__init__()
        self.recurrent = A3TGCN(node_features, a3tgcn_dim, periods)
        self.linear = nn.Linear(a3tgcn_dim, 1)

    def forward(self, x, edge_index, edge_weight):
        x = x.squeeze(2)
        h = self.recurrent(x.view(x.shape[0], 1, x.shape[1]), edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
        

class A3TGCN(DeepLearningModelCore):
    """
    Pytorch - example of a A3TGCN model.
    Inspired from : https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/examples/recurrent/a3tgcn_example.py

    TODO: deal with multitude of features that are not related to the lags (week sin/cos)
    """
    def __init__(self, 
                 dataloader: GNNDataLoader, 
                 name: Optional[str] = None):
        super().__init__(dataloader, name=name)
        if not self.name:
            self.name = 'pytorch example - A3TGCN'

        self.model_color = '#4ECDC4'
        self.dataloader = dataloader

    def set_model_hparams(self, a3tgcn_dim):
        self.model_hparams_set = True
        self.model = A3TGCNArchitecture(
            node_features=1,
            a3tgcn_dim = a3tgcn_dim,
            periods = self.dataloader.periods, 
        ).to(self.device)

        return self