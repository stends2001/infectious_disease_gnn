import torch

from .baseloader import DeepBaseDataLoaderManager
from .datacontainers import GraphData, GraphDataList, GraphStructureError


import os
from typing import Optional, Literal

from ....dataloading.epidataorchestration import EpiDataOrchestrator
from ....graphconstruction.containers import GraphStructure
from ....utils.helpers import get_project_utilities_env

class GraphDataLoaderManager(DeepBaseDataLoaderManager):
    """DataLoader manager for graph neural networks"""
    
    def __init__(self, dataorchestrator: EpiDataOrchestrator):
        super().__init__(dataorchestrator)
        self._graphmode: Optional[Literal['static','dynamic']] = None
        self.basedir = get_project_utilities_env()
    
    def retrieve_static_graph(self, graphname: str, graphdirectory: str = 'graphs') -> 'GraphDataLoaderManager':
        graphpath = os.path.join(self.basedir, graphdirectory, 
                                 self.dataorchestrator.config.nuts_level, graphname, graphname)
        
        try:
            i = torch.load(graphpath + '_edge_index.pt', weights_only=False)
            w = torch.load(graphpath + '_edge_weight.pt', weights_only=False)
            graph = GraphStructure(i, w)
        except Exception as e:
            raise RuntimeError(f'graph by the name of {graphname} not found')
        
        self._validate_graphstructure(i, w, graphname)
        self.graph = graph
        self._graphmode = 'static'
        return self
    
    def _create_data_object(self, x_seq: torch.Tensor, y_seq: torch.Tensor):
        """Create GraphData object with edge structure"""
        return GraphData(
            x = x_seq.clone().detach().float().permute(1, 2, 0),
            y = y_seq.clone().detach().float(),
            edge_index = self.graph.edge_index,
            edge_weight = self.graph.edge_weight
        )
    
    def build(self):
        X, y     = self._split_Xyt(self.dataorchestrator.data_final.data)
        datasets = self._build_sequences(X, y)
        
        # Wrap in GraphDataList containers
        self.dataloader_main = GraphDataList(datasets[0])
        self.dataloader_train = GraphDataList(datasets[1])
        self.dataloader_val = GraphDataList(datasets[2])
        self.dataloader_test = GraphDataList(datasets[3])
        return self
    
    def _validate_graphstructure(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, structure_name: str):
        """validates edge index and edge weight"""
        num_nodes_graph = len(edge_index.unique())
        
        if len(edge_index[0]) != len(edge_weight):
            raise GraphStructureError(f'edge_index and edge_weight have a different length')
        
        if num_nodes_graph != self.dataorchestrator.data_context.num_nodes:
            raise GraphStructureError(f'{structure_name} has {num_nodes_graph} nodes while data_orchestrator has {self.dataorchestrator.data_context.num_nodes} nodes')
    
    def __repr__(self):
        return '<GraphDataLoaderManager(dataloaders at .dataloader_main, .dataloader_train, .dataloader_val, .dataloader_test)>'