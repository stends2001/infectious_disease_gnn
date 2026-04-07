import torch
from torch import Tensor as Tensor
import os
from typing import Optional, Literal, cast, List
from ..issues import DataEntryError
from ....dataloading.epidataorchestration.orchestrator import EpiDataOrchestrator
from .baseloader import DeepBaseDataLoaderManager
from .datacontainers import GraphData, GraphDataList, GraphStructure

from ....utils.helpers import get_project_utilities_env

class GraphDataLoaderManager(DeepBaseDataLoaderManager):
    """DataLoader manager for GNN-based-approaches"""
    
    def __init__(self, dataorchestrator: EpiDataOrchestrator):
        super().__init__(dataorchestrator)
        self._graphmode: Optional[Literal['static','dynamic']] = None
        self.basedir = get_project_utilities_env()
    
    def retrieve_static_graph(self, graphname: str, graphdirectory: str = 'graphs') -> 'GraphDataLoaderManager':
        """retrieves a static (opposed to dynamic) graph structure"""
        graphpath = os.path.join(self.basedir, graphdirectory, 
                                 self.dataorchestrator.config.level, graphname, graphname)
        
        edge_index      = torch.load(graphpath + '_edge_index.pt', weights_only=False)
        edge_weight     = torch.load(graphpath + '_edge_weight.pt', weights_only=False)
        graph_structure = GraphStructure(edge_index, edge_weight)

        self.graph      = graph_structure
        self._graphmode = 'static'
        return self
    
    def _create_data_object(self, x_seq: Tensor, y_seq: Tensor) -> 'GraphData':
        """Create GraphData object with graphstructure"""
        return GraphData(
            x = x_seq.clone().detach().float().permute(1, 2, 0),
            y = y_seq.clone().detach().float(),
            edge_index  = self.graph.edge_index,
            edge_weight = self.graph.edge_weight
        )
    
    def build(self) -> 'GraphDataLoaderManager':
        """
        Orchestrates the entire GraphDataLoaderManager - creation. 

        First retrieve the graph structure
        """
        if self._graphmode is None:
            raise DataEntryError("No graph loaded. Call retrieve_static_graph() before build()")

        X, y     = self._split_Xyt(self.dataorchestrator.data_final.data)

        main, train, val, test = cast(
            tuple[list[GraphData], list[GraphData], list[GraphData], list[GraphData]],
            self._build_sequences(X, y)
        )
                
        # Wrap in simple lists or custom container
        self._dataloader_main    = GraphDataList(main)
        self._dataloader_train   = GraphDataList(train)
        self._dataloader_val     = GraphDataList(val)
        self._dataloader_test    = GraphDataList(test)        
        return self