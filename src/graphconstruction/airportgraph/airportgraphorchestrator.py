import os 
import numpy as np
import pandas as pd
import geopandas as gpd
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple, Optional, Literal, List
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon

import torch

from .airportrawgraphbuilder import AirportRawGraphBuilder
from .airportgraphrawprocessor import AirportRawGraphProcessor

if TYPE_CHECKING:
    from .graphstructure import GraphStructure
    from ...dataloading.airp import AirpOrchestrator

class ShapefileError(Exception):
    def __init__(self, explanation: str):
        statement = "GraphBuilderL2 couldnt be initialized" + "\n" + explanation
        super().__init__(statement)

class AirportGraphOrchestrator():
    """
    Orchestrates creation of static graph airport - graph-structures. not the nuts-graphs!

    HL2: airports
    HL3: nuts

    GL2: airports>nuts
    GL3: nuts>nuts

    Examples
    --------
    graph_orchestrator   = AirportGraphOrchestrator(airp_orchestrator)
    graph_structure      = graph_orchestrator.generate_graphstructure('distance',  decay = 'boolean')
    graph_previewer = AirportGraphViewer(graph_orchestrator, airp_orchestrator.data_context,graph_structure)
    figure          = graph_previewer.view(graphname = 'a', node_layer2=0).ticks.change_xticks([]).ticks.change_yticks([])
    figure.labels.change_suptitle('Boolean distance based graph', fontweight='bold', fontsize = 13).labels.set_suptitle_y(0.93)
    figure.show()    

    graph_orchestrator.save_graphstructures(graph_structure, 'boolean_distance')
    """
    def __init__(self,                  
                 data_orchestrator: 'AirpOrchestrator',
                 graph_dir:         str = "data/graphs/airports"):
        
        # extract metadata
        self.context_data       = data_orchestrator.data_context

        self.airport_shapefile  = self.context_data.airport_shapefile.rename(columns = {'id2':'node_layer2'})
        self.nuts_shapefile     = self.context_data.nuts_shapefile.rename(columns = {'node':'node_layer3'})
        
        # self.nuts_level      = self.context_data.nuts_level
        self.graph_dir          = graph_dir
        self.graph_methods      = ['distance']

        # registry of graphs
        os.makedirs(self.graph_dir, exist_ok=True)
        self.graph_registry = None
          
    def generate_graphstructure(self, 
                                graph_method: Literal['distance'],
                                edge_normalization_method = None,
                                **kwargs):
        """
        """
        
        if graph_method not in self.graph_methods:
            raise ValueError(f'invalid method {graph_method} for StaticGraphOrchestrator. Supported methods: {self.graph_methods}')
        
        creator = AirportRawGraphBuilder(self.nuts_shapefile, self.airport_shapefile)

        # Generate graph structure through 3 steps:
        # 1. generate an instance of GraphConnectionsDataFrame like: 
        #   | node_layer2 | node_layer3 | weight | 
        # 2. graph processing: edge-weight normalization and tensorization

        # ============ STEP 1 ============= #

        if graph_method == 'distance':
            raw_graph = creator.distance(**kwargs)

        # ============ STEP 2 ============= #
        processed_graph = AirportRawGraphProcessor(raw_graph, edge_normalization_method).process()
        return processed_graph

    def save_graphstructures(self, graph_structure: 'GraphStructure', graphname: str):
        """
        """
        directory   = os.path.join(self.graph_dir, graphname)
        os.makedirs(directory, exist_ok=True)

        edge_index  = graph_structure.edge_index 
        edge_weight = graph_structure.edge_weight

        torch.save(edge_index, os.path.join(directory,  f'{graphname}_edge_index.pt'))
        torch.save(edge_weight, os.path.join(directory, f'{graphname}_edge_weight.pt'))

        print(f'{graphname} saved')

        

    # TODO
    def __repr__(self) -> str:
        return f'...'
