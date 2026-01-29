import os 
import numpy as np
import pandas as pd
import geopandas as gpd
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple, Optional, Literal
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon

from .airportrawgraphbuilder import AirportRawGraphBuilder
from .airportgraphrawprocessor import AirportRawGraphProcessor

if TYPE_CHECKING:
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

    def __repr__(self) -> str:
        return f'<StaticGraphOrchestrator(level {self.nuts_level}. Registry: {self.graph_registry})>'
