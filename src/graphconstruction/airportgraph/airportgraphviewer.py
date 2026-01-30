from src.dataloading.airp.airpdatacontainers import ContextAirpData
import seaborn as sns 
from matplotlib.axes import Axes
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely import union_all
from typing import Optional, List, TYPE_CHECKING
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from ...utils import get_data_env

if TYPE_CHECKING:
    from .graphstructure import GraphStructure
    from .airportgraphorchestrator import AirportGraphOrchestrator
    from ...plotting import ManagedFigure

from ...plotting import convert_managedfigure

shapefile_germany_nuts1 = gpd.read_file(os.path.join(get_data_env(),'processed/germany/geospatial/shapefiles/shape_nuts1.shp'))

lightred_color = '#FFB6B6'

palette_blues  = sns.color_palette("Blues",  n_colors = 100)
palette_reds   = sns.color_palette("Reds",   n_colors = 100)
palette_greens = sns.color_palette("Greens", n_colors = 100)

class AirportGraphViewer():
    
    def __init__(self, airport_graph_orchestrator: 'AirportGraphOrchestrator', airport_context_data: 'ContextAirpData', graphstructure: 'GraphStructure'):
        self.airport_graph_orchestrator = airport_graph_orchestrator
        self.airport_context_data = airport_context_data
        self.graphstructure       = graphstructure
        self.figsize              = (10,10)

    @convert_managedfigure
    def view(self, graphname: str, node_layer2: Optional[int] = None, node_layer3: Optional[int] = None)-> 'ManagedFigure':

        if node_layer2 is not None and node_layer3 is not None:
            raise ValueError(f'when previewing a nested graph, please supply either node_layer2 or node_layer3, not both.')    

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # ======= basic makeup ======== #
        self._plot_shape_edge(ax, self.airport_context_data.nuts_shapefile)     # national border
        self._plot_shape(ax, self.airport_context_data.nuts_shapefile, color = 'white', edgecolor='darkgrey', linewidth= 0.2, alpha = 1) # background nuts regions
        legend_elements: List[Line2D | Patch] = [
            Line2D([0], [0], marker='o', color='w',
                markerfacecolor='darkgreen', markeredgecolor='black',
                markersize=8, label='airports')
        ]            

        # ====== user - specifications ====== #
        # empty graph => just show all nodes
        if graphname == 'empty':
            self._plot_shape(ax, self.airport_context_data.airport_shapefile, color = 'darkgreen', markersize = 75, linewidth = 1, alpha = 0.9)

            if node_layer2 or node_layer3:
                print('empty graph will be visualized instead of any neighborhoods. change graphname.')

            self._plot_shape(ax, shapefile_germany_nuts1, linewidth = 1, edgecolor='black', alpha = 0.4, color = 'none', facecolor = 'none')
            legend_elements.append(Patch(facecolor='white', edgecolor='darkgrey', label='nuts units'))

            title = f'Nested graph: L2: airports, L3: nuts-units'
        
        # specified nodes => show select neighborhood
        else:
            neighborhood = self._get_neighborhood(node_layer2, node_layer3)

            # if the specified node is an airport
            if node_layer2 is not None:
                layer2_neighborhood = self.airport_graph_orchestrator.airport_shapefile[self.airport_graph_orchestrator.airport_shapefile['node_layer2'].isin([node_layer2])]
                layer3_neighborhood = self.airport_graph_orchestrator.nuts_shapefile[self.airport_graph_orchestrator.nuts_shapefile['node_layer3'].isin(neighborhood)]
                title = f' Neighborhood selection layer 2 node: {node_layer2}'
            else:
                layer2_neighborhood = self.airport_graph_orchestrator.airport_shapefile[self.airport_graph_orchestrator.airport_shapefile['node_layer2'].isin(neighborhood)]
                layer3_neighborhood = self.airport_graph_orchestrator.nuts_shapefile[self.airport_graph_orchestrator.nuts_shapefile['node_layer3'].isin([node_layer3])]                
                title = f' Neighborhood selection layer 3 node: {node_layer3}'            

            title    = title

            self._plot_shape(ax,layer3_neighborhood, color = lightred_color, linewidth = 0.2, alpha = 1) 
            self._plot_shape(ax, shapefile_germany_nuts1, linewidth = 1, edgecolor='black', alpha = 0.4, color = 'none', facecolor = 'none')
            self._plot_shape(ax, self.airport_context_data.airport_shapefile, color = 'darkgreen', markersize = 75, linewidth = 1, alpha = 0.6)
            self._plot_shape(ax,layer2_neighborhood, color = 'red', markersize = 75, linewidth = 1, alpha = 0.9)
        
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='black', markersize=8, label=f'airport neighborhood'))
            legend_elements.append(Patch(facecolor=lightred_color, edgecolor='black', label='neighborhood area'))


        ax.set_title(title)
            

        ax.legend(handles=legend_elements)
    
        plt.close()
        return fig

    def _get_neighborhood(self, node_layer2: Optional[int] = None, node_layer3: Optional[int] = None):
        outgoing_L2 = self.graphstructure.edge_index[0]
        ingoing_L1  = self.graphstructure.edge_index[1]

        if node_layer2 is not None:
            neighborhood = [ingoing_L1[idx].item() for idx, id in enumerate(outgoing_L2) if id == node_layer2]        
        else:
            neighborhood = [outgoing_L2[idx].item() for idx, id in enumerate(ingoing_L1) if id == node_layer3]   
        return neighborhood

    def _plot_shape_edge(self,
                         ax: Axes,
                         df: gpd.GeoDataFrame,
                         edgecolor: str = 'black',
                         linewidth: float = 1):
        
        merged_geom = union_all(df.geometry)  # replaces unary_union
        gpd.GeoSeries([merged_geom]).plot(ax=ax, facecolor='none', edgecolor=edgecolor, linewidth = linewidth)

    def _plot_shape(self, 
                    ax: Axes, 
                    df: gpd.GeoDataFrame, 
                    color: str = 'lightgrey',
                    edgecolor: str = 'black', 
                    linewidth: float = 0.5, 
                    alpha: float = 1,
                    **kwargs):
        """Plot geographical shapes."""
        df.plot(color=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, ax=ax, **kwargs)
