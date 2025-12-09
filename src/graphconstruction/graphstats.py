from dataclasses import dataclass
from typing import Union

from ..utils.textformatting import align

@dataclass 
class StaticGraphStats:
 
    num_nodes:          int
    num_edges:          int
    edge_density:       float
    num_isolated_nodes: int

    edge_weight_mean:   float
    edge_weight_min:    float
    edge_weight_max:    float

    out_degree_mean:    Union[int,float]
    out_degree_max:     int
    out_degree_min:     int    
   
    in_degree_mean:     Union[int,float]
    in_degree_max:      int
    in_degree_min:      int    

    def __repr__(self) -> str:

        largest_key = len("num_isolated_nodes")

        statement = self._get_small_summary() +"\n"
                        
        statement += align('edge_weight_mean',   self.edge_weight_mean,  width=largest_key + 2, newline=True)   
        statement += align('edge_weight_min',    self.edge_weight_min,   width=largest_key + 2, newline=True)                   
        statement += align('edge_weight_max',    self.edge_weight_max,   width=largest_key + 2, newline=True)    

        statement += "\n"      
        statement += align('out_degree_mean',   self.out_degree_mean,  width=largest_key + 2, newline=True)   
        statement += align('out_degree_max',    self.out_degree_max,   width=largest_key + 2, newline=True)                   
        statement += align('out_degree_min',    self.out_degree_min,   width=largest_key + 2, newline=True)      

        statement += "\n"      
        statement += align('in_degree_mean',   self.in_degree_mean,  width=largest_key + 2, newline=True)   
        statement += align('in_degree_max',    self.in_degree_max,   width=largest_key + 2, newline=True)                   
        statement += align('in_degree_min',    self.in_degree_min,   width=largest_key + 2, newline=True)                

        return statement      

    def _get_small_summary(self) -> str:

        largest_key = len("num_isolated_nodes")

        statement = ""

        statement += align('num_nodes',          self.num_nodes,            width=largest_key + 2, newline=True)
        statement += align('num_edges',          self.num_edges,            width=largest_key + 2, newline=True)
        statement += align('edge_density',       self.edge_density,         width=largest_key + 2, newline=True)
        statement += align('num_isolated_nodes', self.num_isolated_nodes,   width=largest_key + 2, newline=True)                                 

        return statement           

@dataclass 
class DynamicGraphStats:

    """ 
    Statistics for a dyanmic graph structure

    for now very limited:
    
    Parameters
    ----------
    num_graphs: int
        number of graphs inside
    num_nodes: int
        number of nodes per graph structure

    #TODO: as the parameters / summary for now are very limited, small and large summary are the same.
    """

    num_graphs:     int
    num_nodes:      int 

    def __repr__(self) -> str:

        largest_key = len("num_graphs")

        statement = self._get_small_summary() +"\n"      

        return statement      

    def _get_small_summary(self) -> str:

        largest_key = len("num_graphs")

        statement = ""

        statement += align('num_graphs',         self.num_graphs,           width=largest_key + 2, newline=True)
        statement += align('num_nodes',          self.num_nodes,            width=largest_key + 2, newline=True)
        return statement     
