from dataclasses import dataclass
from typing import Union, List

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
    Statistics for dynamic graphs - stores per-snapshot stats
    """
    timestamps: List[str]  # ISO format strings for JSON serialization
    num_edges: List[int]
    num_nodes: List[int]
    edge_density: List[float]
    edge_weight_mean: List[float]
    edge_weight_min: List[float]
    edge_weight_max: List[float]
    
    def _get_small_summary(self) -> str:
        """Return condensed summary across all timestamps"""
        return (
            f"Dynamic Graph: T={len(self.timestamps)}, "
            f"Edges={min(self.num_edges)}-{max(self.num_edges)}, "
            f"Nodes={self.num_nodes[0]}"
        )
    
    def __repr__(self) -> str:
        lines = ["Dynamic Graph Statistics:"]
        lines.append(f"  Time range: {self.timestamps[0]} to {self.timestamps[-1]}")
        lines.append(f"  Snapshots: {len(self.timestamps)}")
        lines.append(f"  Nodes (constant): {self.num_nodes[0]}")
        lines.append(f"  Edges: {min(self.num_edges)} - {max(self.num_edges)} (min-max)")
        lines.append(f"  Edge Density: {min(self.edge_density):.4f} - {max(self.edge_density):.4f}")
        lines.append(f"  Edge Weight Mean: {min(self.edge_weight_mean):.4f} - {max(self.edge_weight_mean):.4f}")
        return "\n".join(lines)