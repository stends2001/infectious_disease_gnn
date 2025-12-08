import torch
import os
from dataclasses import dataclass
from typing import List, Union, Dict, Literal
import json

from .graphstructures import GraphStructure, DynamicGraphStructure
from .graphstats import StaticGraphStats, DynamicGraphStats

from ..utils.textformatting import checkmark, warning_emoji, error_emoji, align

class InvalidGraphEntryName(Exception):
    pass


@dataclass 
class GraphEntry:
    """
    Single entry to the GraphRegistry with 

    Parameters
    ---------
    structure:  GraphStructure

    summary:    GraphStatistics
    
    config:     GraphConfig
    """

    structure: Union[GraphStructure, DynamicGraphStructure]
    summary:   Union[StaticGraphStats, DynamicGraphStats]
    config:    Dict[str,str]

    def __repr__(self) -> str:
        representation = f'<GraphEntry(structure, summary, config)>'
        return representation        

    def _get_summary(self, type: Literal['small','large']) -> str:
        """returns str of graph summary"""
        if type == 'large':
            return print(self.summary)

        elif type == 'small':
            return print(self.summary._get_small_summary())

class GraphRegistry:
    """
    Registry of GraphEntry objects

    Attributes
    ----------
    registry: Dict[str, GraphEntry]
        graphnames : GraphEntry object
    
    Methods
    -------
    add_entry       ->  None
    rename_entry    ->  None
    get_entry       ->  GraphEntry
    """
    def __init__(self, graph_dir: str):
        self.registry: Dict[str, GraphEntry]    = {}
        self.alignment_width                    = 19
        self.graph_dir                          = graph_dir

    def add_entry(self, graphname: str, entry: GraphEntry) -> None:
        """Add structure to .registry under graphname"""
        if self.check_entry(graphname):
            print(align(f'{warning_emoji} warning', f'{graphname} already exists, please rename the already existing entry. New entry wasn\'t registered', width=self.alignment_width, newline=False))            
        else:
            self.registry[graphname] = entry
            print(align(f'{checkmark} Graph registered', f'{graphname} successfully registered', width=self.alignment_width, newline=False))                        
        
    def rename_entry(self, current_graphname: str, new_graphname: str) -> None:
        """rename entry from current_graphname to new_graphname; current_graphname is removed"""        
        if self.check_entry(new_graphname):
            print(align(f'{warning_emoji} warning', f'{new_graphname} already exists, please rename the already existing entry. New entry wasn\'t registered', width=self.alignment_width, newline=False))    
        else:
            self.add_entry(new_graphname,self.registry[current_graphname])
            self.remove_entry(current_graphname)           

    def remove_entry(self, graphname: str) -> None:
        del self.registry[graphname]
        print(align(f'{checkmark} Graph removed', f'{graphname} has been deregistered', width=self.alignment_width, newline=False))        

    def get_entry(self, graphname: str) -> 'GraphEntry':
        """return GraphStructure from graphname"""
        if not self.check_entry(graphname):
            print(align(f'{warning_emoji} Graph not found', f'{graphname} wasn\'t found', width=self.alignment_width, newline=False))                  
            registered_entries = ', '.join(self.return_entrynames())
            raise ValueError(f"the following graphs are registered:\n{registered_entries}")
        
        else:
            return self.registry[graphname]

    def check_entry(self, graphname: str) -> bool:
        """return boolean reflecting whether or not graphname is registered"""
        if graphname in self.registry.keys():
            return True
        else:
            return False
        
    def return_entrynames(self) -> List[str]:
        """returns a list of entrynames"""
        return list(self.registry.keys())
        
    def get_graph_stats(self, graphname: str, type: Literal['small','large']) -> str:
        """Returns a string representation of the graph statistics, either small or extensive"""
        
        return self.get_entry(graphname)._get_summary(type)

    def save_graphentry(self, graphname: str) -> None:
        """
        Save graphentry

        Seperately, the following objects are saved:
        - edge_index
        - edge_weight
        - graphconfig
        """
        graph_entry = self.get_entry(graphname)
        
        directory   = os.path.join(self.graph_dir, graphname)
        
        if os.path.exists(directory):
            raise FileExistsError(f'{error_emoji} GraphEntry Not Saved: {graphname} directory already exists')

        os.makedirs(directory, exist_ok=True)

        if graph_entry is not None:
            
            edge_index  = graph_entry.structure.edge_index
            edge_weight = graph_entry.structure.edge_weight

            torch.save(edge_index, os.path.join(directory, f'{graphname}_edge_index.pt'))

            if edge_weight is not None:
                torch.save(edge_weight, os.path.join(directory, f'{graphname}_edge_weight.pt'))       

        # Save config as JSON
        config_path = os.path.join(directory, f'{graphname}_config.json')

        # copy to make sure the original config isn't adjusted
        config_copy = graph_entry.config.copy()

        config_copy['graphname'] = graphname
        
        if config_copy.get("scaling_method") is None:
            config_copy.pop("scaling_method", None)
        
        with open(config_path, 'w') as f:
            json.dump(config_copy, f, indent=2) 
        
        print(align(f'{checkmark} GraphEntry Saved', f'{graphname} has been saved', width=self.alignment_width, newline=False))    

    def _validate_graphentry_name(self, name: str) -> None:
        """test whether name is suitable or not to be saved into a directory"""

        characters_allowed =  set("abcdefghijklmnopqrstuvwxyz0123456789_-")

        if any(ch not in characters_allowed for ch in name):
            raise InvalidGraphEntryName(f"Invalid GraphEntry name! Rename before saving GraphEntry. Accepter characters are:\n{characters_allowed}")

    def __repr__(self) -> str:
        registered_entries = ', '.join(self.return_entrynames())
        return f'<GraphRegistry({registered_entries})'
