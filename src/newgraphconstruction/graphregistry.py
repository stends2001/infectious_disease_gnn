import torch
import os
from dataclasses import dataclass
from typing import List, Union, Dict, Literal
import json

from .containers import GraphStructure, DynamicGraphStructure
from .graphstats import StaticGraphStats, DynamicGraphStats
from .graphconfig import StaticGraphConfig, DynamicGraphConfig

from ..utils.textformatting import checkmark, warning_emoji, error_emoji, align

class InvalidGraphEntryName(Exception):
    pass

@dataclass 
class GraphEntry:
    """
    Single entry to the GraphRegistry with 

    Parameters
    ---------
    structure:  Union[GraphStructure, DynamicGraphStructure]

    summary:    Union[StaticGraphStats, DynamicGraphStats]
    
    config:     Union[StaticGraphConfig, DynamicGraphConfig]

    mode:       Literal['static','dynamic']
    """

    structure: Union[GraphStructure, DynamicGraphStructure]
    summary:   Union[StaticGraphStats, DynamicGraphStats]
    config:    Union[StaticGraphConfig, DynamicGraphConfig]
    mode:      Literal['static','dynamic']

    def _get_summary(self, type: Literal['small','large']) -> str:
        """returns str of graph summary"""

        if type == 'large':
            return print(self.summary)

        elif type == 'small':
            return print(self.summary._get_small_summary())

class GraphRegistry:
    """
    Registry of GraphEntry objects into self.registry: Dict[str, GraphEntry]

    Parameters
    ----------
    graph_dir: str
        directory in which to save graphstructures from the registry

    Methods
    -------
    - add_entry
        adds an entry to the current registry
    - get_entry
        retrieves an entry from the current registry      
    - rename_entry
        renames an entry from current_graphname to new_graphname
    - remove_entry
        removes an entry from the current registry
    - save_entry
        saves an entry from the current registry
    """
    def __init__(self, graph_dir: str):
        self.registry: Dict[str, GraphEntry]= {}
        # saving graphs into:
        self.graph_dir                      = graph_dir

        # for printing
        self._print_alignment_width         = 19

    def add_entry(self, graphname: str, entry: GraphEntry) -> None:
        """adds entry to registry under graphname"""
        if self._check_entry(graphname):
            print(align(f'{warning_emoji} warning', f'{graphname} already exists, please rename the already existing entry. New entry wasn\'t registered', width=self._print_alignment_width, newline=False))            

        else:
            self.registry[graphname] = entry
            print(align(f'{checkmark} Graph registered', f'{graphname} successfully registered', width=self._print_alignment_width, newline=False))                        
        
    def get_entry(self, graphname: str) -> 'GraphEntry':
        """returns entry from graphname"""
        if not self._check_entry(graphname):
            print(align(f'{warning_emoji} Graph not found', f'{graphname} wasn\'t found', width=self._print_alignment_width, newline=False))                  
            registered_entries = ', '.join(self._return_entrynames())
            raise ValueError(f"the following graphs are registered:\n{registered_entries}")
        
        else:
            return self.registry[graphname]

    def rename_entry(self, current_graphname: str, new_graphname: str) -> None:
        """renames entry from current_graphname to new_graphname; current_graphname is removed"""        
        if self._check_entry(new_graphname):
            print(align(f'{warning_emoji} warning', f'{new_graphname} already exists, please rename the already existing entry. New entry wasn\'t registered', width=self._print_alignment_width, newline=False))    
        else:
            self.add_entry(new_graphname,self.registry[current_graphname])
            self.remove_entry(current_graphname)           

    def remove_entry(self, graphname: str) -> None:
        del self.registry[graphname]
        print(align(f'{checkmark} Graph removed', f'{graphname} has been deregistered', width=self._print_alignment_width, newline=False))        

    def save_graphentry(self, graphname: str) -> None:
        """
        Save graphentry (static or dynamic)

        For static graphs:
        in self.graph_dir / {graphname}, graph is saved into:
        - edge_index.pt
        - edge_weight.pt
        - config.json

        For static graphs:
        in self.graph_dir / {graphname}, graph is saved into:
            for timestamp:
            - edge_index_{timestamp}.pt
            - edge_weight_{timestamp}.pt
        - config.json
        """
        graph_entry = self.get_entry(graphname)
        directory   = os.path.join(self.graph_dir, graphname)
        
        if os.path.exists(directory):
            raise FileExistsError(f'{error_emoji} GraphEntry Not Saved: {graphname} directory already exists')

        os.makedirs(directory, exist_ok=True)

        if graph_entry.config.mode == 'static':
            self._save_static_graph(graph_entry, graphname, directory)
        
        elif graph_entry.config.mode == 'dynamic':
            self._save_dynamic_graph(graph_entry, graphname, directory)
        
        print(align(f'{checkmark} GraphEntry Saved', f'{graphname} has been saved', 
                    width=self._print_alignment_width, newline=False))

    def _save_static_graph(self, graph_entry: GraphEntry, graphname: str, directory: str) -> None:
        """Save static graph structure"""
        edge_index = graph_entry.structure.edge_index
        edge_weight = graph_entry.structure.edge_weight

        torch.save(edge_index, os.path.join(directory, f'{graphname}_edge_index.pt'))

        if edge_weight is not None:
            torch.save(edge_weight, os.path.join(directory, f'{graphname}_edge_weight.pt'))

        # Save config as JSON
        config_path = os.path.join(directory, f'{graphname}_config.json')
        config_copy = graph_entry.config.copy()

        # Remove None values
        config_copy = {k: v for k, v in config_copy.items() if v is not None}
    
        with open(config_path, 'w') as f:
            json.dump(config_copy, f, indent=2)

    def _save_dynamic_graph(self, graph_entry: GraphEntry, graphname: str, directory: str) -> None:
        """Save dynamic graph structure with timestamp subdirectories"""
        dynamic_structure   = graph_entry.structure
        timestamps          = dynamic_structure.get_timestamps_as_datetime()
        
        # Save each snapshot in its own subdirectory
        for t_idx, timestamp in enumerate(timestamps):
            # Create timestamp string for folder name (e.g., "2020" or "2020-01-01")
            if hasattr(timestamp, 'year') and hasattr(timestamp, 'month'):
                # Full datetime
                timestamp_str = timestamp.strftime('%Y-%m-%d')
            else:
                # Just year
                timestamp_str = str(timestamp.year)
            
            # Get snapshot data
            edge_index = dynamic_structure.edge_indices[t_idx]
            edge_weight = dynamic_structure.edge_weights[t_idx]
            
            # Save tensors
            torch.save(edge_index, os.path.join(directory, f'{timestamp_str}_edge_index.pt'))
            if edge_weight is not None:
                torch.save(edge_weight, os.path.join(directory, f'{timestamp_str}_edge_weight.pt'))
        
        # Save config at top level
        config_path = os.path.join(directory, f'{graphname}_config.json')
        config_copy = graph_entry.config.copy()
        
        # Remove None values
        config_copy = {k: v for k, v in config_copy.items() if v is not None}
        
        with open(config_path, 'w') as f:
            json.dump(config_copy, f, indent=2)

    def _check_entry(self, graphname: str) -> bool:
        """return boolean reflecting whether or not graphname is registered"""
        if graphname in self.registry.keys():
            return True
        else:
            return False
        
    def _return_entrynames(self) -> List[str]:
        """returns a list of entrynames"""
        return list(self.registry.keys())
        
    def _validate_graphentry_name(self, name: str) -> None:
        """test whether name is suitable or not to be saved into a directory"""

        characters_allowed =  set("abcdefghijklmnopqrstuvwxyz0123456789_-")

        if any(ch not in characters_allowed for ch in name):
            raise InvalidGraphEntryName(f"Invalid GraphEntry name! Rename before saving GraphEntry. Accepter characters are:\n{characters_allowed}")

    def __repr__(self) -> str:
        registered_entries = ', '.join(self._return_entrynames())
        return f'<GraphRegistry(N = {len(self._return_entrynames())} graphs:\n\t{registered_entries})'