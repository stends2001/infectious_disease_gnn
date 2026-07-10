from typing import Union, List, Dict, Optional
from pathlib import Path 

from .updates import * 
from ..graphobjects import GraphObject
from ...utils import PathNotFound


class GraphRegistry:
    """
    Registry of GraphObject objects

    Parameters
    ----------
    graph_dir: str
        directory in which to save graphstructures from the registry

    Methods
    -------
    - `add_entry()`
    - `get_entry()`
    - `rename_entry()`
    - `remove_entry()`
    - `save_entry()`
    - `save_all_entries()`
    - `load_entry()`

    Attributes
    ----------
    - `entry_names`
    """
    def __init__(self, graph_dir: Union[str,Path]):
        self._registry: Dict[str, GraphObject]= {}
        
        graph_path = Path(graph_dir)
        
        # saving graphs into:
        if not graph_path.exists():
            raise PathNotFound(graph_dir)

        self.graph_dir                      = graph_path

    def add_entry(self, graphname: str, entry: GraphObject):
        """adds entry to registry under graphname"""
        if self._check_presence(graphname):
           print(GraphEntryAlreadyExists(graphname))

        else:
            self._registry[graphname] = entry
            print(GraphEntryRegistered(graphname))
        
    def get_entry(self, graphname: str) -> Optional[GraphObject]:
        """returns entry from graphname"""
        if not self._check_presence(graphname):
            print(GraphEntryDoesntExist(graphname, self.entry_names))
        
        else:
            return self._registry[graphname]

    def rename_entry(self, current_graphname: str, new_graphname: str):
        """renames entry from current_graphname to new_graphname; current_graphname is removed"""        
        if self._check_presence(new_graphname):
            print(GraphEntryAlreadyExists(new_graphname))
        else:
            self.add_entry(new_graphname, self._registry[current_graphname])
            self.remove_entry(current_graphname)          

    def remove_entry(self, graphname):
        if not self._check_presence(graphname):
            print(GraphEntryDoesntExist(graphname, self.entry_names))

        del self._registry[graphname]

    def save_entry(self, graphname: str) -> None:
        """save a graph entry"""
        entry_to_save = self.get_entry(graphname)
        
        if entry_to_save is None:
            return

        self._validate_graphentry_name(graphname)
        
        entry_to_save.save(self.graph_dir)

        print(GraphStructureSaved(graphname))

    def save_all_entries(self, confirm: bool = False):
        if not confirm:
            raise ValueError(f'when saving all graph structures, please confirm by argument')
        
        for entryname in self.entry_names:
            self.save_entry(entryname)

    def load_entry(self, graphname: str):
        """load a graph entry from file"""       
        graph_object = GraphObject.load(self.graph_dir / graphname)

        self.add_entry(graphname, graph_object)

    @property 
    def entry_names(self) -> List[str]:
        return list(self._registry.keys())
    
    # ============ HIDDEN METHODS ================ #

    def _check_presence(self, graphname: str) -> bool:
        return graphname in self._registry.keys()

    def _validate_graphentry_name(self, graphname: str) -> None:
        """test whether name is suitable or not to be saved into a directory"""

        characters_allowed =  set("abcdefghijklmnopqrstuvwxyz0123456789_-")

        if any(ch not in characters_allowed for ch in graphname):
            print(InvalidGraphName(graphname, characters_allowed))

    def __repr__(self) -> str:
        registered_entries = ', '.join(self.entry_names)
        return f'<{self.__class__.__name__}(N = {len(self.entry_names)} graphs:\n\t{registered_entries})>'