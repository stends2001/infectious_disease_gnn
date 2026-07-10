from typing import Union, List, Dict, Optional
from pathlib import Path 

from .updates import * 
from ..graphobjects import GraphObject
from ...utils import PathNotFound


class GraphRegistry:
    """
    Registry of GraphObject instances: stores, saves and loads them.

    Parameters
    ----------
    graph_partition_dir: str
        partition-directory in which to save graphstructures from the registry.
        That is, the path in which different graph representations of the same 
        thing are stored (same country/level combination). These structures
        have a shared `tokenization_map`.

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

    See Also
    --------
    - GraphObject
    - GraphConfig
    - GraphStrucure

    Downstream Use
    ---------------
    GraphManager
        Each instance of GraphManager has a registry as attribute.
    """
    def __init__(self, graph_partition_dir: Union[str,Path]):
        self._registry: Dict[str, GraphObject]= {}

        graph_partition_dir = Path(graph_partition_dir)
        
        if not graph_partition_dir.exists():
            raise PathNotFound(graph_partition_dir)

        self.graph_partition_dir = graph_partition_dir

    def add_entry(self, graphname: str, entry: GraphObject) -> None:
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

    def rename_entry(self, current_graphname: str, new_graphname: str) -> None:
        """renames entry from current_graphname to new_graphname; current_graphname is removed"""        
        if self._check_presence(current_graphname):
            print(GraphEntryDoesntExist(current_graphname, self.entry_names))
            return
        
        if self._check_presence(new_graphname):
            print(GraphEntryAlreadyExists(new_graphname))
            return 

        self.add_entry(new_graphname, self._registry[current_graphname])
        self.remove_entry(current_graphname)          

    def remove_entry(self, graphname) -> None:
        """removes entry from registry"""
        if not self._check_presence(graphname):
            print(GraphEntryDoesntExist(graphname, self.entry_names))
            return 
        
        del self._registry[graphname]

    def save_entry(self, graphname: str) -> None:
        """save a graph entry from registry to `self.graph_partition_dir` / `graphname` """
        entry_to_save = self.get_entry(graphname)
        
        if entry_to_save is None:
            return

        self._validate_graphentry_name(graphname)
        
        # save into partition (country / level) / graphname
        entry_to_save.save(self.graph_partition_dir / graphname)

        print(GraphStructureSaved(graphname))

    def save_all_entries(self, confirm: bool = False) -> None:
        if not confirm:
            raise ValueError(f'when saving all graph structures, please confirm by argument')
        
        for entryname in self.entry_names:
            self.save_entry(entryname)

    def load_entry(self, graphname: str) -> None:
        """load a graph entry from directory"""       

        # load from partition (country / level) / graphname        
        graph_object = GraphObject.load(self.graph_partition_dir / graphname)

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