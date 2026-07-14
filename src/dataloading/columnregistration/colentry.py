from dataclasses import dataclass
from typing import Optional, Literal
from .transformation_params import TransformationParams

class ColEntryMissingAttribute(Exception):
    def __init__(self, entry_name: str, attribute_name: str):
        msg = f"Accession attempt was made on ColEntry {entry_name} for attribute {attribute_name} which is unavailable."
        super().__init__(msg)

class InvalidColEntry(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)    

@dataclass
class ColEntry:
    """
    Entry for a single column into the column registry

    Parameters:
    ----------
    column_name: str
        name of column, as appearing in the final df of the EpiDataOrchestrator
    column_type: Literal['context','feature','target','pred','split']
        type of column
    transformation: bool
        whether or not column requires a transformation
    transformation_group: Optional[str]
        what directs the transformation of this column. There's three options:
        - transformation_group == 'self'    => individual transformation (based on parameters from itself)
        - transformation_group == {column_name} of another ColEntry
        - transformation_group == None      => only possible when self.transformation == False
    """
    column_name:            str
    column_type:            Literal['context','feature','target','pred','split']
    transformation:         bool 
    _transformation_group:  Optional[str]                   = None
    _transformation_params: Optional[TransformationParams]  = None    
    
    def __post_init__(self):
        # lower case input
        self.column_name = self.column_name.lower()
        self._validate_input()

        # ensure lower case in transformation group
        if self._transformation_group:
            self._transformation_group = self._transformation_group.lower()

    def _validate_input(self) -> None:
        """validate against invalid states"""
        if not self.transformation:
            if self._transformation_group is not None:
                raise InvalidColEntry(f"transformation is False, but _transformation_group is given for {self.column_name}.")
            if self._transformation_params is not None:
                raise InvalidColEntry(f"transformation is False, but _transformation_params is given for {self.column_name}.")                

    @property
    def transformation_group(self) -> str:
        if self._transformation_group:
            return self._transformation_group
        else:
            raise ColEntryMissingAttribute(self.column_name, "transformation_group")

    @property
    def transformation_params(self) -> TransformationParams:
        if self._transformation_params:
            return self._transformation_params
        else:
            raise ColEntryMissingAttribute(self.column_name, "transformation_params")     
    
    def __repr__(self) -> str:
        representation = (
            f"<{self.__class__.__name__}("+
            f"column_name = {self.column_name}, " +
            f"column_type = {self.column_type}, "
        )
        
        if self.transformation:
            representation += f", transformation = {self.transformation}"
            representation += f", transformation_group = {self._transformation_group}"            
            representation += f", transformation_params = {self._transformation_params}"                 
        
        representation += ")>"
        return representation
