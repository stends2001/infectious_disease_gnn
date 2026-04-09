from dataclasses import dataclass
from typing import Optional, Dict, Literal
from .issues import InvalidColEntry
from .transformation_params import TransformationParams

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
    # TODO
    _transformation_params: ... 
    """
    column_name:            str
    column_type:            Literal['context','feature','target','pred','split']
    transformation:         bool 
    _transformation_group:  Optional[str]            = None
    _transformation_params: Optional[TransformationParams] = None    
    
    def __post_init__(self):
        
        # lower case input
        self.column_name = self.column_name.lower()
        self.column_type = self.column_type.lower()  # type: ignore[assignment]

        if self._transformation_group:
            self._transformation_group = self._transformation_group.lower()

        # validate against impossible states
        self._validate_input()

    def _validate_input(self):
        """to be run post init: validate whether combination of input makes sense or not"""
        allowed_types = ['context','feature','target','pred','split']
        
        if self.column_type not in allowed_types:
            raise InvalidColEntry(self.column_name, f'got unsupported value for column_type ({self.column_type}). Supported values are: {allowed_types}') 

    @property
    def transformation_group(self) -> str:
        if self._transformation_group:
            return self._transformation_group
        else:
            raise ValueError(f'attempted to access transformation group for {self.column_name}')

    @property
    def transformation_params(self) -> Dict[str,Dict[str,dict]]:
        if self._transformation_params:
            return self._transformation_params
        else:
            raise ValueError(f'attempted to access transformation_params for {self.column_name}')        
    
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
