from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from .issues import InvalidColEntry, MissingColEntry, MissingTransformationReferral
from ...utils.textformatting import align

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
    _transformation_params: Optional[Dict[str,Any]]
        transformation parameters. Only possible when self.transformation is True, and when self.transformation_group == 'self'
        NOTE: to be accesed with attribute `self.transformation_params`. Allows for the checking of absence: error is thrown.
    """
    column_name:            str
    column_type:            Literal['context','feature','target','pred','split']
    transformation:         bool 
    _transformation_group:  Optional[str]            = None
    _transformation_params: Optional[Dict[str, Any]] = None    
    
    def __post_init__(self):
        # lower case input
        self.column_name = self.column_name.lower()
        self.column_type = self.column_type.lower()  # type: ignore[assignment]
        if self._transformation_group:
            self._transformation_group = self._transformation_group.lower()

        # validate against impossible states
        self._validate_input()

    def _validate_input(self) -> None:
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
    def transformation_params(self) -> Dict[str, Any]:
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

@dataclass
class ColumnRegistration:
    """
    Stores and manages column metadata for the data pipeline.
    Provides easy access to columns by type and normalization information.
    """
    columns: List[ColEntry] = field(default_factory=list)
    
    # ========= ADJUSTING / UPDATING COLUMNREGISTRATION ======= #
    def add_column(self, 
                   column_name:             str, 
                   column_type:             Literal['context','feature','target','pred','split'], 
                   needs_normalization:     bool = False,                   
                   transformation_group:    Optional[str] = None, 
                   transformation_params:   Optional[dict] = None):
        """Add a column to the registry. Please see ColEntry for more information"""

        # if transformation is guided by another column, check whether that column 
        # already exists in registry
        if transformation_group and transformation_group != 'self' and transformation_group not in self.registered_columns:
                raise MissingTransformationReferral(column_name, transformation_group)

        # Create the column entry
        entry = ColEntry(column_name            = column_name,
                         column_type            = column_type,
                         transformation         = needs_normalization,
                         _transformation_group  = transformation_group,
                         _transformation_params = transformation_params)
        
        # Append to the registry
        self.columns.append(entry)
    
    def update_transformation(self, column_name: str, transformation_params: dict):
        """Update transformation info for a column"""
        col = self.get_by_name(column_name)
        # if transformation params are not yet initiated for this col, then insert the dict
        if col._transformation_params is None:
            col._transformation_params = transformation_params
        # if transformation params are initiated for this col, then simply update the dict            
        else:
            col._transformation_params.update(transformation_params)
        
    # ========= INTERACTING ======= #        
    def get_by_type(self, column_type: str) -> List[str]:
        """Get all column names of a specific type"""
        return [col.column_name for col in self.columns if col.column_type == column_type]
    
    def get_entries_by_type(self, column_type: str) -> List[ColEntry]:
        """Get all column entries of a specific type"""
        return [col for col in self.columns if col.column_type == column_type]
    
    def get_transformation_groups(self) -> Dict[str, List[str]]:
        """
        Get columns grouped by their normalization reference.
        
        Returns:
        --------
        dict : {normalization_group: [column_names]}
            Keys are the reference columns (or column name itself if ColEntry.transformation_group == 'self')
            Values are lists of columns that share that normalization
        """
        groups = {}
        for col in self.columns:
            if col.transformation:
                # Use the column itself as key if normalization_group is 'self'
                key = col.transformation_group if col.transformation_group != 'self' else col.column_name
                if key not in groups:
                    groups[key] = []
                groups[key].append(col.column_name)
        return groups
    
    def get_by_name(self, column_name: str) -> ColEntry:
        """Get a specific column entry by name"""
        for col in self.columns:
            if col.column_name == column_name:
                return col
        raise MissingColEntry(column_name)

    # ========= PROPERTIES ======= #
    @property
    def context_columns(self) -> List[str]:
        """Get context column names"""
        return self.get_by_type('context')
    
    @property
    def feature_columns(self) -> List[str]:
        """Get feature column names"""
        return self.get_by_type('feature')
    
    @property
    def target_columns(self) -> List[str]:
        """Get target column names"""
        return self.get_by_type('target')
    
    @property 
    def pred_columns(self) -> List[str]:
        """Get pred column names"""
        return self.get_by_type('pred')        

    @property
    def split_columns(self) -> List[str]:
        """Get split column names"""
        return self.get_by_type('split')
    
    @property 
    def registered_columns(self) -> List[str]:
        return [col.column_name for col in self.columns]        

    # ========= OBJECT REPRESENTATION ======= #    
    def __repr__(self) -> str:
        type_counts = {
            'context'   : len(self.context_columns),
            'feature'   : len(self.feature_columns),            
            'target'    : len(self.target_columns),
            'pred'      : len(self.pred_columns),
            'split'     : len(self.split_columns)
        }

        representation = (            
            f"<{self.__class__.__name__}("+
            f"registered_columns = {sum(type_counts.values())}, "+
            ", ".join([f"{key} = {value}" for key,value in type_counts.items()]) +
            ")>"      
            )
        
        return representation
    
    def __str__(self) -> str:
        all_keys        = ['context','feature','split','target','pred']
        width           = max(len(k) for k in all_keys)
        indent          = 4
        
        lines = [f"<{self.__class__.__name__}("]     
        lines.append(align('context', self.context_columns, width, indent))
        lines.append(align('feature', self.feature_columns, width, indent))
        lines.append(align('target',  self.target_columns,  width, indent))
        lines.append(align('pred',    self.pred_columns,    width, indent))
        lines.append(align('split',   self.split_columns,   width, indent))        
        
        lines.append(")>")
        representation = '\n'.join(lines)
        return representation