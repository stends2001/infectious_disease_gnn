from dataclasses import dataclass, field
from typing import Optional, List, Dict, Literal, Union

from .colentry import ColEntry
from .exceptions import MissingColEntry, MissingTransformationReferral, TransformationParamsAlreadySet 
from .transformation_params import TransformationParams, LogParams, ZScoreParams, MinMaxParams

from ...utils.textformatting import align
from ...utils.types import ColumnType

@dataclass
class ColumnRegistry:
    """
    Stores and manages column metadata for the data-preparation pipeline (EpiDataOrchestrator).
    Provides easy access to columns by type and normalization information.

    Methods
    -------
    - `add_column()`
    - `update_transformation()`

    - `get_entries_names_by_type()`
    - `get_entries_by_type()`
    - `get_transformation_groups()`
    - `get_entry_by_name()`

    - `context_columns`
    - `feature_columns`
    - `target_columns`
    - `pred_columns`
    - `split_columns`
    - `registered_column`

    Attributes
    ----------
    - `_entries`
        the list of columns stored

    See Also
    --------
    - ColEntry

    Examples
    --------
    """
    _entries: List[ColEntry] = field(default_factory=list)
    
    # ========= ADJUSTING / UPDATING COLUMNREGISTRATION ======= #
    def add_column(self, 
                   column_name:             str, 
                   column_type:             ColumnType, 
                   needs_normalization:     bool                            = False,                   
                   transformation_group:    Optional[str]                   = None, 
                   transformation_params:   Optional[TransformationParams]  = None):
        """
        Adds columns to its list of entries (`._entries`)
        All inputs here correspond to those of a ColEntry.
        For further information, please see ColEntry.
        """
        # if transformation is guided by another column, validate that that column 
        # already exists in registry
        if transformation_group is not None and transformation_group != 'self':
            if transformation_group not in self.registered_columns:
                raise MissingTransformationReferral(column_name, transformation_group)

        # Create the column entry
        entry = ColEntry(column_name            = column_name,
                         column_type            = column_type,
                         transformation         = needs_normalization,
                         _transformation_group  = transformation_group,
                         _transformation_params = transformation_params)
        
        # Append to the registry
        self._entries.append(entry)
    
    def update_transformation(self, 
                              column_name: str, 
                              params:      Union[LogParams, ZScoreParams, MinMaxParams]) -> None:
        """
        Adjust the transformation_params of a ColEntry that may or may not already exist.

        Parameters
        ----------
        column_name: str
            name under which column is saved in registry
        params: Union[LogParams, ZScoreParams, MinMaxParams]
            parameters to be saved at ColEntry
        """
        col = self.get_entry_by_name(column_name)

        if col._transformation_params is None:
            col._transformation_params = TransformationParams()

        # match the type of the parameters, and if these have already been set, then throw an error
        match params:

            case LogParams():
            
                if col._transformation_params.log is not None:
                    raise TransformationParamsAlreadySet(column_name, params.__class__.__name__)
            
                col._transformation_params.log = params
                
            case ZScoreParams():

                if col._transformation_params.zscore is not None:
                    raise TransformationParamsAlreadySet(column_name, params.__class__.__name__)
            
                col._transformation_params.zscore = params

            case MinMaxParams():
                if col._transformation_params.minmax is not None:
                    raise TransformationParamsAlreadySet(column_name, params.__class__.__name__)
            
                col._transformation_params.minmax = params                            

            case _:
                raise ValueError(f"Unsupported params type: {type(params)}")
        
    # ========= INTERACTING ======= #        
    def get_entries_names_by_type(self, column_type: str) -> List[str]:
        """Get all column names of a specific type"""
        return [col.column_name for col in self._entries if col.column_type == column_type]
    
    def get_entries_by_type(self, column_type: str) -> List[ColEntry]:
        """Get all ColEntry instances of a specific type"""
        return [col for col in self._entries if col.column_type == column_type]
    
    def get_transformation_groups(self) -> Dict[str, List[str]]:
        """
        Get columns grouped by their normalization reference.
        
        Returns:
        --------
        dict : {normalization_group: [column_names]}
            Keys are the reference columns (or column name itself if ColEntry.transformation_group == 'self')
            Values are lists of columns that share that normalization
        """
        groups: Dict[str, List[str]] = {}
        for entry in self._entries:
            if entry.transformation:
                # Use the column itself as key if normalization_group is 'self'
                if entry._transformation_group:
                    key = entry.transformation_group if entry.transformation_group != 'self' else entry.column_name
                else:
                    key = 'undefined'

                if key not in groups:
                    groups[key] = []

                groups[key].append(entry.column_name)

        return groups
    
    def get_entry_by_name(self, column_name: str) -> ColEntry:
        """Get a specific column entry by name"""
        for col in self._entries:
            if col.column_name == column_name:
                return col
            
        raise MissingColEntry(column_name)

    # ========= PROPERTIES ======= #
    @property
    def context_columns(self) -> List[str]:
        """Get context column names"""
        return self.get_entries_names_by_type('context')
    
    @property
    def feature_columns(self) -> List[str]:
        """Get feature column names"""
        return self.get_entries_names_by_type('feature')
    
    @property
    def target_columns(self) -> List[str]:
        """Get target column names"""
        return self.get_entries_names_by_type('target')
    
    @property 
    def pred_columns(self) -> List[str]:
        """Get pred column names"""
        return self.get_entries_names_by_type('pred')        

    @property
    def split_columns(self) -> List[str]:
        """Get split column names"""
        return self.get_entries_names_by_type('split')
    
    @property 
    def registered_columns(self) -> List[str]:
        return [col.column_name for col in self._entries]        

    # ========= REPRESENTATION ======= #    
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