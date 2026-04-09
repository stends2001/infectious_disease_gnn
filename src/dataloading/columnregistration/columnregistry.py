from dataclasses import dataclass, field
from typing import Optional, List, Dict, Literal

from .colentry import ColEntry
from .transformation_params import TransformationParams

from .issues import MissingColEntry, MissingTransformationReferral
from ...utils.textformatting import align

@dataclass
class ColumnRegistry:
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
                   transformation_params:   Optional[TransformationParams] = None):
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
    
    def update_transformation(self, column_name: str, params: TransformationParams) -> None:
        col = self.get_by_name(column_name)
        if col._transformation_params is None:
            col._transformation_params = params
        else:
            # merge: only overwrite fields that are explicitly set in the new params
            existing = col._transformation_params
            col._transformation_params = TransformationParams(
                log    = params.log    if params.log    is not None else existing.log,
                zscore = params.zscore if params.zscore is not None else existing.zscore,
                minmax = params.minmax if params.minmax is not None else existing.minmax,
            )
        
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