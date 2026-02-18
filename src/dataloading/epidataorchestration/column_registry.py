from dataclasses import dataclass, field
from typing import Optional, List, Dict, Literal, Union

# =========== EXCEPTIONS ==========
class ColEntryError(Exception):
    """Base class for all column entry related errors."""
    def __init__(self, entryname: str, message: str):
        self.entryname = entryname
        self.message = message
        super().__init__(f"{self.__class__.__name__} - {self.message}")

class ColEntryMissingError(ColEntryError):
    """Raised when a column entry is missing."""
    def __init__(self, entryname: str):
        message = f"Column Registration entry '{entryname}' is missing."
        super().__init__(entryname, message)

class ColEntryMissingTransformationError(ColEntryError):
    """Raised when a transformation is invalid."""
    def __init__(self, entryname: str):
        message = f"Column Registration entry '{entryname}' has transformation_group None (independent) but no transformation attribute was found."
        super().__init__(entryname, message)

class ColEntryMissingTransformationReferralError(ColEntryError):
    """Raised when a transformation referral is invalid."""
    def __init__(self, entryname: str, referral: str):
        message = f"Column Registration entry '{entryname}' has transformation_group {referral} for which no transformation attribute was found."
        super().__init__(entryname, message)

@dataclass
class ColEntry:
    """
    Entry for a single column into the column registry
    """
    column_name:            str
    column_type:            str
    transformation_group:   Union[Literal[None, 'NA'], str] = 'NA'
    transformation:         Optional[dict] = None    
    
    def has_transformation(self) -> bool:
        """Check if this column should be normalized"""
        return self.transformation_group != 'NA' and self.transformation_group is not None
    
    def __repr__(self):
        norm_info = f", norm_group={self.transformation_group}"
        return f"<ColEntry({self.column_name}, type={self.column_type}{norm_info})>"

@dataclass
class ColumnRegistration:
    """
    Stores and manages column metadata for the data pipeline.
    Provides easy access to columns by type and normalization information.
    """
    columns: List[ColEntry] = field(default_factory=list)
    
    def add_column(self, 
                   column_name: str, 
                   column_type: str, 
                   transformation_group: Optional[Union[Literal[None, 'NA'], str]] = None, 
                   needs_normalization: bool = False,
                   transformation: Optional[dict] = None):
        """Add a column to the registry"""
        
        if needs_normalization:
            transformation_group = transformation_group  # use the provided normalization group
        else:
            transformation_group = 'NA'  # normalization is not needed, mark as 'NA'
        
        # Create the column entry
        entry = ColEntry(
            column_name=column_name,
            column_type=column_type,
            transformation_group=transformation_group,
            transformation=transformation
        )
        
        # Append to the registry
        self.columns.append(entry)
    
    def get_by_type(self, column_type: str) -> List[str]:
        """Get all column names of a specific type"""
        return [col.column_name for col in self.columns if col.column_type == column_type]
    
    def get_entries_by_type(self, column_type: str) -> List[ColEntry]:
        """Get all column entries of a specific type"""
        return [col for col in self.columns if col.column_type == column_type]
    
    @property
    def context_columns(self) -> List[str]:
        """Get context column names"""
        return self.get_by_type('context')
    
    @property
    def target_columns(self) -> List[str]:
        """Get target column names"""
        return self.get_by_type('target')
    
    @property 
    def pred_columns(self) -> List[str]:
        """Get pred column names"""
        return self.get_by_type('pred')        

    @property
    def feature_columns(self) -> List[str]:
        """Get feature column names"""
        return self.get_by_type('feature')
    
    @property
    def split_columns(self) -> List[str]:
        """Get split column names"""
        return self.get_by_type('split')
    
    @property 
    def registered_columns(self) -> List[str]:
        return [col.column_name for col in self.columns]        

    def get_transformation_groups(self) -> Dict[str, List[str]]:
        """
        Get columns grouped by their normalization reference.
        
        Returns:
        --------
        dict : {normalization_group: [column_names]}
            Keys are the reference columns (or column name itself if None)
            Values are lists of columns that share that normalization
        """
        groups = {}
        for col in self.columns:
            if col.transformation_group != 'NA':
                # Use the column itself as key if normalization_group is None
                key = col.transformation_group if col.transformation_group else col.column_name
                if key not in groups:
                    groups[key] = []
                groups[key].append(col.column_name)
        return groups
    
    def get_by_name(self, column_name: str) -> ColEntry:
        """Get a specific column entry by name"""
        for col in self.columns:
            if col.column_name == column_name:
                return col
        raise ColEntryMissingError(column_name)
    
    def update_transformation(self, column_name: str, transformation: dict):
        """Update transformation info for a column"""
        col = self.get_by_name(column_name)
        if col:
            if col.transformation is None:
                col.transformation = transformation
            else:
                col.transformation.update(transformation)
        else:
            print(f"col {col} not found")
    
    def __repr__(self):
        type_counts = {
            'context'   : len(self.context_columns),
            'target'    : len(self.target_columns),
            'feature'   : len(self.feature_columns),
            'split'     : len(self.split_columns)
        }
        return f"<ColumnRegistration(context={type_counts['context']}, target={type_counts['target']}, feature={type_counts['feature']}, split={type_counts['split']})>"