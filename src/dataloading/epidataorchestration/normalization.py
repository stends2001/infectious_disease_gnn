import pandas as pd
from typing import List, Tuple, Dict
import numpy as np

# MINMAX Normalization Functions

def pipeline_minmax_normalization(train_df: pd.DataFrame, 
                            columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    takes training data and standardizes the columns specified into minmax
    
    returns the same dataframe with the specified columns standardized. Other columns remain untouched.
    """
    params = {}
    scaled_df = train_df.copy()
    
    for col in columns:
        if col in train_df.columns:
            col_min = train_df[col].min()
            col_max = train_df[col].max()
            params[col] = {'min': col_min, 'max': col_max}
            
            if col_max - col_min == 0:
                scaled_df[col] = 0.0
            else:
                scaled_df[col] = (train_df[col] - col_min) / (col_max - col_min)
        else:
            print(f'trying to pipeline minmax normalization {col} but not found')
        
    return scaled_df, params

def apply_minmax_scaling(val_df: pd.DataFrame, columns: List[str], params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    takes data and standardizes following parameters given. 
    This will be used for the validation dataset after having standardized the training dataset.

    returns scaled data
    """
    scaled_df = val_df.copy()
    
    for col in columns:
        if col in val_df.columns:
            if col not in params:
                raise ValueError(f"Scaling parameters for column '{col}' not found.")
            
            col_min = params[col]['min']
            col_max = params[col]['max']
            
            if col_max - col_min == 0:
                scaled_df[col] = 0.0
            else:
                scaled_df[col] = (val_df[col] - col_min) / (col_max - col_min)     
        else:
            print(f'trying to apply minmax scaling {col} but not found')
    return scaled_df

def reverse_minmax_scaling(scaled_df: pd.DataFrame, params: dict, column: str) -> pd.DataFrame:
    """
    Reverses min-max scaling using original min-max parameters.
    
    Returns the unscaled DataFrame.
    """
    unscaled_df = scaled_df.copy()
    
    col_min = params['min']
    col_max = params['max']

    if col_max - col_min == 0:
        unscaled_df[column] = col_min  # All values were originally the same
    else:
        unscaled_df[column] = scaled_df[column] * (col_max - col_min) + col_min
            
    return unscaled_df

# Z-Score Normalization Functions

def pipeline_zscore_normalization(train_df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    takes training data and standardizes the columns specified using Z-score normalization
    
    returns the same dataframe with the specified columns standardized. Other columns remain untouched.
    """
    params = {}
    scaled_df = train_df.copy()
    
    for col in columns:
        if col in train_df.columns:
            col_mean = train_df[col].mean()
            col_std = train_df[col].std()
            params[col] = {'mean': col_mean, 'std': col_std}
            
            if col_std == 0:
                scaled_df[col] = 0.0  # All values were the same
            else:
                scaled_df[col] = (train_df[col] - col_mean) / col_std
        else:
            print(f'trying to pipeline zscore normalization {col} but not found')
        
    return scaled_df, params

def apply_zscore_scaling(val_df: pd.DataFrame, columns: List[str], params:Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    takes data and standardizes following Z-score parameters given. 
    This will be used for the validation dataset after having standardized the training dataset.

    returns scaled data
    """
    scaled_df = val_df.copy()
    
    for col in columns:
        if col in val_df.columns:        
            if col not in params:
                raise ValueError(f"Scaling parameters for column '{col}' not found.")
            
            col_mean = params[col]['mean']
            col_std = params[col]['std']
            
            if col_std == 0:
                scaled_df[col] = 0.0
            else:
                scaled_df[col] = (val_df[col] - col_mean) / col_std
        else:
            print(f'trying to apply zscore scaling {col} but not found')
    return scaled_df

def reverse_zscore_scaling(scaled_df: pd.DataFrame, params: dict, column: str) -> pd.DataFrame:
    """
    Reverses Z-score scaling using original mean and standard deviation parameters.
    Returns the unscaled DataFrame.
    """
    unscaled_df = scaled_df.copy()

    if column not in scaled_df.columns.tolist():
        # raise ValueError(f"Scaling parameters for column '{col}' not found.")
        pass 

    else:
        col_mean = params['mean']
        col_std  = params['std']
        
        if col_std == 0:
            unscaled_df[column] = col_mean  # All values were originally the same

        else:
            unscaled_df[column] = scaled_df[column] * col_std + col_mean
            
    return unscaled_df

def reverse_log(logged_df: pd.DataFrame, shift: float, column: str) -> pd.DataFrame:

    unlogged_df = logged_df.copy() 

    if column not in logged_df.columns.tolist():
        pass 
        
    else:
        unlogged_df[column] = np.exp(unlogged_df[column]) - shift

    return unlogged_df