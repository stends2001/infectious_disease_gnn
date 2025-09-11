import pandas as pd
from typing import *
import numpy as np

def pipeline_normalization(train_df, 
                            columns):
    """
    takes training data and standardizes the columns specified into minmax
    
    returns the same dataframe with the specified columns standardized. Other columns remain untouched.
    """
    params = {}
    scaled_df = train_df.copy()
    
    for col in columns:
        col_min = train_df[col].min()
        col_max = train_df[col].max()
        params[col] = {'min': col_min, 'max': col_max}
        
        if col_max - col_min == 0:
            scaled_df[col] = 0.0
        else:
            scaled_df[col] = (train_df[col] - col_min) / (col_max - col_min)
    
    return scaled_df, params

def apply_minmax_scaling(val_df, columns, params):
    """
    takes data and standardizes following parameters given. 
    This will be used for the validation dataset after having standardized the training dataset.

    returns scaled data
    """
    scaled_df = val_df.copy()
    
    for col in columns:
        if col not in params:
            raise ValueError(f"Scaling parameters for column '{col}' not found.")
        
        col_min = params[col]['min']
        col_max = params[col]['max']
        
        if col_max - col_min == 0:
            scaled_df[col] = 0.0
        else:
            scaled_df[col] = (val_df[col] - col_min) / (col_max - col_min)
    
    return scaled_df

def reverse_minmax_scaling(scaled_df, columns, params):
    """
    Reverses min-max scaling using original min-max parameters.
    
    Returns the unscaled DataFrame.
    """
    unscaled_df = scaled_df.copy()
    
    cases_min = params['min']
    cases_max = params['max']

    for col in columns:
        
        if cases_max - cases_min == 0:
            unscaled_df[col] = cases_min  # All values were originally the same
        else:
            unscaled_df[col] = scaled_df[col] * (cases_max - cases_min) + cases_min
            
    return unscaled_df

# Z-Score Normalization Functions

def pipeline_zscore_normalization(train_df, columns):
    """
    takes training data and standardizes the columns specified using Z-score normalization
    
    returns the same dataframe with the specified columns standardized. Other columns remain untouched.
    """
    params = {}
    scaled_df = train_df.copy()
    
    for col in columns:
        col_mean = train_df[col].mean()
        col_std = train_df[col].std()
        params[col] = {'mean': col_mean, 'std': col_std}
        
        if col_std == 0:
            scaled_df[col] = 0.0  # All values were the same
        else:
            scaled_df[col] = (train_df[col] - col_mean) / col_std
    
    return scaled_df, params

def apply_zscore_scaling(val_df, columns, params):
    """
    takes data and standardizes following Z-score parameters given. 
    This will be used for the validation dataset after having standardized the training dataset.

    returns scaled data
    """
    scaled_df = val_df.copy()
    
    for col in columns:
        if col not in params:
            raise ValueError(f"Scaling parameters for column '{col}' not found.")
        
        col_mean = params[col]['mean']
        col_std = params[col]['std']
        
        if col_std == 0:
            scaled_df[col] = 0.0
        else:
            scaled_df[col] = (val_df[col] - col_mean) / col_std
    
    return scaled_df

def reverse_zscore_scaling(scaled_df: pd.DataFrame, params: Dict):
    """
    Reverses Z-score scaling using original mean and standard deviation parameters.
    Returns the unscaled DataFrame.
    """

    columns = list(params.keys())
    unscaled_df = scaled_df.copy()
    
    for col in columns:
        if col not in scaled_df.columns.tolist():
            # raise ValueError(f"Scaling parameters for column '{col}' not found.")
            pass 

        else:
            col_mean = params[col]['mean']
            col_std  = params[col]['std']
            
            if col_std == 0:
                unscaled_df[col] = col_mean  # All values were originally the same

            else:
                unscaled_df[col] = scaled_df[col] * col_std + col_mean
            
    return unscaled_df


def reverse_log(logged_df: pd.DataFrame, params: Dict):

    columns = list(params.keys())

    unlogged_df = logged_df.copy() 

    for col in columns:
        if col not in logged_df.columns.tolist():
            pass 
            
        else:
            unlogged_df[col] = np.exp(unlogged_df[col]) - params[col]['shift']

    return unlogged_df