import numpy as np
from typing import Tuple
def calculate_subplot_layout(n_plots: int, target_width: float=8, target_height: float=6) -> Tuple[int, int, Tuple[float,float]]:
    """
    Calculate optimal number of rows and columns for subplots.
    Each subplot will be approximately target_width x target_height.
    
    Parameters:
    -----------
    n_plots : int
        Number of subplots needed
    target_width : float
        Target width for each subplot
    target_height : float
        Target height for each subplot
    
    Returns:
    --------
    nrows : int
        Number of rows
    ncols : int
        Number of columns
    figsize : tuple
        Total figure size (width, height)
    """
    # Calculate ncols and nrows to get roughly square arrangement
    ncols = int(np.ceil(np.sqrt(n_plots)))
    nrows = int(np.ceil(n_plots / ncols))
    
    # Calculate total figure size
    figsize = (ncols * target_width, nrows * target_height)
    
    return nrows, ncols, figsize