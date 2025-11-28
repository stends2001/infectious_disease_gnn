import numpy as np
import pandas as pd 
from typing import Optional, cast, Tuple, List, Union
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from pandas._libs.missing import NAType

class Metrics:

    """
    compute metrics for dataframe    
    """

    def __init__(self, 
                 target_col: str = 'incidence', 
                 pred_col: str = 'pred', 
                 id_col: str = 'node', 
                 temporal_col: str = 'timestamp',
                 edge_index: Optional[torch.Tensor] = None, 
                 edge_weight: Optional[torch.Tensor] = None):
        """
        Initialize metrics calculator (configuration only, not data).
        
        Parameters:
        -----------
        target_col, pred_col, id_col, temporal_col : str
            Column names
        edge_index : torch.Tensor, optional
            Graph structure [2, num_edges]
        edge_weight : torch.Tensor, optional
            Edge weights [num_edges]
        """
        self.target_col = target_col
        self.pred_col = pred_col
        self.id_col = id_col
        self.temporal_col = temporal_col
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    # =============== Node-level metrics =============== #
    def mse(self, df: pd.DataFrame) -> float:
        """return mean square error"""
        return float(np.mean((df[self.target_col] - df[self.pred_col]) ** 2)) # return of np.mean is floating[any] (supports complex numbers)
    
    def rmse(self, df: pd.DataFrame) -> float:
        """Root mean squared error."""
        return np.sqrt(self.mse(df))

    def spearman_corr(self, df: pd.DataFrame) -> Union[float, NAType]:
        """Spearman correlation, with check for zero variance"""
        
        # Check variance of both columns
        if df[self.target_col].var() == 0 or df[self.pred_col].var() == 0:
            return pd.NA
        
        # Calculate Spearman correlation
        corr, _ = spearmanr(df[self.target_col], df[self.pred_col], nan_policy='omit')
        
        return float(corr)  # type: ignore
        
    # def ccc(self, df: pd.DataFrame) -> Union[float, NAType]:
    #     """concordance correlation coefficient."""
    #     target, pred = df[self.target_col], df[self.pred_col]
                  
    #     mean_target, mean_pred  = target.mean(), pred.mean()
    #     var_target,  var_pred   = target.var(ddof=0), pred.var(ddof=0)

    #     if var_target == 0 or var_pred == 0:
    #         return pd.NA

    #     cov = ((target - mean_target) * (pred - mean_pred)).mean()
        
    #     numerator   = 2 * cov
    #     denominator = var_target + var_pred + (mean_target - mean_pred) ** 2 # type: ignore

    #     if denominator == 0:
    #         return pd.NA
    #     else:
    #         return cast(float, numerator / denominator)
            
    def ccc(self, df: pd.DataFrame) -> Union[float, NAType]:
        """concordance correlation coefficient."""
        target, pred = df[self.target_col], df[self.pred_col]
                
        mean_target, mean_pred  = target.mean(), pred.mean()
        var_target,  var_pred   = target.var(ddof=1), pred.var(ddof=1)

        if var_target == 0 or var_pred == 0:
            return pd.NA

        # Sample covariance (divides by n-1)
        cov = ((target - mean_target) * (pred - mean_pred)).sum() / (len(target) - 1)
        
        numerator   = 2 * cov
        denominator = var_target + var_pred + (mean_target - mean_pred) ** 2 # type: ignore

        if denominator == 0:
            return pd.NA
        else:
            return cast(float, numerator / denominator)


    def node_smape(self, df, epsilon=1e-6):
        # Mask rows where both true and predicted values are zero
        true_col = self.target_col
        pred_col  = self.pred_col
        non_zero_mask = ~(df[true_col] == 0) & ~(df[pred_col] == 0)
        df_filtered = df[non_zero_mask]

        # Replace 0's in the true or predicted values with epsilon to avoid division by zero
        true_values = df_filtered[true_col].replace(0, epsilon)
        pred_values = df_filtered[pred_col].replace(0, epsilon)

        # Compute the numerator (absolute difference)
        numerator = abs(true_values - pred_values)
        
        # Compute the denominator (mean of absolute values)
        denominator = (abs(true_values) + abs(pred_values)) / 2
        
        # Ensure the denominator isn't zero or extremely small
        denominator = np.maximum(denominator, epsilon)  # Prevent division by very small numbers

        # Calculate SMAPE as a percentage
        smape = (numerator / denominator).mean() * 100
        return smape


    # =============== Neighborhood-level metrics ============ #        
    def neighborhood_ccc(self, node_df: pd.DataFrame, wide_df: pd.DataFrame, pred_col: Optional[str] = None, self_loops: bool = False) -> Union[float, NAType]:
        """
        Correlation between node's predictions and weighted average of neighbors.
        
        Parameters:
        -----------
        node_df : pd.DataFrame
            Time series for single node (columns: timestamp, node, pred, incidence)
        wide_df : pd.DataFrame
            Wide format with all nodes (columns: timestamp, node1, node2, ...)
        self_loops: bool = True
            Whether to include self-loops
        
        Returns:
        --------
        float : Correlation coefficient, or None if insufficient data
        """
        if self.edge_index is None:
            return pd.NA
        
        if pred_col:
            prediction_column = pred_col
        else:
            prediction_column = self.pred_col

        node_id             = int(node_df[self.id_col].iloc[0])
        neighbors, weights  = self._get_neighborhood(node_id)
        
        neighbor_cols       = [n for n in neighbors if n in wide_df.columns]

        # Merge node data with neighbor data on timestamp
        neighbors_data      = wide_df[[self.temporal_col] + neighbor_cols]
        neighborhood_data   = pd.merge(node_df, neighbors_data, on=self.temporal_col)    

        weighted_ccc_values = []
        total_weight        = 0

        for neighbor_id in neighbor_cols:
            
            if neighbor_id == node_id and not self_loops:
                neighbor_weight = 0.0            
            
            else:
                neighbor_weight = weights[neighbors.index(neighbor_id)]


            # Create temporary df with node's pred and neighbor's values
            temp_df = pd.DataFrame({
                self.target_col: neighborhood_data[prediction_column],    # Compare predictions
                prediction_column:   neighborhood_data[neighbor_id]       # to neighbor's predictions
            })
            
            ccc_value = self.ccc(temp_df)

            if ccc_value is not None:
                weighted_ccc_values.append(ccc_value * neighbor_weight)
                total_weight += neighbor_weight
        
        if total_weight == 0:
            return pd.NA
        
        neighborhood_ccc =  sum(weighted_ccc_values) / total_weight        
        return neighborhood_ccc

    def _get_neighborhood(self, node_id: int) -> Tuple[List[int], List[float]]:
        """Get neighbor node IDs from edge_index."""
        if self.edge_index is None:
            raise ValueError('edge index not found')
        
        if self.edge_weight is None:
            raise ValueError('edge weight not found')        
        
        neighborhood_edges      = self.edge_index[0] == node_id
        
        neighbors               = self.edge_index[1][neighborhood_edges].cpu().numpy()
        neighbors_list          = [int(nn) for nn in neighbors]

        neighbors_weights       = self.edge_weight[neighborhood_edges].cpu().numpy()
        neighbors_weights_list  = [float(ww) for ww in neighbors_weights]        

        return neighbors_list, neighbors_weights_list
