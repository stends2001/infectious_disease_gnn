from typing import Literal, Dict
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class PredictionCompilation:
    """
    A class to store and manage predictions across multiple datasets (train, val, test, etc.).
    """
    _data: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)  # Stores datasets with horizons as keys
    
    def add_horizon(self, data: pd.DataFrame, horizon: str, dataset: str):
        """Add a new horizon for a specified dataset."""
        if dataset not in self._data:
            self._validate_dataset(dataset)
            self._data[dataset] = {}  # Initialize the dataset if it doesn't exist
            
        self._data[dataset][horizon] = data
    
    def get_compilation(self, horizon: str, dataset: str) -> pd.DataFrame:
        """Get the data for a specified horizon and dataset."""
        if dataset not in self._data or horizon not in self._data[dataset]:
            raise ValueError(f'No compilation found for {dataset} with horizon {horizon}')
        return self._data[dataset][horizon]
    
    @property
    def compilations(self) -> Dict[str, list[str]]:
        """Return the horizons for each dataset."""
        return {dataset: sorted(horizons.keys()) for dataset, horizons in self._data.items()}
    
    def _validate_dataset(self, dataset):
        if dataset not in ['train','val','test']:
            raise ValueError(f'{dataset} not valid. Please supply train, val or test')
          
        
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd


@dataclass
class MetricCompilation:
    """
    A class to store and manage metrics across multiple datasets (train, val, test)
    and prediction horizons.
    
    Structure
    ---------
    _data[dataset][horizon][metric_name] = DataFrame
    
    Example
    -------
    _data['test']['horizon_1']['mse'] = DataFrame with columns [node, model1, model2, ...]
    
    Examples
    --------
    >>> metrics = MetricCompilation()
    >>> metrics.add_horizon(metrics_dict, 'horizon_1', 'test')
    >>> mse_df = metrics.get_metric('horizon_1', 'test', 'mse')
    >>> all_metrics = metrics.get_compilation('horizon_1', 'test')
    """
    _data: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = field(default_factory=dict)
    
    def add_horizon(self, data: Dict[str, pd.DataFrame], horizon: str, dataset: str):
        """
        Add metrics for a new horizon and dataset.
        
        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Dictionary mapping metric names to DataFrames
            e.g., {'mse': df_mse, 'rmse': df_rmse, ...}
        horizon : str
            Horizon identifier (e.g., 'horizon_0', 'horizon_1')
        dataset : str
            Dataset name ('train', 'val', or 'test')
        """
        self._validate_dataset(dataset)
        
        if dataset not in self._data:
            self._data[dataset] = {}
        
        self._data[dataset][horizon] = data
    
    def get_compilation(self, horizon: str, dataset: str) -> Dict[str, pd.DataFrame]:
        """
        Get all metrics for a specified horizon and dataset.
        
        Parameters
        ----------
        horizon : str
            Horizon identifier
        dataset : str
            Dataset name
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping metric names to DataFrames
            
        Raises
        ------
        ValueError
            If dataset or horizon not found
        """
        if dataset not in self._data:
            raise ValueError(
                f"No data found for dataset '{dataset}'. "
                f"Available datasets: {list(self._data.keys())}"
            )
        
        if horizon not in self._data[dataset]:
            raise ValueError(
                f"No data found for horizon '{horizon}' in dataset '{dataset}'. "
                f"Available horizons: {list(self._data[dataset].keys())}"
            )
        
        return self._data[dataset][horizon]
    
    def get_metric(self, horizon: str, dataset: str, metric_name: str) -> pd.DataFrame:
        """
        Get a specific metric for a specified horizon and dataset.
        
        Parameters
        ----------
        horizon : str
            Horizon identifier
        dataset : str
            Dataset name
        metric_name : str
            Name of the metric (e.g., 'mse', 'rmse')
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns [node, model1, model2, ...]
            
        Raises
        ------
        ValueError
            If dataset, horizon, or metric not found
        """
        compilation = self.get_compilation(horizon, dataset)
        
        if metric_name not in compilation:
            raise ValueError(
                f"Metric '{metric_name}' not found for horizon '{horizon}', "
                f"dataset '{dataset}'. Available metrics: {list(compilation.keys())}"
            )
        
        return compilation[metric_name]
    
    @property
    def datasets(self) -> List[str]:
        """Return list of available datasets."""
        return sorted(self._data.keys())
    
    @property
    def compilations(self) -> Dict[str, List[str]]:
        """
        Return the horizons available for each dataset.
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping dataset names to lists of horizon identifiers
            
        Example
        -------
        >>> metrics.compilations
        {'train': ['horizon_0', 'horizon_1'], 'test': ['horizon_0']}
        """
        return {
            dataset: sorted(horizons.keys()) 
            for dataset, horizons in self._data.items()
        }
    
    @property
    def metrics(self) -> Dict[str, List[str]]:
        """
        Return unique metric names available for each dataset.
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping dataset names to lists of available metric names
            
        Example
        -------
        >>> metrics.metrics
        {'train': ['ccc', 'mse', 'rmse'], 'test': ['ccc', 'mse', 'rmse']}
        """
        result = {}
        
        for dataset, horizons in self._data.items():
            # Collect all unique metric names across all horizons for this dataset
            all_metrics = set()
            for horizon_data in horizons.values():
                all_metrics.update(horizon_data.keys())
            
            result[dataset] = sorted(all_metrics)
        
        return result
    
    def get_metrics_for_horizon(self, horizon: str, dataset: str) -> List[str]:
        """
        Get list of available metrics for a specific horizon and dataset.
        
        Parameters
        ----------
        horizon : str
            Horizon identifier
        dataset : str
            Dataset name
            
        Returns
        -------
        List[str]
            List of metric names available
        """
        compilation = self.get_compilation(horizon, dataset)
        return sorted(compilation.keys())
    
    def has_metric(self, horizon: str, dataset: str, metric_name: str) -> bool:
        """
        Check if a specific metric exists for given horizon and dataset.
        
        Parameters
        ----------
        horizon : str
            Horizon identifier
        dataset : str
            Dataset name
        metric_name : str
            Name of the metric
            
        Returns
        -------
        bool
            True if metric exists, False otherwise
        """
        try:
            self.get_metric(horizon, dataset, metric_name)
            return True
        except ValueError:
            return False
    
    def _validate_dataset(self, dataset: str):
        """
        Validate that dataset name is one of the allowed values.
        
        Parameters
        ----------
        dataset : str
            Dataset name to validate
            
        Raises
        ------
        ValueError
            If dataset is not 'train', 'val', or 'test'
        """
        valid_datasets = ['train', 'val', 'test']
        if dataset not in valid_datasets:
            raise ValueError(
                f"Invalid dataset '{dataset}'. "
                f"Must be one of: {valid_datasets}"
            )
    
    def __repr__(self) -> str:
        """Return string representation of the compilation."""
        if not self._data:
            return "<MetricCompilation(empty)>"
        
        summary = []
        for dataset, horizons in self._data.items():
            n_horizons = len(horizons)
            n_metrics = len(self.metrics.get(dataset, []))
            summary.append(f"{dataset}({n_horizons} horizons, {n_metrics} metrics)")
        
        return f"<MetricCompilation({', '.join(summary)})>"
    
    def summary(self) -> str:
        """
        Return detailed summary of stored metrics.
        
        Returns
        -------
        str
            Formatted summary string
        """
        if not self._data:
            return "MetricCompilation is empty"
        
        lines = ["MetricCompilation Summary:", "=" * 50]
        
        for dataset in sorted(self._data.keys()):
            lines.append(f"\n{dataset.upper()}:")
            horizons = self._data[dataset]
            
            for horizon in sorted(horizons.keys()):
                lines.append(f"  {horizon}:")
                metrics_dict = horizons[horizon]
                
                for metric_name in sorted(metrics_dict.keys()):
                    df = metrics_dict[metric_name]
                    n_nodes = len(df)
                    n_models = len(df.columns) - 1  # Subtract 'node' column
                    lines.append(
                        f"    - {metric_name}: {n_nodes} nodes × {n_models} models"
                    )
        
        return "\n".join(lines)