import seaborn as sns 
import matplotlib.pyplot as plt
from typing import List, Tuple

from matplotlib.figure import Figure 
from matplotlib.axes import Axes
import numpy as np

from ...utils.colors import traincolor, valcolor, testcolor
from .orchestrator import EpiDataOrchestrator

class EpiDataPreviewer:
    """ 
    Allows the previewing of data in DataOrchestrator.

    Methods
    -------
    - `finalized_data()`
    - `splits()`
    - `target_lag()`

    Examples
    --------
    >>> previewer = EpiDataPreviewer(dataorchestrator_de)
    >>> previewer.target_lag(transformed=True)

    >>> previewer.finalized_data(transformed=False, features = ['border_denmark', 'border_czech'])    

    """

    def __init__(self, 
                 epidataorchestrator: EpiDataOrchestrator,
                 preview_node: int = 0):
        
        self.DO             = epidataorchestrator
        self.preview_node   = preview_node

    # ======== METHODS ======== #   
    def finalized_data(self, transformed: bool, features: List[str]):
        """
        Preview data in data-orchestrator final - data
        
        Parameters
        ----------
        transformed: bool
            original scale (False) or transformed scale (True)
        features: List[str]
            the list of features to be plotted. One per subplot
        """
        fig, axes   = self._create_fig(len(features), 1)
        axes        = axes.flatten()

        if transformed:
            timeseries  = self.DO.data_final.data

        else:
            timeseries  = self.DO.data_final.data_denorm

        timeseries = timeseries[timeseries['node'] == self.preview_node]
        timeseries = timeseries[['timestamp']+features]


        for ii, feature in enumerate(features):
            ax: Axes = axes[ii]
            sns.lineplot(timeseries,   x = 'timestamp', y = feature, ax = ax)

    def splits(self):
        """ 
        Preview target - split; the target in the train / val / test regimens
        """
        # NOTE wrong timestamp: need to shift by +horizon_leadtime timesteps
        fig, axes       = self._create_fig()
        ax: Axes        = axes.flatten()[0]
        target_colum    = self.DO.column_registration.get_entries_names_by_type('target')[1]
        timeseries      = self.DO.data_final.data_denorm
        timeseries      = timeseries[timeseries['node'] == self.preview_node]
        timeseries      = timeseries[['timestamp',target_colum,'train','val','test']]

        sns.lineplot(timeseries[timeseries['train']],   x = 'timestamp', y = target_colum, color = traincolor, label = 'train', ax = ax)
        sns.lineplot(timeseries[timeseries['val']],     x = 'timestamp', y = target_colum, color = valcolor, label = 'val', ax = ax)
        sns.lineplot(timeseries[timeseries['test']],    x = 'timestamp', y = target_colum, color = testcolor, label = 'test', ax = ax)        
    
    def target_lag(self, transformed: bool):
        """ 
        Preview target - feature lag
        """        

        if not self.DO.config.lag_column == self.DO.config.target_column:
            raise ValueError('currently only supported when lag column is the same as target column. Not the case.')

        fig, axes = self._create_fig()        
        ax: Axes  = axes.flatten()[0]

        if transformed:
            timeseries  = self.DO.data_normalized.data
            title       = f'Transformed {self.DO.config.target_column}'

        else:
            timeseries  = self.DO.data_feature.data
            title       = f'Non-transformed {self.DO.config.target_column}'            

        timeseries = timeseries[timeseries['node'] == self.preview_node]
        timeseries = timeseries[['timestamp']+['target','incidence_lag0']]

        sns.lineplot(timeseries, x = 'timestamp',    y = 'target', color = testcolor, label = 'target', ax = ax)
        sns.lineplot(timeseries, x = 'timestamp',   y = 'incidence_lag0', color = traincolor, label = 'most_recent_lag', ax = ax)

        ax.set_title(title, fontsize = 10, fontweight = 'bold')

    # ======== HIDDEN METHODS ======== #
    def _create_fig(self, nrows = 1, ncols = 1) -> Tuple[Figure, np.ndarray]:
        return plt.subplots(nrows,ncols, figsize = (10 * ncols, 2.5 * nrows), squeeze=False) 
