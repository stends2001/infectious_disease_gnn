from typing import Optional 
import pandas as pd 
import os
from ..issues import MetricsException

class MetricsDFStateMixin:

    metrics_df:     Optional[pd.DataFrame]
    path_exp:       str

    def save_metrics(self):
        if self.metrics_df is None:
            raise MetricsException('no attribute metrics_df. Run `compile_metrics()` first.')

        self.metrics_df.to_csv(os.path.join(self.path_exp,'metrics.csv'), index=False)     

    def load_metrics(self):
        if self.metrics_df is None:
            self.metrics_df = pd.read_csv(os.path.join(self.path_exp,'metrics.csv'))     
        else:
            raise MetricsException('metrics_df already loaded')

        if not hasattr(self, 'model_names'):
            self.model_names = list(self.metrics_df['model'].unique())
