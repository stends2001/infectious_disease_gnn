from typing import Optional 
import pandas as pd 
import os
from ..exceptions import MetricsException

import logging
logger = logging.getLogger(__name__)

class MetricsDFStateMixin:
    """"
    Mixinclass to ExperimentAnalyzer that deals with the saving of metrics_df.  
    """

    metrics_df:     Optional[pd.DataFrame]
    path_exp:       str

    metrics_df_filename: str

    def save_metrics(self):
        if self.metrics_df is None:
            raise MetricsException('no attribute metrics_df. Run `compile_metrics()` first.')

        self.metrics_df.to_csv(os.path.join(self.path_exp, self.metrics_df_filename), index=False)     
        logger.info("metrics-df saved at %s", os.path.join(self.path_exp, self.metrics_df_filename))

    def load_metrics(self):
        if self.metrics_df is None:
            self.metrics_df = pd.read_csv(os.path.join(self.path_exp, self.metrics_df_filename))     
            logger.info("metrics-df loaded")            
        else:
            raise MetricsException('metrics_df already loaded')

        if not hasattr(self, 'model_names'):
            self.model_names = list(self.metrics_df['model'].unique())
