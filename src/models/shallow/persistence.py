from typing import Literal, Union, Optional
from ...utils.textformatting import section, align
from ..base.basemodel import BaseModel, EpiDataLoader

class PersistenceModel(BaseModel):

    """
    Persistence model predicts persistence: i.e. most recently 
    observed value in the lagged features is predicted.

    Training is therefore not necessary
    """

    def __init__(self, dataloader: EpiDataLoader, name: Optional[str] = None):
        super().__init__(dataloader, name)

        if not self.name:
            self.name = 'Persistence Model'

        self.prediction_col = self._get_lag_column()

    def train(self):

        print("This naive model doesn't train")


    def forecast(self, dataset: Literal['train','val','test'] = 'test'):

        evaluation_df           = self.dataloader.data['final'][self.dataloader.data['final'][dataset]].copy().rename(columns = {'incidence_h0': 'incidence'})
        evaluation_df['pred']   = evaluation_df[self.prediction_col]

        evaluation_df_dict      = {'transformed'    : {'horizon_0': evaluation_df},
                                   'nontransformed' : {'horizon_0': self._denorm_predictions(evaluation_df)}}


        self.evaluation_datasets[dataset] = evaluation_df_dict

        return self

    def _get_lag_column(self) -> Union[None, str]:
        

        return f'incidence_lag{min(self.dataloader.lags)+self.dataloader.horizon_leadtime}'
        
    def __str__(self):
        # Calculate width
        all_keys = ['model name', 'model class', 'prediction column', 'forecasted']
        width = max(len(k) for k in all_keys)
        
        # Build output
        lines = ['<PersistenceModel(']
        lines.append(align('model name', self.name, width))
        lines.append(align('model class', self.model_class, width))
        lines.append(align('prediction column', self.prediction_col, width))
        lines.append('')
        
        # Status section
        lines.extend(section('status', {'forecasted': list(self.evaluation_datasets.keys())}, width))
        
        lines.append(')>')
        
        return '\n'.join(lines)