from typing import Literal, Union, Optional

from .basemodel import BaseModel, EpiDataLoader

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

        print("Naive model doesn't train")


    def forecast(self, dataset: Literal['train','val','test'] = 'test'):

        evaluation_df           = self.dataloader.data['final'][self.dataloader.data['final'][dataset]].copy().rename(columns = {'incidence_h0': 'incidence'})
        evaluation_df['pred']   = evaluation_df[self.prediction_col]

        evaluation_df_dict      = {'transformed'    : {'horizon_0': evaluation_df},
                                   'nontransformed' : {'horizon_0': self._denorm_predictions(evaluation_df)}}


        self.evaluation_datasets[dataset] = evaluation_df_dict

        return self

    def _get_lag_column(self) -> Union[None, str]:
        prediction_col = None
        for (ii, feature_col) in enumerate(self.dataloader.feature_columns):
            if 'lag' in feature_col:
                prediction_col = feature_col        
        return prediction_col
        
    def __repr__(self) -> str:
        """String representation for PersistenceModel"""
        max_len = max(
            len('model name'),
            len('model class'),
            len('prediction column'),
            len('forecasted')
        )
        
        lines = [
            '<PersistenceModel(',
            f"    {'model name':<{max_len}} : {self.name}",
            f"    {'model class':<{max_len}} : {self.model_class}",
            f"    {'prediction column':<{max_len}} : {self.prediction_col}",
            '',
            '    ----------- STATUS --------------',
            f"    {'forecasted':<{max_len}} : {list(self.evaluation_datasets.keys())}",
            ')>'
        ]
        
        return '\n'.join(lines)