# import pandas as pd
# import numpy as np
# from ._basemodel import BaseModel

# class LocalTrendPredictor(BaseModel):

#     """
#     repeats the observed trend (predicts by one delay)
    
#     """

#     def __init__(self, dataloader, name= None):
#         super().__init__(dataloader, name= name)
#         if not self.name:
#             self.name = f'LTP'
#         self.model_color = "#666666"

#         self.XYt_test = self.dataloader.XYt_test.copy()

#     def forecast(self):


#         evaluation_df = self.XYt_test

#         evaluation_df['preds'] = evaluation_df[f'{self.target_column}_lag1'] + (evaluation_df[f'{self.target_column}_lag1']-evaluation_df[f'{self.target_column}_lag2'])


#         self.evaluation_df = evaluation_df
#         return self        
