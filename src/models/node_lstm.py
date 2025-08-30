
import torch, random, numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from ..dataloading.gnndataloader import GNNDataLoader
from .modelcore import DeepLearningModelCore

class LSTMArchitecture(torch.nn.Module):

    """

    """

    def __init__(self, node_features, periods, lstm_hidden_size, num_layers):
        super(LSTMArchitecture, self).__init__()
        self.periods = periods
        self.lstm_hidden_size = lstm_hidden_size
        self.node_features = node_features

        # LSTM: input size = node_features, 
        #       output = lstm_hidden_size
        self.lstm = torch.nn.LSTM(input_size=node_features,
                                  hidden_size=lstm_hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True)

        # Final linear layer to predict next step (or output size 1)
        self.linear = torch.nn.Linear(lstm_hidden_size, 1)

    def forward(self, x, edge_index = None, edge_weight = None):
        """
        x: [num_nodes, node_features, periods]
        takes in edge_index and edge_weight for duck-typing
        purposes, these arguments are not used.
        """
        # Permute to (num_nodes, periods, node_features) for LSTM
        x = x.permute(0, 2, 1)

        # Run LSTM: outputs shape (num_nodes, periods, hidden_size)
        lstm_out, _ = self.lstm(x)  

        # Take the last time step output for each node. 
        # TODO create a proper sequence of predictied timepoints
        last_period_out = lstm_out[:, -1, :]  # (num_nodes, lstm_hidden_size)

        # Apply linear layer and relu
        out = self.linear(last_period_out)  # (num_nodes, 1)
        
        #out = torch.tanh(out) # clipped output!

        return out.squeeze(-1)  # shape (num_nodes,)


class NodeLSTM(DeepLearningModelCore):
    """
    'simple' LSTM: a single LSTM that infers the same temporal dependency for each node. In other words, a single model
    adapts weights and biases. The only difference among the nodes is the features, leading to different predictions.    
    Single LSTM model making Kreise-specific-predictions

    The default has become:
    
    >>> epidata = EpiDataLoader(disease_name = 'influenza', min_date = ' 2014-06-01', data_env_dir = data_env)
    >>> epidata.add_time_features()
    >>> epidata.normalize('2018-06-01','2019-06-01', method = 'zscore')
    >>> epidata.add_lagged_features(lags = range(4,9))
    >>> epidata.split_data()
    >>> epidata.preview()
    >>> n_epochs = 250


    >>> dataloader_NodeLSTM = GNNDataLoader(epidata).construct_dataloaders(periods = 10)
    >>> nodelstm_model = NodeLSTM(dataloader_NodeLSTM)

    >>> nodelstm_model.set_model_hparams(hidden_size=256, num_layers = 4)

    >>> nodelstm_model.set_global_hparams(lr = 0.00001, 
    >>>                                 n_epochs = n_epochs,     
    >>>                                 optimizer='adam', 
    >>>                                 scheduler='step',
    >>>                                 scheduler_kwargs={'step_size': 15,
    >>>                                                     'gamma' : 0.8},
    >>>                                 loss = 'spike_weighted_mse',
    >>>                                 min_delta = 1e-4
    >>>                                 )

    >>> nodelstm_model.train()
    >>> nodelstm_model.forecast()
    >>> nodelstm_model.show_forecasts(391)
    >>> nodelstm_model.show_forecasts(200)
    >>> nodelstm_model.show_forecasts(15)    
    """
    def __init__(self, dataloader: GNNDataLoader, name= None):
        super().__init__(dataloader, name= name)
        if not self.name:
            self.name = f'NodeLSTM'

        self.model_color = '#FF7F0E'
        self.dataloader  = dataloader

    def set_model_hparams(self, 
                            hidden_size: int           = 32,
                            num_layers: int            = 2, 
                            ):
        
        self.model_hparams_set = True
        self.model = LSTMArchitecture(node_features    = len(self.dataloader.feature_columns), 
                                      periods          = self.dataloader.periods, 
                                      lstm_hidden_size = hidden_size,
                                      num_layers       = num_layers
                                      ).to(self.device)

        return self