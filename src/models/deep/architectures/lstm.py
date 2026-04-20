import torch 
import torch.nn as nn    
from torch import Tensor
from typing import Optional, Tuple, Literal, Union

from ....dataloading.dataloaders.deepdataloaders.deepdataloader import DeepDataLoaderManager
from ....dataloading.dataloaders.deepdataloaders.graphdataloader import GraphDataLoaderManager

from ..deepmodel import DeepModel

from ..debugging import DebuggingLine, ModelDebuggingReport
from ..strategies.lstmstrategy import StatelessLSTMStrategy, StatefullLSTMStrategy

class LSTMArchitecture(nn.Module):
    """
    LSTM applied independently to each node's time series.
    
    Input: [num_nodes, node_features, seq_length]
    Output: [num_nodes, prediction_horizon]
    """
    def __init__(self, 
                 num_features:      int, 
                 num_nodes:         int,
                 hidden_size:       int, 
                 num_layers:        int, 
                 dropout:           float, 
                 horizon_size:      int, 
                 seq_length:        int,
                 num_quantiles:     int
                 ):
        super().__init__()
        
        self.seq_length     = seq_length
        self.num_features   = num_features
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        self.num_nodes      = num_nodes
        self.horizon_size   = horizon_size
        self.num_quantiles  = num_quantiles

        # LSTM expects [seq_len, batch, features]
        # We'll treat num_nodes as batch dimension
        self.lstm = nn.LSTM(
            input_size  = num_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0,
            batch_first = False # [seq_len, num_nodes, features]
        )
            
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon_size * num_quantiles)
        )


    def forward(self, x: Tensor, hidden_state: Optional[Tensor]=None) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [num_nodes, node_features, seq_length]
            hidden_state: Optional tuple of (h, c) where each is [num_layers, num_nodes, hidden_size]
        
        Returns:
            output: [num_nodes, prediction_horizon]
            h: [num_layers, num_nodes, hidden_size]
            c: [num_layers, num_nodes, hidden_size]
        """
        x_seq = x.permute(2, 0, 1)  # [seq_len, num_nodes, features]
        
        # Run LSTM
        if hidden_state is None:
            lstm_out, (h, c) = self.lstm(x_seq)

        else:
            lstm_out, (h, c) = self.lstm(x_seq, hidden_state)
        
        # Take the last timestep's output
        # lstm_out: [seq_len, num_nodes, hidden_size]
        last_output = lstm_out[-1]  # [num_nodes, hidden_size]
        
        # Project to prediction horizon
        output = self.output_proj(last_output)


        output = output.view(self.num_nodes, self.horizon_size, self.num_quantiles)
        
        return output, h, c

    def debug(self, x: Tensor, hidden_state = None) ->  Tuple[Tensor, 'ModelDebuggingReport']:
        lstm_out: Tensor
        output:   Tensor
        x_seq = x.permute(2, 0, 1)              # [seq_len, num_nodes, features]

        # lstm layer expects input: [seq_len, batch, features]

        lstm_out, (h, c)= self.lstm(x_seq)

        last_output     = lstm_out[-1]                  # [num_nodes, hidden_size]
        output          = self.output_proj(last_output) # [num_nodes, prediction_horizon]
        
        dbl1 = DebuggingLine(list(x_seq.shape), [self.seq_length, self.num_nodes, self.num_features])        
        dbl2 = DebuggingLine(list(lstm_out.shape), [self.seq_length, self.num_nodes, self.hidden_size])        
        dbl3 = DebuggingLine(list(last_output.shape), [self.num_nodes, self.hidden_size])
        dbl4 = DebuggingLine(list(output.shape), [self.num_nodes, self.horizon_size])

        debugging_report = ModelDebuggingReport([dbl1, dbl2, dbl3, dbl4])

        return output, debugging_report

class LSTMModel(DeepModel):
    """
    Pure temporal LSTM model (no spatial structure)

    Parameters
    ----------
    dataloadermanager: DeepDataLoaderManager
    name: Optional[str]
    reset: Literal['epoch','dataset'] = 'epoch' 
        whether to reset the state each dataset only ('dataset'), or between epochs as well ('epoch')
    verbose: Literal[-1,0,1,2]
        quantification of output
    """
    _expected_dataloadermanager = 'DeepDataLoaderManager'

    def __init__(self, 
                 dataloadermanager: DeepDataLoaderManager, 
                 name:              str = 'lstm_model',
                 reset:             Literal['epoch','dataset'] = 'epoch',
                 verbose:           Literal[-1, 0, 1, 2] = -1):        
        
        #self._expected_dataloadermanager = 'DeepDataLoaderManager'

        strategy = StatelessLSTMStrategy() if reset == 'epoch' else StatefullLSTMStrategy()
        
        super().__init__(
            dataloadermanager   = dataloadermanager, 
            name                = name, 
            verbose             = verbose, 
            strategy            = strategy
        )

    def set_model_hparams(self, 
                          hidden_size: int = 64, 
                          num_layers: int = 1,
                          dropout: float = 0.3):
        
        _num_features   = len(self.column_registration.get_by_type('feature'))
        _num_nodes      = len(self.dataloadermanager.dataorchestrator.data_context.local_shapedata)
        _horizon_size   = self.dataloadermanager.dataorchestrator.config.horizon_size
        _seq_length     = self.dataloadermanager.dataorchestrator.config.sequence_length
        _num_quantiles  = max(self.dataloadermanager.dataorchestrator.config._num_quantiles,1)        
        
        self.model = LSTMArchitecture(
            num_features    = _num_features,
            num_nodes       = _num_nodes,
            hidden_size     = hidden_size,
            num_layers      = num_layers,
            dropout         = dropout,
            horizon_size    = _horizon_size,
            seq_length      = _seq_length,
            num_quantiles   = _num_quantiles
        ).to(self.device)
        
        model_hparams_config = {
            'hidden_size'     : hidden_size,
            'num_layers'      : num_layers,
            'dropout'         : dropout,
        }
        
        self.config_info['model_hparams'] = model_hparams_config
        self._update_status('model_hparams_set')
