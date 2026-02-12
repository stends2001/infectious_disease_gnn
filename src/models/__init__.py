"""
Three types of models:
- baseline models
- shallow models
- deep models
"""
from .baselines import PersistenceModel, ClimateologyModel, ConstantModel, ClimaScaleModel

# from .shallow.noderf import NodeRFModel

from .deep import SimpleGCNModel, GATv2LSTMModel, LSTMModel

# from .deep import SimpleGCNModel, GATv2LSTMModel, NodeLSTMModel, NodeBiLSTMModel, NodeGRUModel, SeqNodeLSTMModel,GATv2Model