"""
Three types of models:
- baseline models
- shallow models
- deep models
"""
from .baselines import PersistenceModel, ClimateologyModel, ConstantModel, ClimaScaleModel
from .deep import GCNModel, GATModel

# from .shallow.noderf import NodeRFModel

# from .deep import SimpleGCNModel, GATv2LSTMModel, LSTMModel, GCNLSTMModel, GCN2Model, DecoupledGCNModel, GCNModel, GATModel

# from .deep import SimpleGCNModel, GATv2LSTMModel, NodeLSTMModel, NodeBiLSTMModel, NodeGRUModel, SeqNodeLSTMModel,GATv2Model