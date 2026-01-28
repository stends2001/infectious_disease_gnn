from .baselines import PersistenceModel, ClimateologyModel, ConstantModel

from .shallow.noderf import NodeRFModel

from .deep import SimpleGCNModel, GATv2LSTMModel, NodeLSTMModel, NodeBiLSTMModel, NodeGRUModel, SeqNodeLSTMModel,GATv2Model

MODELSREGISTRY = {
        'unknown'               : 0,                    # all unknowns will be shown in black
        'ConstantModel'         : ConstantModel,
        'PersistenceModel'      : PersistenceModel,
        'NaiveLinearModel'      : 2,
        'SpatioTemporalXGBModel': 3,
        'SpatialGNNModel'       : SimpleGCNModel,
        'GATv2LSTMModel'            : GATv2LSTMModel,
        'SeqNodeLSTMModel'      : SeqNodeLSTMModel,
        'GATv2Model'             : GATv2Model,
        # 'GConvLSTMModel'        : GConvLSTMModel,
        # 'GATv2EmbeddingsModel'  : GATv2EmbeddingsModel,
        'NodeRFModel'           : NodeRFModel,
        'ClimateologyModel'     : ClimateologyModel,
        'NodeLSTMModel'         : NodeLSTMModel,
        'NodeBiLSTMModel'       : NodeBiLSTMModel,
        'NodeGRUModel'          : NodeGRUModel
    }   