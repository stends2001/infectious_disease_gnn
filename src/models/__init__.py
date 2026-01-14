from .baselines import PersistenceModel, ClimateologyModel, ConstantModel

from .shallow.noderf import NodeRFModel

from .deep import SpatialGNNModel, GATv2Model, NodeLSTMModel, NodeBiLSTMModel, NodeGRUModel

MODELSREGISTRY = {
        'unknown'               : 0,                    # all unknowns will be shown in black
        'ConstantModel'         : ConstantModel,
        'PersistenceModel'      : PersistenceModel,
        'NaiveLinearModel'      : 2,
        'SpatioTemporalXGBModel': 3,
        'SpatialGNNModel'       : SpatialGNNModel,
        'GATv2Model'            : GATv2Model,
        # 'TGCNModel'             : TGCNModel,
        # 'GConvLSTMModel'        : GConvLSTMModel,
        # 'GATv2EmbeddingsModel'  : GATv2EmbeddingsModel,
        'NodeRFModel'           : NodeRFModel,
        'ClimateologyModel'     : ClimateologyModel,
        'NodeLSTMModel'         : NodeLSTMModel,
        'NodeBiLSTMModel'       : NodeBiLSTMModel,
        'NodeGRUModel'          : NodeGRUModel
    }   