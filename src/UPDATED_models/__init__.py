from .shallow.persistence import PersistenceModel
from .shallow.noderf import NodeRFModel
from .deep.gatv2model import GATv2Model
from .deep.nodelstm import NodeLSTMModel

# from .deep.gatv2 import GATv2Model
# from .deep.tgcn import TGCNModel
# from .deep.gconvlstm import GConvLSTMModel
from .deep.spatialgnnmodel import SpatialGNNModel
# from .deep.gatv2_embeddings import GATv2EmbeddingsModel

# from .shallow.node_rf import NodeRFModel

MODELSREGISTRY = {
        'unknown'               : 0,                    # all unknowns will be shown in black
        'PersistenceModel'      : PersistenceModel,
        'NaiveLinearModel'      : 2,
        'SpatioTemporalXGBModel': 3,
        'SpatialGNNModel'       : SpatialGNNModel,
        'GATv2Model'            : GATv2Model,
        # 'TGCNModel'             : TGCNModel,
        # 'GConvLSTMModel'        : GConvLSTMModel,
        # 'GATv2EmbeddingsModel'  : GATv2EmbeddingsModel,
        'NodeRFModel'           : NodeRFModel,
        'NodeLSTMModel'         : NodeLSTMModel
    }   