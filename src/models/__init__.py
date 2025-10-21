from .base.persistencemodel import PersistenceModel
from .deep.gatv2 import GATv2Model
from .deep.tgcn import TGCNModel
from .deep.gconvlstm import GConvLSTMModel
from .deep.spatial import SpatialGNNModel
from .deep.gatv2_embeddings import GATv2EmbeddingsModel

MODELSREGISTRY = {
        'unknown'               : 0,                    # all unknowns will be shown in black
        'PersistenceModel'      : PersistenceModel,
        'NaiveLinearModel'      : 2,
        'SpatioTemporalXGBModel': 3,
        'SpatialGNNModel'       : SpatialGNNModel,
        'GATv2Model'            : GATv2Model,
        'TGCNModel'             : TGCNModel,
        'GConvLSTMModel'        : GConvLSTMModel,
        'GATv2EmbeddingsModel'  : GATv2EmbeddingsModel
    }   