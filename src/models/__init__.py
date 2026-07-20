"""
Models

The models that make the spatiotemporal epidemiological predictions. Divided into:
- Base (relevant to all models: BaseModel, Predictions)
- BaseLine (Climatology and Persistence)
- Deep (GNNs and other Neural Networks)
"""

from .baselines import PersistenceModel, ClimateologyModel, ConstantModel, ClimaScaleModel
from .deep import GCNModel, GATModel