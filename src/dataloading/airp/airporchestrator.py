from typing import TYPE_CHECKING

from .airporchestrator_children import AirpDataReader, AirpDataProcessor, AirpFeatureBuilder, AirpNormalizer

if TYPE_CHECKING:
    from .airpconfig import AirpConfig
    from ..dataorchestration.dataorchestrator import DataOrchestrator

from ...utils.textformatting import checkmark

class AirpOrchestrator:
    """
    orchestrates the entire orchestration

    Parameters
    ----------
    airpconfig: 'AirpConfig'
    """        

    def __init__(self, airpconfig: 'AirpConfig', nutsdataorchestrator: 'DataOrchestrator'):
        self.airpconfig = airpconfig 
        self.nutsdataorchestrator = nutsdataorchestrator


    def load_raw_data(self):
        self._dataloader = AirpDataReader(self.airpconfig)
        self.data_raw    = self._dataloader.orchestrate()

    def process_raw_data(self):
        self._dataprocessor = AirpDataProcessor(self.airpconfig, self.data_raw, self.nutsdataorchestrator.data_context)
        self.data_processed, self.data_context = self._dataprocessor.orchestrate()

    def build_features(self):
        self._featurebuilder = AirpFeatureBuilder(self.airpconfig, self.data_processed)
        self.data_features = self._featurebuilder.orchestrate()

    def normalize_feature_data(self):
        self._datanormalizer = AirpNormalizer(self.airpconfig, self.data_features, self.data_context)
        self.data_normalized = self._datanormalizer.orchestrate()

    def build_orchestration(self):

        self.load_raw_data()
        self.process_raw_data()
        self.build_features()
        self.normalize_feature_data()
        
        return self
    
    def __repr__(self) -> str:
        representation = ("<AirpOrchestrator(")

        if hasattr(self, 'data_raw'):
            representation += f"data_raw {checkmark}, "
        if hasattr(self, "data_processed"):
            representation += f"data_processed {checkmark}, "    
        if hasattr(self, "data_context"):
            representation += f"data_context {checkmark}, "                 
        if hasattr(self, "data_features"):
            representation += f"data_features {checkmark}, "                        
        if hasattr(self, "data_normalized"):
            representation += f"data_normalized {checkmark}, "  

        # remove final comma and space
        representation = representation[:-2]+")>"

        return representation                    