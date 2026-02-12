from .epiconfig import EpiConfig
from .column_registry import ColumnRegistration
from .epidataorchestrator_children import EpiDataReader, Harmonizer, EpiDataProcessor, EpiFeatureBuilder, EpiNormalizer, EpiDataFinalizer
from .epidatacontainers import RawEpiData, HarmonizedData, ContextData, ProcessedEpiData, FeatureEpiData, NormalizedEpiData, FinalizedEpiData

# ====================================================
# ============ MAIN LOADER (ORCHESTRATOR) ============
# ====================================================
class DataOrchestrationContainerNotFound(Exception):
    def __init__(self, datastage: str, previous_method: str):
        super().__init__(f"No {datastage} attribute found for DataOrchestrator. Run {previous_method}() first")

class EpiDataOrchestrator:
   
    def __init__(self, config: 'EpiConfig'):
        self.config         = config

        # Initialize column registration
        self.column_registration = ColumnRegistration()
        self.column_registration.add_column(
            config.temporal_column, 
            'context'
        )
        self.column_registration.add_column(
            self.config.id_column, 
            'context'
        )   

        self.column_registration.add_column(
            'target', 
            'target',
            needs_normalization  = False if self.config.target_column == 'cases' else True,
            transformation_group = None
        )   

        # Store results at each stage
        self._data_raw       = None
        self._data_harmonized= None
        self._data_context   = None
        self._data_processed = None
        self._data_feature   = None
        self._data_normalized= None
        self._data_final     = None
          
    def load_raw(self) -> 'EpiDataOrchestrator':
        """Load raw data from files"""
        self.reader         = EpiDataReader(self.config)
        self._data_raw = self.reader.orchestrate()
        return self
    
    def harmonize_raw(self) -> 'EpiDataOrchestrator':
        """Harmonize data on NUTS-level"""     
        self.harmonizer     = Harmonizer(self.config)   
        self._data_harmonized, self._data_context = self.harmonizer.orchestrate(self.data_raw)        
        return self
    
    def process_data(self) -> 'EpiDataOrchestrator':
            """Preprocess the harmonized data"""
            self.processor = EpiDataProcessor(
                self.config, 
                self.data_context.temporal_summary
            )
            self._data_processed = self.processor.orchestrate(self.data_harmonized)
            return self

    def build_features(self) -> 'EpiDataOrchestrator':
        """build features. Note that this method adjusts self.column_registry."""
        self.feature_builder= EpiFeatureBuilder(self.config, self.column_registration, self.data_context.temporal_summary)
        self._data_feature = self.feature_builder.orchestrate(self.data_processed)
        return self        
   
    def normalize(self) -> 'EpiDataOrchestrator':
        """normalize data"""
        self.normalizer = EpiNormalizer(
            self.config, 
            self.column_registration,
            self.data_context.temporal_summary
        )
        self._data_normalized = self.normalizer.orchestrate(self.data_feature)
        return self      

    def finalize(self) -> 'EpiDataOrchestrator':
        """Finalize data."""
        self.finalizer      = EpiDataFinalizer(self.config, self.column_registration)
        self._data_final = self.finalizer.orchestrate(self.data_normalized)
        return self        

    def build(self) -> 'EpiDataOrchestrator':
        """Execute full pipeline and return final dataset."""
        return (self
            .load_raw()
            .harmonize_raw()
            .process_data()
            .build_features()
            .normalize()
            .finalize()
        )
    
    def __repr__(self):
        stages = []
        if self._data_raw is not None:
            stages.append("raw")
        if self._data_harmonized is not None:
            stages.append("harmonized")            
        if self._data_processed is not None:
            stages.append('processed')
        if self._data_feature is not None:
            stages.append('features') 
        if self._data_normalized is not None:
            stages.append('normalized')
        if self._data_final is not None:
            stages.append('finalized')            
        
        return f"<EpiDataOrchestrator(disease={self.config.disease}, data stages={stages})>"
    
    @property
    def data_raw(self) -> RawEpiData:
        if not self._data_raw:
            raise DataOrchestrationContainerNotFound(datastage = 'data_raw', previous_method = 'load_raw')
        return self._data_raw    
    
    @property
    def data_context(self) -> ContextData:
        if not self._data_context:
            raise DataOrchestrationContainerNotFound(datastage = 'data_context', previous_method = 'harmonize_raw')

        return self._data_context      

    @property
    def data_harmonized(self) -> HarmonizedData:
        if not self._data_harmonized:
            raise DataOrchestrationContainerNotFound(datastage = 'data_harmonized', previous_method = 'harmonize_raw')

        return self._data_harmonized        
           
    @property
    def data_processed(self) -> ProcessedEpiData:
        if not self._data_processed:
            raise DataOrchestrationContainerNotFound(datastage = 'data_processed', previous_method = 'harmonize_raw')

        return self._data_processed    

    @property 
    def data_feature(self) -> FeatureEpiData:
        if not self._data_feature:
            raise DataOrchestrationContainerNotFound(datastage = 'data_feature', previous_method = 'data_processed')        

        return self._data_feature    
    
    @property
    def data_normalized(self) -> NormalizedEpiData:
        if not self._data_normalized:
            raise DataOrchestrationContainerNotFound(datastage = 'data_normalized', previous_method = 'build_features')        

        return self._data_normalized    

    @property
    def data_final(self) -> FinalizedEpiData:
        if not self._data_final:
            raise DataOrchestrationContainerNotFound(datastage = 'data_final', previous_method = 'normalize')        

        return self._data_final         