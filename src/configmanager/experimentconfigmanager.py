from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, List, Union

from src.utils import get_data_env
from src.dataloading import DeepDataLoader
from src.configmanager._baseconfigmanager import ConfigManager
from src.models import MODELSREGISTRY
from src.models import PersistenceModel, GATv2Model, GConvLSTMModel, TGCNModel, NodeRFModel, SpatialGNNModel

from src.evaluation import Evaluator

@dataclass
class DataConfig:
    disease_name:       str
    nuts_level:         Literal['nuts1','nuts2','nuts3'] = 'nuts2'
    min_date:           str = '2001-01-01'
    max_date:           str = '2020-06-01'
    split_berlin:       bool = False
    include_population: bool = False
    
@dataclass
class TimeSeriesConfig:
    sequence_length:    int = 12
    horizon_size:       int = 1
    horizon_leadtime:   int = 3
    lags:               int = 1
    
@dataclass
class PreprocessingConfig:
    log_transform_target:   bool = True
    add_time_features:      bool = True
    normalization_method:   Literal['minmax','zscore'] = 'zscore'
    
@dataclass
class SplitConfig:
    split_trainval: str = '2018-06-01'
    split_valtest:  str = '2019-06-01'

@dataclass
class ExperimentConfig:
    name:           str
    data:           DataConfig
    timeseries:     TimeSeriesConfig
    preprocessing:  PreprocessingConfig
    splits:         SplitConfig
    graphs:         Union[List[str], str]
    model:          Union[List[str], str] = 'spatialgcn'
    baseline:       bool                  = True
    global_hparams: dict = field(default_factory=dict)
    model_hparams:  dict = field(default_factory=dict)
    train_hparams:  dict = field(default_factory=dict)

    def __post_init__(self):

        if isinstance(self.model, str):
            self.models_list = [self.model]
        else:
            self.models_list = self.model

        # Validate each model in the list
        for model in self.models_list:  # Changed from self.model to self.models_list
            # Normalize model name to lowercase for comparison
            model_name = model.lower()  # Simplified - model is already a string here
            
            # Create a case-insensitive lookup
            registry_lower = {k.lower(): k for k in MODELSREGISTRY.keys()}
            
            # Check if model exists in registry (case-insensitive)
            if model_name not in registry_lower:
                available = ', '.join(sorted(MODELSREGISTRY.keys()))
                raise ValueError(
                    f"Model '{model}' not found in registry. "  # Changed from self.model to model
                    f"Available models: {available}"
                )

class ExperimentRunner(ConfigManager):
    """
    Manages configurations of:

    - experiments
    """

    def __init__(self, base_dir: str = "config/experiments/"):
        self.base_dir       = Path(base_dir)
        self.experiment_log = {}
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def define(self, config: ExperimentConfig) -> 'ExperimentRunner':
        """
        Define an experiment using an ExperimentConfig object
        """
        if config.name in self.experiment_log:
            raise KeyError(
                f'An experiment under the name {config.name} has been found in the log. '
                f'Please ensure no double naming.'
            )
        
        self.experiment_log[config.name] = config
        return self
    def execute(self, experiment_name: str):
        """
        Execute an experiment
        """
        if experiment_name not in self.experiment_log:
            raise KeyError(f'No experiment {experiment_name} found in experiment_log')
        
        config = self.experiment_log[experiment_name]

        # Section 1: Get epi-dataloader
        epidata = self._prepare_data(config)

        # Section 2: Get gnn-dataloaders
        gnn_datasets = self._prepare_datasets(epidata, config)
        self.gnn_datasets = gnn_datasets

        # Section 3: run baseline model
        persistence_baseline = PersistenceModel(epidata, name='baseline')
        persistence_baseline.forecast('test')

        # Section 4: Models - iterate over all models in the list
        models_dict = {}
        
        for model_name in config.models_list:  # Iterate over the list of models
            # Get the model class from registry
            ModelClass = MODELSREGISTRY[model_name]
            
            for graph, dataset in gnn_datasets.items():
                # Create unique name combining model and graph
                instance_name = f'{model_name}_{graph}'
                
                ml_instance = ModelClass(
                    name=instance_name,
                    dataloader=dataset
                )
                models_dict[instance_name] = ml_instance  

        # Section 5: Train and evaluate
        for instance_name, ml_instance in models_dict.items():
            if instance_name == 'persistence_baseline':
                continue
                
            ml_instance.set_model_hparams(**config.model_hparams)
            ml_instance.set_global_hparams(**config.global_hparams)
            ml_instance.train(**config.train_hparams)
            ml_instance.forecast()
            # ml_instance.show_forecasts(dataset='test', target_h=0)
        
        models_dict['persistence_baseline'] = persistence_baseline
        self.models_dict = models_dict

        self._evaluate() 
        return self.evaluation    
    def _prepare_data(self, config: ExperimentConfig) -> DeepDataLoader:
        """Handle all data loading and preprocessing"""
        epidata = DeepDataLoader(
            disease_name=config.data.disease_name,
            data_env_dir=get_data_env(),
            min_date=config.data.min_date,
            max_date=config.data.max_date,
            nuts_level=config.data.nuts_level,
            include_population=config.data.include_population,
            sequence_length=config.timeseries.sequence_length,
            horizon_size=config.timeseries.horizon_size,
            horizon_leadtime=config.timeseries.horizon_leadtime,
            split_berlin=config.data.split_berlin
        )
        
        if config.preprocessing.add_time_features:
            epidata.add_time_features()
        if config.preprocessing.log_transform_target:
            epidata.log_transform_target()

        epidata.set_splits(
            split_trainval=config.splits.split_trainval,
            split_valtest=config.splits.split_valtest
        )
        
        epidata.normalize(normalization_method=config.preprocessing.normalization_method)
        epidata.add_lagged_features(lags=config.timeseries.lags)
        epidata.finalize()
        
        return epidata

    def _evaluate(self):
        self.evaluation = Evaluator(list(self.models_dict.values()))      

    def _prepare_datasets(self, epidata: DeepDataLoader, config: ExperimentConfig) -> dict:
        """Prepare GNN datasets for all graphs"""
        graphs = config.graphs if isinstance(config.graphs, list) else [config.graphs]
        
        gnn_datasets = {}
        for graph in graphs:
            gnn_datasets[graph] = (
                epidata.copy(deep=True)
                .retrieve_graph(graphname=graph)
                .construct_dataloaders()
            )
        
        return gnn_datasets