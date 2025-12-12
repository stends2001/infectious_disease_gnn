from typing import Literal, Optional
from ..dataloading import DataOrchestrator, GraphDataLoaderManager, ShallowDataLoaderManager
from ..models import MODELSREGISTRY
from ..models.base.basemodel import BaseModel
from ..evaluation import Evaluator

class ExperimentRunner:

    """
    class to run Experiments with
    add models using .add_baseline_model(), .add_shallow_model()
    or .add_gnn(), and go for .run()

    Parameters
    ----------
    data_orchestrator: DataOrchestrator
        object of finalized and processed EpiConfig

    Returns
    -------
    None, attributes .evaluation and .models are set    

    Examples
    --------
    >>> runner = (ExperimentRunner(data_orchestrator)
    >>>    .add_baseline_model('PersistenceModel', 'persistence')
    >>>    .add_baseline_model('ClimateologyModel', 'climateology')    
    >>>    .add_shallow_model('NodeRFModel', 'node_rf', model_hparams={'n_estimators': 100},                                       train_kwargs={'verbose': 0})
    >>>    .add_gnn('SpatialGNNModel', 'gcn-identity_graph',    'identity_graph',                 global_hparams=global_hparams,   train_kwargs={'verbose': 0, 'show_loss': False})
    >>>    .add_gnn('SpatialGNNModel', 'gcn-mesh_graph',        'mesh_graph',                     global_hparams=global_hparams,   train_kwargs={'verbose': 0, 'show_loss': False})        
    >>>    .add_gnn('SpatialGNNModel', 'gcn-commuter14_2',      'static_commuter14_2',            global_hparams=global_hparams,   train_kwargs={'verbose': 0, 'show_loss': False})        
    >>>    .add_gnn('SpatialGNNModel', 'gcn-commuter24_2',      'static_commuter24_2',            global_hparams=global_hparams,   train_kwargs={'verbose': 0, 'show_loss': False})                      
    >>>    .add_gnn('SpatialGNNModel', 'gcn-dynamic_commuter2', 'dynamic_commuter2','dynamic', data_kwargs={'timesteps': range(2006,2021)} , global_hparams=global_hparams, train_kwargs={'verbose': 0, 'show_loss': False})                     
    >>>    # GATVs
    >>>    .add_gnn('GATv2Model',      'gatv2-identity_graph', 'identity_graph',                 global_hparams=global_hparams,            train_kwargs={'verbose': 0, 'show_loss': False})
    >>>    .add_gnn('GATv2Model',      'gatv2-mesh_graph',     'mesh_graph',                     global_hparams=global_hparams,            train_kwargs={'verbose': 0, 'show_loss': False})    
    >>>    .add_gnn('GATv2Model',      'gatv2-commuter14_2',   'static_commuter14_2',                     global_hparams=global_hparams,   train_kwargs={'verbose': 0, 'show_loss': False})    
    >>>    .add_gnn('GATv2Model',      'gatv2-commuter24_2',   'static_commuter24_2',                     global_hparams=global_hparams,   train_kwargs={'verbose': 0, 'show_loss': False})        
    >>>    .add_gnn('GATv2Model',      'gatv2-dynamic_commuter2',     'dynamic_commuter2', data_kwargs={'timesteps': range(2006,2021)} , global_hparams=global_hparams, train_kwargs={'verbose': 0, 'show_loss': False})                     
    >>>     )

    >>> results = runner.run()    
    """


    def __init__(self, data_orchestrator: DataOrchestrator):
        self.data_orchestrator = data_orchestrator
        self.models = []
        self.graph_data = {}
        self.shallow_data = None
    
    def add_baseline_model(self, model_class, name, **kwargs):
        """Add baseline model to the experiment"""
        self.models.append({
            'class': model_class,
            'name': name,
            'model_type': 'baseline',
            'data_type': 'shallow',
            'kwargs': kwargs
        })
        return self

    def add_shallow_model(self, model_class, name, **kwargs):
        """Add shallow model to the experiment"""
        self.models.append({
            'class': model_class,
            'name': name,
            'model_type': 'shallow',
            'data_type': 'shallow',
            'kwargs': kwargs
        })
        return self
    
    def add_gnn(self, model_class, name, graph, graph_mode: Literal['static','dynamic'] = 'static', data_kwargs: Optional[dict] = None, **kwargs):
        """
        add GNN to experiment
        
        Examples
        --------
        >>> .add_gnn('GATv2Model', 
                    'gatv2-commuter24_2', 
                    'static_commuter24_2',
                    global_hparams=global_hparams,
                    train_kwargs = {'verbose': 0, 'show_loss': False}
                    )        
        >>> .add_gnn('GATv2Model',
                    'gatv2-dynamic_commuter2',
                    'dynamic_commuter2',
                    data_kwargs={'timesteps': range(2006,2021)},
                    global_hparams=global_hparams,
                    train_kwargs={'verbose': 0, 'show_loss': False}
                    )          
        """
        self.models.append({
            'class': model_class,
            'name': name,
            'model_type': 'deep',
            'graph_mode': graph_mode,
            'graph_name': graph,
            'data_type': 'deepgraph',
            'data_kwargs': data_kwargs,
            'kwargs': kwargs
        })
        return self
    
    def run(self):
        """run an experiment"""
        models_dictionary = {}
        for model_spec in self.models:
            # Get data
            if model_spec['data_type'] == 'shallow':
                data = self._get_shallow()
            elif model_spec['data_type'] == 'deepgraph':

                data = self._get_graph(model_spec['graph_name'], model_spec['graph_mode'], model_spec['data_kwargs'])
          
            self._execute_experiment_modeltype(model_spec, data, models_dictionary)
        
        self.evaluation = Evaluator(list(models_dictionary.values()))
        self.evaluation.add_evaluation(0,'test')
        self.models_dictionary = models_dictionary
    
    def _get_shallow(self) -> ShallowDataLoaderManager:
        """return ShallowDataLoaderManager"""
        if self.shallow_data is None:
            self.shallow_data = ShallowDataLoaderManager(self.data_orchestrator).construct_dataloaders()
        return self.shallow_data
    
    def _execute_experiment_modeltype(self, model_spec, data, results) -> None:
        """for a specific model, run the pipeline"""
        ModelClass = MODELSREGISTRY[model_spec['class']]
        model: BaseModel = ModelClass(data, name=model_spec['name'])
        kwargs = model_spec['kwargs']
        
        # Set hyperparameters (order depends on model type)
        if model_spec['model_type'] == 'shallow':
            model.set_global_hparams(**kwargs.get('global_hparams', {}))
            model.set_model_hparams(**kwargs.get('model_hparams', {}))
        elif model_spec['model_type'] == 'deep':
            model.set_model_hparams(**kwargs.get('model_hparams', {}))
            model.set_global_hparams(**kwargs.get('global_hparams', {}))
        
        # Train non-baseline models
        if model_spec['model_type'] != 'baseline':
            model.train(**kwargs.get('train_kwargs', {}))
        
        # Forecast
        results[model_spec['name']] = model.forecast('test')  

    def _get_graph(self, graph, mode: Literal['static','dynamic'], data_kwargs):
        """return GraphDataLoaderManager"""
        if graph not in self.graph_data:
            if mode == 'static':
                self.graph_data[graph] = (GraphDataLoaderManager(self.data_orchestrator)
                                            .retrieve_static_graph(graph)
                                            .construct_dataloaders())
            elif mode == 'dynamic':
                self.graph_data[graph] = (GraphDataLoaderManager(self.data_orchestrator)
                                            .retrieve_dynamic_graph(graph,**data_kwargs)
                                            .construct_dataloaders())                
        return self.graph_data[graph]
