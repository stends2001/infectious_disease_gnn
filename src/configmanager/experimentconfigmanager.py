
from typing import List, Dict, Optional, Union, Literal
import yaml
import os
import json
from pathlib import Path
from ._baseconfigmanager import ConfigManager
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..models._basemodel import BaseModel

from ..utils.helpers import list_files, write_yaml_file, to_underscore_string,reorder_dict, get_data_env
from ..dataloading.gnndataloader import GNNDataLoader
from ..models.a3tgcn import A3TGCNModel
from ..models.spatialgcn_baseline import SpatialGCNModel
from ..models.spatiotemporal_baseline import TGCNModel

class ExperimentConfigManager(ConfigManager):
    """
    Manages configurations of:

    - experiments
    """

    def __init__(self, base_dir: str = "config/experiments/"):
        self.base_dir  = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)   
        self.experiment_log = {}

        # self.default_config_order   = ['name', 'id', 'child', 'model', 
                                    #    'global_hparams', 'model_hparams', 'task'] 
    def define(self,
               experiment_name:    str,
               disease_name:       str,
               min_date:           str                              = '2001-01-01',
               max_date:           str                              = '2020-06-01',
               nuts_level:         Literal['nuts1','nuts2','nuts3'] = 'nuts2',
               include_population: bool                             = False,
               sequence_length:    int                              = 12,
               horizon_size:       int                              = 1,
               horizon_leadtime:   int                              = 3,
               lags              : int                            = 1,
               log_transform_target:      bool                             = True,
               split_trainval:     str                              = '2018-06-01', 
               split_valtest:             str                       = '2019-06-01',
               add_time_features:         bool = True,
               normalization_method: Literal['minmax','zscore'] = 'zscore'  ,
               graphs: Union[List[str], str]  = 'identity_graph',
               model: Union[List[str], str]  = 'spatialgcn',
               global_hparams: dict = {},
               model_hparams: dict = {},
               train_hparams: dict ={},
               ) -> 'ExperimentConfigManager':
        """
        define an experiment
        """
        experiment_config = {
            'name'                  : experiment_name,
            'disease_name'          : disease_name,
            'nuts_level'            : nuts_level,
            'min_date'              : min_date,
            'max_date'              : max_date,
            'include_population'    : include_population,
            'sequence_length'       : sequence_length,
            'horizon_size'          : horizon_size,
            'horizon_leadtime'      : horizon_leadtime,
            'lags'                  : lags,
            'log_transform_target'  : log_transform_target,
            'add_time_features'     : add_time_features,
            'split_trainval'        : split_trainval, 
            'split_valtest'         : split_valtest,
            'normalization_method'  : normalization_method,
            'graphs'                : graphs,
            'model'                 : model,
            'global_hparams'        : global_hparams,
            'model_hparams'         : model_hparams,
            'train_hparams'         : train_hparams
        }
            
        if experiment_name in self.experiment_log.keys():
            raise KeyError(f'an experiment under the name {experiment_name} has been found in the log. Please ensure no double naming.')
        
        self.experiment_log[experiment_name] = experiment_config
        return self

    def execute(self, experiment_name: str):
        """
        execute an experiment
        """
        if experiment_name not in self.experiment_log.keys():
            raise KeyError(f'no experiment {experiment_name} found in experiment_log')
        
        else:
            experiment_config = self.experiment_log[experiment_name]

        # section 1: get epi-dataloader
        epidata = GNNDataLoader(disease_name        = experiment_config['disease_name'],
                                data_env_dir        = get_data_env() ,
                                min_date            = experiment_config['min_date'],
                                max_date            = experiment_config['max_date'],
                                nuts_level          = experiment_config['nuts_level'],
                                include_population  = experiment_config['include_population'],
                                sequence_length     = experiment_config['sequence_length'],
                                horizon_size        = experiment_config['horizon_size'],
                                horizon_leadtime    = experiment_config['horizon_leadtime'])
        
        if experiment_config['add_time_features']:
            epidata.add_time_features()
        if experiment_config['log_transform_target']:        
            epidata.log_transform_target()

        epidata.set_splits(split_trainval= experiment_config['split_trainval'],
                           split_valtest = experiment_config['split_valtest'])
        
        epidata.normalize(normalization_method= experiment_config['normalization_method'])
        epidata.add_lagged_features(lags = experiment_config['lags'])
        epidata.finalize()

        # section 2: get gnn-dataloaders
        graphs = experiment_config['graphs']
        if isinstance(graphs, str):
            graphs = [graphs]

        gnn_datasets = {}

        for graph in graphs:
            gnn_datasets[graph]  = epidata.copy(deep=True).retrieve_graph(graphname = graph).construct_dataloaders()

        self.gnn_datasets = gnn_datasets

        # models:
        models_dict = {}
        model = experiment_config['model']
        
        if model == 'SpatialGCN'.lower():

            for graph, dataset in gnn_datasets.items():
                ml_instance = SpatialGCNModel(name = f'experiment-{experiment_name}-spatialgcn-{graph}', dataloader=dataset)
                models_dict[graph] = ml_instance
        
        elif model == 'TGCN'.lower():
            for graph, dataset in gnn_datasets.items():
                ml_instance = TGCNModel(name = f'experiment-{experiment_name}-tgcn-{graph}', dataloader=dataset)
                models_dict[graph] = ml_instance            

        elif model == 'A3TGCN'.lower():
            for graph, dataset in gnn_datasets.items():
                ml_instance = A3TGCNModel(name = f'experiment-{experiment_name}-a3tgcn-{graph}', dataloader=dataset)
                models_dict[graph] = ml_instance        

        else:
            print('model not found')    

        for graph, ml_instance in models_dict.items():
            ml_instance.set_model_hparams(**experiment_config['model_hparams'])
            ml_instance.set_global_hparams(**experiment_config['global_hparams'])
            ml_instance.run_snapshot(debug=True)
            ml_instance.train(**experiment_config['train_hparams'])
            ml_instance.forecast()
            ml_instance.show_forecasts(dataset='test', target_h = 0)
            models_dict[graph] = ml_instance
                
        self.models_dict = models_dict


    def save_experiment(self, experiment_name: str):
        for _, ml_instance in self.models_dict.items():
            ml_instance.save_model()
            ml_instance.save_weights()