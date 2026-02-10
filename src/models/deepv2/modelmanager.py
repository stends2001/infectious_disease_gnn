import torch
import pickle
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Type
from datetime import datetime

import os

if TYPE_CHECKING:
    from .deepmodel import DeepModel

from ...utils.helpers import get_project_utilities_env

class ModelManager:
    """Simple manager for saving and loading trained models."""
    
    def __init__(self, dir: str = 'models'):
        self.base_dir = Path(os.path.join(get_project_utilities_env(), dir))
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, model: 'DeepModel', filename: Optional[str] = None) -> None:
        """
        Save a trained model.
        
        Parameters
        ----------
        model : DeepModel
            The trained model to save
        filename : Optional[str]
            Custom filename (without extension). If None, auto-generates one.
            
        Returns
        -------
        str : Path where model was saved
        """
        if model.model is None:
            raise ValueError('No model to save')
        
        # Generate filename if not provided
        filename = model.name
        
        # Remove extension if provided
        filename = filename.replace('.pt', '')
        filepath = self.base_dir / f"{filename}.pt"
        
        # Package everything needed
        save_dict = {
            'model_class':      model.__class__.__name__,
            'model_state':      model.model.state_dict(),
            'model_hparams':        model.config_info.get('model_hparams', {}),
            'global_hparams':       model.config_info.get('global_hparams', {}),
            'strategy'      :       model.strategy,
            'deepfamily'    :       model.deepfamily,
            'dataloadermanager':    model.dataloadermanager,
            'name':                 model.name,
            'monitoring_metrics':   model.monitoring_metrics
        }
        
        torch.save(save_dict, filepath)
        
        # filepath is this thing including the local/job_number/ ... 
        # which we can remove for printing clarity
        local_path_wissdaten = str(filepath).split("/wissdaten/")[1]
        path_return          = 'Wissdaten/' + local_path_wissdaten
        print(f"✓ Model saved: {path_return}")
        
    
    def load(self, model_name: str, model_class: type) -> Type['DeepModel']:
        """
        Load a trained model.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model file
        model_class : type
            The model class to instantiate (e.g., MyGNN)
            
        Returns
        -------
        DeepModel : The loaded model
        """
        print('loading model initiated')

        filepath = os.path.join(self.base_dir, f"{model_name}.pt")

        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model not found: {filepath}")
        print('model path found')
        
        # Load saved data
        save_dict = torch.load(filepath, map_location='cpu')
        print('save dict loaded')
        
        from .deepmodel import DeepModel  # runtime import, avoids circular dependency

        if save_dict['model_class'].lower() in DeepModel._childclasses:
   
            model_instance = DeepModel._childclasses[save_dict['model_class'].lower()](name = save_dict['name']+"_loaded", dataloadermanager =  save_dict['dataloadermanager'])
            model_instance.set_model_hparams(**save_dict['model_hparams'])
            model_instance.set_global_hparams(**save_dict['global_hparams'])   
            model_instance.model.load_state_dict(save_dict['model_state'])
            model_instance.model.to(model_instance.device)                
            model_instance.monitoring_metrics = save_dict.get('monitoring_metrics') 
            model_instance._update_status('trained')
        else:
            raise ValueError('invalid model class trying to be loaded')


        return model_instance

        # # Restore architecture
        # model.set_model_hparams(**save_dict['model_hparams'])
        
        # # Restore training config (but don't train)
        # model.set_global_hparams(**save_dict['global_hparams'])
        
        # # Load weights
        # model.model.load_state_dict(save_dict['model_state'])
        # model.model.to(model.device)
        
        # # Restore training history
        # model.monitoring_metrics = save_dict.get('monitoring_metrics')
        
        # # Mark as trained
        # model._update_status('trained')
        
        # print(f"✓ Model loaded: {filepath}")
        # return model