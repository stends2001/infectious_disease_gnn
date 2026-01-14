import torch
import pickle
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .deepmodel import DeepModel

class ModelManager:
    """Simple manager for saving and loading trained models."""
    
    def __init__(self, base_dir: str = 'saved_models'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, model: 'DeepModel', filename: Optional[str] = None) -> str:
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
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{model.name}_{timestamp}"
        
        # Remove extension if provided
        filename = filename.replace('.pt', '')
        filepath = self.base_dir / f"{filename}.pt"
        
        # Package everything needed
        save_dict = {
            'model_class': model.__class__.__name__,
            'model_state': model.model.state_dict(),
            'model_hparams': model.config_info.get('model_hparams', {}),
            'global_hparams': model.config_info.get('global_hparams', {}),
            'dataloadermanager': model.dataloadermanager,
            'name': model.name,
            'monitoring_metrics': model.monitoring_metrics
        }
        
        torch.save(save_dict, filepath)
        print(f"✓ Model saved: {filepath}")
        return str(filepath)
    
    def load(self, filepath: str, model_class: type) -> 'DeepModel':
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
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model not found: {filepath}")
        
        # Load saved data
        save_dict = torch.load(filepath, map_location='cpu')
        
        # Check model class matches
        if save_dict['model_class'] != model_class.__name__:
            raise ValueError(
                f"Model class mismatch: file contains {save_dict['model_class']}, "
                f"but trying to load as {model_class.__name__}"
            )
        
        # Create new model instance
        model = model_class(
            dataloadermanager=save_dict['dataloadermanager'],
            name=save_dict['name']
        )
        
        # Restore architecture
        model.set_model_hparams(**save_dict['model_hparams'])
        
        # Restore training config (but don't train)
        model.set_global_hparams(**save_dict['global_hparams'])
        
        # Load weights
        model.model.load_state_dict(save_dict['model_state'])
        model.model.to(model.device)
        
        # Restore training history
        model.monitoring_metrics = save_dict.get('monitoring_metrics')
        
        # Mark as trained
        model._update_status('trained')
        
        print(f"✓ Model loaded: {filepath}")
        return model