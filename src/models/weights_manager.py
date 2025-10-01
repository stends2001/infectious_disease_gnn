from pathlib import Path 
from typing import Optional, Dict, List, Union

from datetime import datetime
import os
import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._deepmodel import DeepModel

from ..utils.helpers import to_underscore_string

class ModelWeightsManager:
    """
    Manages model weights (PyTorch state dicts) separately from configs.
    Weights are stored as .pt files.
    """
    
    def __init__(self, base_dir: str = "config/models/checkpoints/"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_weights(self,
                     model: 'DeepModel',
                     filename: Optional[str] = None,
                     save_optimizer: bool = True,
                     save_scheduler: bool = True,
                     metadata: Optional[Dict] = None) -> str:
        """
        Save model weights and training state.
        
        Parameters:
        ----------
        model : DeepModel
            The model to save
        filename : Optional[str]
            Custom filename (without extension)
        save_optimizer : bool
            Save optimizer state for training resumption
        save_scheduler : bool
            Save scheduler state for training resumption
        metadata : Optional[Dict]
            Additional metadata to store with weights
            
        Returns:
        -------
        str : Path to saved weights file
        """
        if model.model is None:
            raise ValueError('No model to save')
        
        try:
            model_id = model.config_info['id']
        except KeyError:
            raise KeyError("No model ID found in the config. Save the model before the weights!")

        # Generate filename
        if filename is None:
            filename = f"{model_id}"
        
        filename = to_underscore_string(filename)
        
        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': model.model.state_dict(),
            'model_class': model.__class__.__name__,
            'config_id': model.config_info.get('id'),
            'config_name': model.config_info.get('name'),
            'saved_timestamp': datetime.now().isoformat(),
            'train_losses': getattr(model, 'train_losses', []),
            'val_losses': getattr(model, 'val_losses', []),
            'test_loss': getattr(model, 'test_loss', None),
            'state': model._state.copy(),
            'metadata': metadata or {}
        }
        
        # Add optimizer/scheduler if requested
        if save_optimizer and model.optimizer is not None:
            checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()
        
        if save_scheduler and model.scheduler is not None:
            checkpoint['scheduler_state_dict'] = model.scheduler.state_dict()
        
        # Save weights
        weights_path = self.base_dir / f"{filename}.pt"
        torch.save(checkpoint, weights_path)
        
        print(f"✓ Weights saved: {weights_path}")
        return str(weights_path)
    
    def load_weights(self,
                     model: 'DeepModel',
                     model_number: int,
                     load_optimizer: bool = False,
                     load_scheduler: bool = False) -> Dict:
        """
        Load model weights and return checkpoint metadata.
        
        Parameters:
        ----------
        model : DeepModel
            Model instance to load weights into
        weights_path : str
            Path to weights file
        load_optimizer : bool
            Load optimizer state
        load_scheduler : bool
            Load scheduler state
        strict : bool
            Strictly enforce key matching
            
        Returns:
        -------
        dict : Checkpoint metadata
        """
        weights_path = Path(os.path.join(self.base_dir, f'{model_number}.pt'))
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=model.device)
        
        # Validate
        saved_class = checkpoint.get('model_class')
        if saved_class and saved_class != model.__class__.__name__:
            print(f"⚠️  Warning: Loading {saved_class} weights into {model.__class__.__name__}")
        
        # Load model weights
        if model.model is None:
            raise ValueError('Model not initialized. Call set_model_hparams() first.')
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history
        if 'train_losses' in checkpoint:
            model.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            model.val_losses = checkpoint['val_losses']
        if 'test_loss' in checkpoint:
            model.test_loss = checkpoint['test_loss']
        if 'state' in checkpoint:
            model._state.update(checkpoint['state'])
        
        # print(f"✓ Weights loaded from: {weights_path}")
        
        # Return metadata
        return {
            'config_id': checkpoint.get('config_id'),
            'config_name': checkpoint.get('config_name'),
            'saved_timestamp': checkpoint.get('saved_timestamp'),
            'metadata': checkpoint.get('metadata', {})
        }
    
    def list_weights(self) -> List[Dict]:
        """List all saved weight files with metadata"""
        weights_list = []
        
        for weights_file in self.base_dir.glob("*.pt"):
            try:
                checkpoint = torch.load(weights_file, map_location='cpu')
                weights_list.append({
                    'filename': weights_file.stem,
                    'path': str(weights_file),
                    'model_class': checkpoint.get('model_class'),
                    'config_id': checkpoint.get('config_id'),
                    'config_name': checkpoint.get('config_name'),
                    'saved_timestamp': checkpoint.get('saved_timestamp'),
                    'has_optimizer': 'optimizer_state_dict' in checkpoint,
                    'has_scheduler': 'scheduler_state_dict' in checkpoint
                })
            except Exception as e:
                print(f"⚠️  Error loading {weights_file}: {e}")
        
        return weights_list
    
    def delete_weights(self, weights_path: Union[str, Path]):
        """Delete a weights file"""
        weights_path = Path(weights_path)
        if weights_path.exists():
            weights_path.unlink()
            print(f"✓ Weights deleted: {weights_path}")
