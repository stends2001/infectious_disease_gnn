import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Type, Any
from typing import TypeVar

T = TypeVar('T', bound='DeepModel')
import torch

from ...utils.helpers import get_project_utilities_env

if TYPE_CHECKING:
    from .deepmodel import DeepModel


class ModelManager:
    """Simple manager for saving and loading trained DeepModel instances."""

    def __init__(self, dir: str = 'models'):
        self.base_dir = Path(os.path.join(get_project_utilities_env(), dir))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── save ──────────────────────────────────────────────────────────────

    def save(self, model: 'DeepModel', dir: Optional[Path], minimal: bool) -> None:
        """Save a trained model to disk."""
        if model.model is None:
            raise ValueError('No model to save — call set_model_hparams() first.')

        base_dir    = self.base_dir 
        sub_dir     = dir

        if not base_dir.exists():
            raise FileNotFoundError(f".base_dir {base_dir} does not exist")

        if sub_dir is not None:
            full_sub_dir = base_dir / sub_dir
            full_sub_dir.mkdir(exist_ok=True)  # creates if not exists, errors if base_dir missing
            filepath = full_sub_dir / f"{model.name}.pt"
        else:
            filepath = base_dir / f"{model.name}.pt"

        save_dict = {
            'model_class':        model.__class__.__name__,
            'model_state':        model.model.state_dict(),
            'model_hparams':      model.config_info.get('model_hparams', {}),
            'global_hparams':     model.config_info.get('global_hparams', {}),
            'name':               model.name,
            'monitoring_metrics': model.monitoring_metrics,
        }

        if not minimal:
            save_dict['strategy']          = model.strategy
            save_dict['dataloadermanager'] = model.dataloadermanager      

        torch.save(save_dict, filepath)

        # print relative path for readability
        try:
            local_path = str(filepath).split("/wissdaten/")[1]
            print(f"✓ Model saved: Wissdaten/{local_path}")
        except IndexError:
            print(f"✓ Model saved: {filepath}")

    # ── load ──────────────────────────────────────────────────────────────

    def load(self, 
            model_name:        str, 
            dataloadermanager: Optional[Any] = None,
            subdir:            Optional[str] = None) -> 'DeepModel':

        base     = self.base_dir / subdir if subdir else self.base_dir
        filepath = base / f"{model_name}.pt"

        if not filepath.exists():
            raise FileNotFoundError(f"Model not found: {filepath}")

        # determine if minimal save
        # try weights_only first, fall back if it fails
        try:
            save_dict = torch.load(filepath, map_location='cpu', weights_only=True)
            is_minimal = True
        except Exception:
            save_dict  = torch.load(filepath, map_location='cpu', weights_only=False)
            is_minimal = False

        from .deepmodel import DeepModel

        model_key = save_dict['model_class'].lower()
        if model_key not in DeepModel._childclasses:
            raise ValueError(
                f"Unknown model class '{save_dict['model_class']}'. "
                f"Available: {list(DeepModel._childclasses.keys())}"
            )

        cls = DeepModel._childclasses[model_key]

        # resolve dataloader — prefer supplied, fall back to saved
        if dataloadermanager is not None:
            dlm = dataloadermanager
        elif not is_minimal and save_dict.get('dataloadermanager') is not None:
            dlm = save_dict['dataloadermanager']
        else:
            raise ValueError(
                f"Model '{model_name}' was saved in minimal format. "
                f"Supply a dataloadermanager to load()."
            )

        instance = cls(name=save_dict['name'], dataloadermanager=dlm)

        instance.set_model_hparams(**save_dict['model_hparams'])
        instance.set_global_hparams(**save_dict['global_hparams'])
        instance.model.load_state_dict(save_dict['model_state'])
        instance.model.to(instance.device)
        instance.monitoring_metrics           = save_dict.get('monitoring_metrics')
        instance.config_info['model_hparams'] = save_dict['model_hparams']
        instance.config_info['global_hparams']= save_dict['global_hparams']
        instance._update_status('trained')

        print(f"✓ Model loaded: {filepath.name}")
        return instance