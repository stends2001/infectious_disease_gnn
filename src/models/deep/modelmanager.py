import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Type
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

    def save(self, model: 'DeepModel', dir: Optional[Path] = None) -> None:
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
            'strategy':           model.strategy,
            'dataloadermanager':  model.dataloadermanager,
            'name':               model.name,
            'monitoring_metrics': model.monitoring_metrics,
        }

        torch.save(save_dict, filepath)

        # print relative path for readability
        try:
            local_path = str(filepath).split("/wissdaten/")[1]
            print(f"✓ Model saved: Wissdaten/{local_path}")
        except IndexError:
            print(f"✓ Model saved: {filepath}")

    # ── load ──────────────────────────────────────────────────────────────

    def load(self, model_name: str, sub_dir: Optional[Path] = None) -> 'DeepModel':
        """
        Load a trained model from disk.

        Parameters
        ----------
        model_name : str
            Filename without extension.
        model_class : optional
            Unused — kept for backward compatibility. Class is resolved
            from the saved dict via DeepModel._childclasses.

        Returns
        -------
        DeepModel
            Fully initialised instance with trained weights and correct status.
        """
        if sub_dir is None:
            filepath = self.base_dir / f"{model_name}.pt"
        else:
            filepath = self.base_dir / sub_dir / f"{model_name}.pt"            

        if not filepath.exists():
            raise FileNotFoundError(f"Model not found: {filepath}")

        # weights_only=False needed because we serialise the dataloadermanager
        # and strategy objects — these are trusted internal objects
        save_dict = torch.load(filepath, map_location='cpu', weights_only=False)

        from .deepmodel import DeepModel  # avoid circular import

        model_key = save_dict['model_class'].lower()
        if model_key not in DeepModel._childclasses:
            raise ValueError(
                f"Unknown model class '{save_dict['model_class']}'. "
                f"Available: {list(DeepModel._childclasses.keys())}"
            )

        cls = DeepModel._childclasses[model_key]

        instance = cls(
            name              = save_dict['name'],
            dataloadermanager = save_dict['dataloadermanager'],
        )

        # restores model_hparams_set and global_hparams_set status flags
        instance.set_model_hparams(**save_dict['model_hparams'])
        instance.set_global_hparams(**save_dict['global_hparams'])

        # restore weights
        instance.model.load_state_dict(save_dict['model_state'])
        instance.model.to(instance.device)

        # restore metadata
        instance.monitoring_metrics           = save_dict.get('monitoring_metrics')
        instance.config_info['model_hparams'] = save_dict['model_hparams']
        instance.config_info['global_hparams']= save_dict['global_hparams']

        # trained=True, forecasted stays False until caller runs forecast()
        instance._update_status('trained')
        return instance