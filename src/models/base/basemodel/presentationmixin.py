from typing import Dict

from ...utils import ModelStatus
from ....utils import checkmark, crossmark, section

class PresentationMixin:
    """ 
    Mixin class that deals with the model's (re)presentation.
    Contains dunder-methods `repr()` and `str()` as well as
    hidden methods for status-update printing.
    """

    name:           str
    status_dict:    Dict[ModelStatus, bool]
    model_class:    str 
    verbose:        int

    def _print_status_update(self, status: ModelStatus):
        """Print status update depending on `verbose`"""

        if self.verbose <= 0:
            return

        if status == "model_initialized" and self.verbose > 1:
            self._print_header()
        else:
            print(f"{status} {checkmark}")

    def _print_header(self):
        """Print formatted model header."""

        total_width     = 50
        title           = self.name

        inner_width     = total_width - 4
        centered_title  = title.center(inner_width)

        print(
            "\n"
            + "=" * total_width + "\n"
            + f"=={centered_title}==\n"
            + "=" * total_width
        )

    def __repr__(self) -> str:
        """minimal representation"""
        return f"{self.__class__.__name__}(name={self.name!r})"    

    def __str__(self) -> str:
        """extensive representation"""
        all_keys = (
            ['name', 'model_class'] + list(self.status_dict.keys())
        )

        width = max(len(k) for k in all_keys) if all_keys else 20
        
        # Build output
        lines = [f'<{self.__class__.__name__}(']
        lines.append('')
        general_items = {'name': self.name, 'model_class': self.model_class}
        lines.extend(section('generics', general_items, width))
        lines.append('')
        
        # Status section
        status_items = {k: f"{checkmark}" if v else f"{crossmark}" for k, v in self.status_dict.items()}
        lines.extend(section('status', status_items, width))
        lines.append('')
        
        lines.append(')>')
        
        return '\n'.join(lines)