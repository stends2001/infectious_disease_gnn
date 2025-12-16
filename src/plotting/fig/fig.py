from typing import Optional
import copy

import matplotlib

from .managers import LegendManager, TickManager, LabelManager

def returns_fig(func):
    """Decorator that wraps matplotlib figures in Fig class"""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, matplotlib.figure.Figure):
            return Fig(result)
        return result
    return wrapper

class Fig:
    """Main Figure Wrapper"""

    def __init__(self, matplotlib_fig: matplotlib.figure.Figure):
        # Store the matplotlib figure object
        self.mpl_figure = copy.deepcopy(matplotlib_fig)

        if len(self.mpl_figure.axes) != 1:
            raise ValueError("currently no subplots supported")

        # Store the single axis for convenience
        self.mpl_axis = self.mpl_figure.axes[0]
        
        # Create manager instances
        self.legend = LegendManager(self)
        self.ticks = TickManager(self)
        self.labels = LabelManager(self)

    def change_figsize(self, width, height) -> 'Fig':
        self.mpl_figure.set_size_inches(width, height)
        return self
    
    def show(self) -> matplotlib.figure.Figure:
        return self.mpl_figure
    
    def __repr__(self):
        return "<Fig(legend, ticks, labels)>"
