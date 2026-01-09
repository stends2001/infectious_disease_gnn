from typing import Optional
import copy

import matplotlib

from .managers import LegendManager, TickManager, LabelManager, LayoutManager

import matplotlib.pyplot as plt 

from typing import Optional
import copy

import matplotlib
import matplotlib.figure 


import pickle

def clone_figure(fig: matplotlib.figure.Figure) -> matplotlib.figure.Figure:
    return pickle.loads(pickle.dumps(fig))

def convert_managedfigure(func):
    """Decorator that wraps matplotlib figures in Fig class"""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, matplotlib.figure.Figure):
            return ManagedFigure(result)
        return result
    return wrapper


class ManagedFigure:
    """
    Main ManagedFigure wrapper

    While the plotting of anything should be done in normal matplotlib or seaborn,
    this class allows the post-creation adjustments of any figure into different
    sizes, with different ticks, labels, titles, etc.

    Parameters
    ----------
    matplotlib_fig: matplotlib.figure.Figure
        the .fig attribute of a matplotlib figure
        a copy of this attribute is made to ensure
        the origional figure doesn't change

    the number of subplots is inferred from here

    Managers
    --------
    legend: LegendManager
        Extension onto ManagedFigure dealing with legends
    ticks: TickManager
        Extension onto ManagedFigure dealing with ticks
    labels: LabalManager    
        Extension onto ManagedFigure dealing with labels    

    Returns
    -------
    Every function returns self: LegendManager, with the exception of .show(), which returns
    self.mpl_figure -> the viewable matplotlib.figure.Figure version

    Examples
    --------
    >>>

    Methods
    -------
    change_figsize()
    show()

    See Also
    --------
    @return_fig
        a decorator to return the MangedFigure object
    @for_axes
        a decorator to deal with applying changes to certain axis    
    """

    def __init__(self, matplotlib_fig: matplotlib.figure.Figure):
        self.mpl_figure: matplotlib.figure.Figure = pickle.loads(pickle.dumps(matplotlib_fig))

        # Store the single axis for convenience
        self.mpl_axes       = self.mpl_figure.axes
        self.num_subplots   = len(self.mpl_axes)
        
        # Create manager instances
        self.legend = LegendManager(self)
        self.ticks  = TickManager(self)
        self.labels = LabelManager(self)
        self.layout = LayoutManager(self)

    def change_figsize(self, width, height) -> 'ManagedFigure':
        self.mpl_figure.set_size_inches(width, height)
        
        # If axes have subplotspec (from gridspec), update their positions
        for ax in self.mpl_axes:
            if hasattr(ax, 'get_subplotspec') and ax.get_subplotspec() is not None:
                ax.set_position(ax.get_subplotspec().get_position(self.mpl_figure))
        
        return self
    
    def show(self) -> matplotlib.figure.Figure:
        return self.mpl_figure
    
    def __repr__(self):
        return f"<Fig(n_axes = {self.num_subplots}, legend, ticks, labels)>"