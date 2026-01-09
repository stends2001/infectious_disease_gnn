from typing import TYPE_CHECKING, Optional, Literal

def return_managedfigure(method):
    """Decorator --> to return ManagedFigure"""
    def wrapper(self, *args, **kwargs):
        method(self, *args, **kwargs)
        return self.managedfigure
    return wrapper

def for_axes(method):
    """
    Decorator to allow ax='all' or ax=int.
    Automatically loops over axes and returns the Fig object.
    """
    def wrapper(self, *args, ax='all', **kwargs):
        # determine axes to operate on
        if ax == 'all':
            axes_to_modify = self.managedfigure.mpl_axes
        elif isinstance(ax, int):
            if ax >= self.managedfigure.num_subplots:
                raise IndexError(f'ax {ax} not found. Found axes: {[ii for ii in range(self.managedfigure.num_subplots)]}')                
            
            axes_to_modify = [self.managedfigure.mpl_axes[ax]]
        else:
            raise ValueError("ax must be 'all' or an integer index")

        for a in axes_to_modify:
            method(self, *args, ax=a, **kwargs)  # pass single Axes to method

        return self.managedfigure  # return Fig, same as returns_fig_from_manager
    return wrapper

if TYPE_CHECKING:
    from .managedfigure import ManagedFigure

class TickManager:
    
    def __init__(self, managedfigure: 'ManagedFigure'):
        self.managedfigure: 'ManagedFigure' = managedfigure

    @for_axes
    def change_xticks(self, ticks, labels=None, ax = None) -> 'ManagedFigure':
        """Set custom x-axis tick positions and optionally labels"""
        ax.set_xticks(ticks)
        if labels is not None:
            ax.set_xticklabels(labels)
    
    @for_axes    
    def change_yticks(self, ticks, labels=None, ax = None) -> 'ManagedFigure':
        """Set custom y-axis tick positions and optionally labels"""
        ax.set_yticks(ticks)
        if labels is not None:
            ax.set_yticklabels(labels)
            
    @for_axes    
    def rotate_xticks(self, rotation, ha=None, ax = None) -> 'ManagedFigure':
        """Rotate x-axis tick labels"""
        ax.tick_params(axis='x', rotation=rotation)
        if ha is not None:
            for label in ax.get_xticklabels():
                label.set_ha(ha)

    @for_axes    
    def rotate_yticks(self, rotation, va=None, ax = None) -> 'ManagedFigure':
        """Rotate y-axis tick labels"""
        ax.tick_params(axis='y', rotation=rotation)
        if va is not None:
            for label in ax.get_yticklabels():
                label.set_va(va)

    @for_axes    
    def change_params(self, axis='both', ax = None, **kwargs) -> 'ManagedFigure':
        """Modify tick appearance (size, width, direction, etc.)"""
        ax.tick_params(axis=axis, **kwargs)
        
    @for_axes    
    def set_xtick_fontsize(self, size, ax = None) -> 'ManagedFigure':
        """Change x-axis tick label font size"""
        ax.tick_params(axis='x', labelsize=size)

    @for_axes    
    def set_ytick_fontsize(self, size, ax = None) -> 'ManagedFigure':
        """Change y-axis tick label font size"""
        ax.tick_params(axis='y', labelsize=size)

    @for_axes    
    def hide_xticks(self, labels=True, marks=True, ax = None) -> 'ManagedFigure':
        """Hide x-axis tick marks and/or labels"""
        if labels:
            ax.set_xticklabels([])
        if marks:
            ax.tick_params(axis='x', length=0)

    @for_axes    
    def hide_yticks(self, labels=True, marks=True, ax = None) -> 'ManagedFigure':
        """Hide y-axis tick marks and/or labels"""
        if labels:
            ax.set_yticklabels([])
        if marks:
            ax.tick_params(axis='y', length=0)

    @for_axes    
    def format_xticks(self, formatter, ax = None) -> 'ManagedFigure':
        """Apply custom formatter to x-axis (e.g., PercentFormatter, StrMethodFormatter)"""
        ax.xaxis.set_major_formatter(formatter)

    @for_axes    
    def format_yticks(self, formatter, ax = None) -> 'ManagedFigure':
        """Apply custom formatter to y-axis"""
        ax.yaxis.set_major_formatter(formatter)

class LabelManager:
    """
    Label Manager of ManagedFigure

    Methods
    -------
    change_suptitle()
    change_title()
    change_xlab()
    change_ylab()
    set_title_pad()
    set_suptitle()

    See Also
    --------
    @return_fig
        a decorator to return the MangedFigure object
    @for_axes
        a decorator to deal with applying changes to certain axis
    """
    def __init__(self, managedfigure: 'ManagedFigure'):
        self.managedfigure: 'ManagedFigure' = managedfigure

    # works on the figure itself
    @return_managedfigure
    def change_suptitle(self, suptitle: str, fontweight: Optional[str] = None, fontsize: Optional[int] = None) -> 'ManagedFigure':
        self.managedfigure.mpl_figure.suptitle(suptitle, fontsize=fontsize, fontweight=fontweight)

    @for_axes
    def change_title(self, title: str, fontweight: Optional[str] = None, fontsize: Optional[int] = None, ax=None) -> 'ManagedFigure':
        ax.set_title(title, fontsize=fontsize, fontweight=fontweight)

    @for_axes
    def change_xlab(self, xlabel: str, ax = None) -> 'ManagedFigure':
        ax.set_xlabel(xlabel)

    @for_axes
    def change_ylab(self, ylabel: str, ax = None) -> 'ManagedFigure':
        ax.set_ylabel(ylabel)
    
    @for_axes    
    def set_title_pad(self, pad: float, ax = None) -> 'ManagedFigure':
        """Adjust space between title and plot (in points)"""
        ax.title.set_pad(pad)

    @return_managedfigure    
    def set_suptitle_y(self, y) -> 'ManagedFigure':
        """Set suptitle vertical position (0 to 1, where 1 is top of figure)"""
        suptitle = self.managedfigure.mpl_figure._suptitle
        if suptitle is not None:
            suptitle.set_y(y)

    @for_axes
    def set_tag(self, tag: str, x: Optional[float] = -0.05, y: Optional[float] = 1.15, fontsize: Optional[float] = 14, fontweight: Optional[str] = 'bold', va: Optional[str] = 'top', ha: Optional[str] = 'left', ax = None) -> 'ManagedFigure':
        ax.text(x, y, tag, transform = ax.transAxes, fontsize = fontsize, fontweight = fontweight, va = va, ha = ha)

class LegendManager:

    def __init__(self, managedfigure: 'ManagedFigure'):
        self.managedfigure: 'ManagedFigure' = managedfigure
    
    @for_axes        
    def move(self, loc=None, bbox=None, ax = None) -> 'ManagedFigure':
        leg = ax.get_legend()
        if leg is None:
            raise RuntimeError("No legend exists yet")

        if loc is not None:
            leg.set_loc(loc)
        if bbox is not None:
            leg.set_bbox_to_anchor(bbox)

    @for_axes
    def remove(self, ax = None) -> 'ManagedFigure':
        """Remove the legend from the axis"""
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    @for_axes       
    def set_labels(self, labels, ax = None) -> 'ManagedFigure':
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

class LayoutManager:

    """
    extension onto ManagedFigure class dealing with general layout of a figure
    """

    def __init__(self, managedfigure: 'ManagedFigure'):
        self.managedfigure: 'ManagedFigure' = managedfigure    
    
    @for_axes
    def set_ylim(self, lim, ax = None) -> 'ManagedFigure':
        ax.set_ylim(lim)

    @for_axes
    def set_xlim(self, lim, ax = None) -> 'ManagedFigure':
        ax.set_xlim(lim)

    @for_axes
    def grid(self, mode: Literal['on','off'] = 'on', ax = None) -> 'ManagedFigure':
        if mode == 'on':
            ax.grid(True)
        else:
            ax.grid(False)