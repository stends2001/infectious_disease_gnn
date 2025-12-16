from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .fig import Fig

class LegendManager:

    def __init__(self, fig_wrapper: 'Fig'):
        self.fig_wrapper = fig_wrapper

    def move(self, loc=None, bbox=None) -> 'Fig':
        leg = self.fig_wrapper.mpl_axis.get_legend()
        if leg is None:
            raise RuntimeError("No legend exists yet")

        if loc is not None:
            leg.set_loc(loc)
        if bbox is not None:
            leg.set_bbox_to_anchor(bbox)

        return self.fig_wrapper

    def set_labels(self, labels) -> 'Fig':
        handles, _ = self.fig_wrapper.mpl_axis.get_legend_handles_labels()
        self.fig_wrapper.mpl_axis.legend(handles, labels)
        return self.fig_wrapper

class TickManager:
    
    def __init__(self, fig_wrapper: 'Fig'):
        self.fig_wrapper = fig_wrapper
    
    def change_xticks(self, ticks, labels=None) -> 'Fig':
        """Set custom x-axis tick positions and optionally labels"""
        self.fig_wrapper.mpl_axis.set_xticks(ticks)
        if labels is not None:
            self.fig_wrapper.mpl_axis.set_xticklabels(labels)
        return self.fig_wrapper
    
    def change_yticks(self, ticks, labels=None) -> 'Fig':
        """Set custom y-axis tick positions and optionally labels"""
        self.fig_wrapper.mpl_axis.set_yticks(ticks)
        if labels is not None:
            self.fig_wrapper.mpl_axis.set_yticklabels(labels)
        return self.fig_wrapper
    
    def rotate_xticks(self, rotation, ha=None) -> 'Fig':
        """Rotate x-axis tick labels"""
        self.fig_wrapper.mpl_axis.tick_params(axis='x', rotation=rotation)
        if ha is not None:
            for label in self.fig_wrapper.mpl_axis.get_xticklabels():
                label.set_ha(ha)
        return self.fig_wrapper
    
    def rotate_yticks(self, rotation, va=None) -> 'Fig':
        """Rotate y-axis tick labels"""
        self.fig_wrapper.mpl_axis.tick_params(axis='y', rotation=rotation)
        if va is not None:
            for label in self.fig_wrapper.mpl_axis.get_yticklabels():
                label.set_va(va)
        return self.fig_wrapper
    
    def change_params(self, axis='both', **kwargs) -> 'Fig':
        """Modify tick appearance (size, width, direction, etc.)"""
        self.fig_wrapper.mpl_axis.tick_params(axis=axis, **kwargs)
        return self.fig_wrapper
    
    def set_xtick_fontsize(self, size) -> 'Fig':
        """Change x-axis tick label font size"""
        self.fig_wrapper.mpl_axis.tick_params(axis='x', labelsize=size)
        return self.fig_wrapper
    
    def set_ytick_fontsize(self, size) -> 'Fig':
        """Change y-axis tick label font size"""
        self.fig_wrapper.mpl_axis.tick_params(axis='y', labelsize=size)
        return self.fig_wrapper
    
    def hide_xticks(self, labels=True, marks=True) -> 'Fig':
        """Hide x-axis tick marks and/or labels"""
        if labels:
            self.fig_wrapper.mpl_axis.set_xticklabels([])
        if marks:
            self.fig_wrapper.mpl_axis.tick_params(axis='x', length=0)
        return self.fig_wrapper
    
    def hide_yticks(self, labels=True, marks=True) -> 'Fig':
        """Hide y-axis tick marks and/or labels"""
        if labels:
            self.fig_wrapper.mpl_axis.set_yticklabels([])
        if marks:
            self.fig_wrapper.mpl_axis.tick_params(axis='y', length=0)
        return self.fig_wrapper
    
    def format_xticks(self, formatter) -> 'Fig':
        """Apply custom formatter to x-axis (e.g., PercentFormatter, StrMethodFormatter)"""
        self.fig_wrapper.mpl_axis.xaxis.set_major_formatter(formatter)
        return self.fig_wrapper
    
    def format_yticks(self, formatter) -> 'Fig':
        """Apply custom formatter to y-axis"""
        self.fig_wrapper.mpl_axis.yaxis.set_major_formatter(formatter)
        return self.fig_wrapper        


class LabelManager:

    def __init__(self, fig_wrapper: 'Fig'):
        self.fig_wrapper = fig_wrapper

    def change_suptitle(self, suptitle: str, fontweight: Optional[str] = None, fontsize: Optional[str] = None) -> 'Fig':
        self.fig_wrapper.mpl_figure.suptitle(suptitle, fontsize=fontsize, fontweight=fontweight)
        return self.fig_wrapper

    def change_title(self, title: str, fontweight: Optional[str] = None, fontsize: Optional[str] = None) -> 'Fig':
        self.fig_wrapper.mpl_axis.set_title(title, fontsize=fontsize, fontweight=fontweight)
        return self.fig_wrapper

    def change_xlab(self, xlabel: str) -> 'Fig':
        self.fig_wrapper.mpl_axis.set_xlabel(xlabel)
        return self.fig_wrapper

    def change_ylab(self, ylabel: str) -> 'Fig':
        self.fig_wrapper.mpl_axis.set_ylabel(ylabel)
        return self.fig_wrapper
    
    def set_title_pad(self, pad) -> 'Fig':
        """Adjust space between title and plot (in points)"""
        self.fig_wrapper.mpl_axis.title.set_pad(pad)
        return self.fig_wrapper

    def set_suptitle_y(self, y) -> 'Fig':
        """Set suptitle vertical position (0 to 1, where 1 is top of figure)"""
        suptitle = self.fig_wrapper.mpl_figure._suptitle
        if suptitle is not None:
            suptitle.set_y(y)
        return self.fig_wrapper


# class MapFig(Fig):