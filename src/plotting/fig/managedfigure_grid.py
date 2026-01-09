import matplotlib.pyplot as plt
from typing import List, Optional,Tuple

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from .managedfigure import ManagedFigure


class ManagedFigureGrid:
    """"""
    def __init__(self, managedfigures: List[ManagedFigure], nrows: int = 1, ncols: Optional[int] = None, figsize: Tuple[float,float] = (10,10)):
        self.managedfigures = managedfigures

        if ncols is None:
            ncols = len(managedfigures)

        self.nrows          = nrows 
        self.ncols          = ncols

        fig, axes   = plt.subplots(nrows=nrows, ncols=ncols, figsize = figsize)
        axes        = axes.flatten() if nrows * ncols > 1 else [axes] 

        for ax, mf in zip(axes, managedfigures):
                # attach Agg canvas
                canvas = FigureCanvas(mf.mpl_figure)
                mf.mpl_figure.set_canvas(canvas)

                # render
                canvas.draw()

                # get the RGBA buffer
                buf = np.asarray(canvas.buffer_rgba())
                # convert RGBA -> RGB by ignoring alpha channel
                img = buf[:, :, :3]

                # display in subplot
                ax.imshow(img)
                ax.axis('off')


        self.fig    = fig 
        self.axes   = axes       

    def suptitle(self, suptitle: str, fontweight: Optional[str] = None, fontsize: Optional[int] = None):
        self.fig.suptitle(suptitle, fontsize=fontsize, fontweight=fontweight)

    def position_suptitle(self, left: Optional[float], right: Optional[float], top: Optional[float], bottom: Optional[float], wspace: Optional[float], hspace: Optional[float]):
        self.fig.subplots_adjust(left, right, top, bottom, wspace, hspace)

    def show(self):
        self.fig.show()

