from pathlib import Path
import os 

class PathManager:
    """ 
    Main path manager
    """
    def __init__(self):
        self.project_root   = Path(__file__).resolve().parent.parent.parent 
        self.data           = self.project_root / 'data'
        self.src            = self.project_root / 'src'
        self.exp_out        = self.data / 'experiment_outcomes'