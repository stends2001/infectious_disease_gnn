from pathlib import Path

class PathManager:
    """ 
    Manages paths in this projects

    Attributes
    ----------
    - project_root
    - data
    - src
    - exp_out
    """
    def __init__(self):
        self.project_root   = Path(__file__).resolve().parent.parent.parent 
        self.data           = self.project_root / 'data'
        self.src            = self.project_root / 'src'
        self.exp_out        = self.data / 'experiment_outcomes'