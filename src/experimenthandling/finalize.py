import os 
from tqdm import tqdm

from ..utils import PathManager
from ..utils.textformatting import crossmark
from ..experimenthandling import ExperimentAnalyzer

def load_and_compile(exp: str) -> None:
    """
    Load, compile and cache metrics for a single experiment.
    
    Loads saved model weights, runs evaluation, and saves compiled
    metrics to disk so subsequent loads can skip the expensive
    model loading step.
    
    Parameters
    ----------
    exp : str
        Experiment directory name (e.g. 'experiment_1a')
    """    
    analyzer = ExperimentAnalyzer(exp)
    analyzer.load_models()
    analyzer.compile_metrics()
    analyzer.save_metrics()   

def main() -> None:
    """
    To be run after all experiments are completed
    loads each experiment's results into an ExperimentAnalyzer
    and compiles and saves metrics such that the experiment can
    be loaded next time much quicker (by just calling `load_metrics()`)

    NOTE: This may take a very long time to execute
    """
    pm = PathManager()

    experiments_found = os.listdir(pm.exp_out)
    experiments_found.sort()

    for experiment in tqdm(experiments_found, desc = "loading and compiling experiments' output"):
        try:
            load_and_compile(experiment)

        except Exception as e:
            print(f"\n{crossmark} {experiment}: {e}")
            continue            

if __name__ == '__main__':
    main()