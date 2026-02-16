# Spatiotemporal modelling of infectious diseases

## New big update to be made: Foldwise train/val/test
from dataclasses import replace
from typing import List, Tuple
import pandas as pd

@dataclass
class CVFold:
    fold_idx:       int
    train_end:      str   # = split_trainval
    val_end:        str   # = split_valtest

class SpatioTemporalCV:
    
    def __init__(self, 
                 base_config:   EpiConfig,
                 cv_strategy:   Literal['expanding', 'sliding'],
                 n_folds:       int,
                 val_size:      str,          # e.g. '52w', '12m'
                 train_size:    Optional[str] = None  # only for sliding
                 ):
        
        self.base_config    = base_config
        self.cv_strategy    = cv_strategy
        self.n_folds        = n_folds
        self.val_size       = val_size
        self.train_size     = train_size

    def _generate_folds(self) -> List[CVFold]:
        folds       = []
        val_end     = pd.Timestamp(self.base_config.split_valtest)
        val_delta   = self._parse_size(self.val_size)

        for i in range(self.n_folds - 1, -1, -1):
            fold_val_end    = val_end   - i * val_delta
            fold_val_start  = fold_val_end - val_delta

            if self.cv_strategy == 'sliding' and self.train_size:
                train_delta     = self._parse_size(self.train_size)
                fold_train_start= fold_val_start - train_delta
            else:
                fold_train_start= pd.Timestamp(self.base_config.min_date)

            folds.append(CVFold(
                fold_idx    = len(folds),
                train_end   = fold_val_start.strftime('%Y-%m-%d'),
                val_end     = fold_val_end.strftime('%Y-%m-%d'),
            ))
        return folds

    def _parse_size(self, size_str: str) -> pd.DateOffset:
        """Parse '52w', '12m', '365d' into DateOffset"""
        num = int(size_str[:-1])
        unit = size_str[-1]
        if unit == 'w':
            return pd.DateOffset(weeks=num)
        elif unit == 'm':
            return pd.DateOffset(months=num)
        elif unit == 'd':
            return pd.DateOffset(days=num)
        else:
            raise ValueError(f'Unknown size unit: {unit}')

    def get_fold_config(self, fold: CVFold) -> EpiConfig:
        """Return a new EpiConfig with dates adjusted for this fold"""
        return replace(
            self.base_config,
            split_trainval  = fold.train_end,
            split_valtest   = fold.val_end,   # val_end becomes the new max
            max_date        = fold.val_end,
        )

    def run(self, model_fn) -> List[dict]:
        """
        model_fn: callable that takes EpiConfig and returns metric dict
        e.g. lambda config: train_and_evaluate(config)
        """
        results = []
        for fold in self._generate_folds():
            fold_config = self.get_fold_config(fold)
            orchestrator = EpiDataOrchestrator(fold_config).build()
            metrics = model_fn(orchestrator)
            results.append({'fold': fold.fold_idx, **metrics})
        return results

## Literature Review

[[Kraemer, 2024]]

[[Croft, 2023]]

[[Jeong, 2025]]

[[Wang, 2025]] use a SIR based model to predict parameters relevant to the spread over multiple regions.

[[Liu, 2024]] propose the GRGNN, the GRU-based GNN. 

[[Grasslyl, 2006]] --> seasonality

## Resources