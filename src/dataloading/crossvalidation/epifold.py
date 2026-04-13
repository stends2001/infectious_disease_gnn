from dataclasses import dataclass, replace
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from .crossvalidationepiconfig import CrossValidationEpiConfig
from ..epidataorchestration import EpiDataOrchestrator
from ..epiconfig import EpiConfig


@dataclass
class EpiFold:
    fold_idx:   int
    metadata:   Dict[str, str] 
    epiconfig:  EpiConfig
    data_orch:  EpiDataOrchestrator

    def __repr__(self) -> str:
        representation = f"<{self.__class__.__name__}(fold_idx = {self.fold_idx}, metadata, epiconfig, data_orch)>"

        return representation
    
@dataclass
class FoldCollection:
    epifolds:   List[EpiFold]
    foldconfig: CrossValidationEpiConfig

    @classmethod
    def build(cls, cv_config: CrossValidationEpiConfig, 
            base_config: EpiConfig) -> 'FoldCollection':
        
        # ── Step 1: run the expensive part once ──────────────────────────
        global_orch = EpiDataOrchestrator(base_config)
        global_orch.load()
        shared_raw = global_orch.data_raw  # RawEpiData — just dataframes, fully shareable


        # ── Step 2: cheap per-fold pipeline ──────────────────────────────
        folds = []
        for fold_idx in tqdm(range(cv_config.num_folds), desc='data orchestrator per fold', total=cv_config.num_folds):
            split = cv_config[fold_idx]

            fold_config = replace(
                base_config,
                min_date       = split['min_date'],
                split_trainval = split['trainval_split'],
                split_valtest  = split['valtest_split'],
                max_date       = split['max_date'],
            )

            fold_orch = (
                EpiDataOrchestrator.from_raw(fold_config, shared_raw)
                .harmonize()
                .process()
                .build_features()
                .normalize()
                .finalize()
            )
            
            folds.append(EpiFold(
                fold_idx  = fold_idx,
                metadata  = split,
                epiconfig = fold_config,
                data_orch = fold_orch,
            ))
        
        return cls(epifolds=folds, foldconfig=cv_config)

    def __iter__(self):
        return iter(self.epifolds)
    
    def __len__(self):
        return len(self.epifolds)
    
    def __getitem__(self, fold_idx: int):
        return self.epifolds[fold_idx]
    
    def summary(self) -> str:
        lines = [self.foldconfig.summary(), ""]
        for fold in self.epifolds:
            lines.append(repr(fold))
        return "\n".join(lines)