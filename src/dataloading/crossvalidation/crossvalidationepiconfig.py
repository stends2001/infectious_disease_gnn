from dataclasses import dataclass, field
from typing import Literal, assert_never, Dict
import pandas as pd

from .issues import CrossValidationEpiConfigError
from ...utils import align

@dataclass(frozen=True)
class CrossValidationEpiConfig:
    """     
    """
    min_date        : str
    max_date        : str
    num_folds       : int
    fold_step       : int
    train_periods   : int
    val_periods     : int
    test_periods    : int
    expansion_method: Literal['fixed_origin', 'rolling_window'] = 'rolling_window'
    unit            : Literal['y','m','w','d']   = 'y'

    _splits         : Dict[int, Dict[str,str]] = field(default_factory=dict, init=False, repr=False)

    # ======== DUNDER ======== #
    def __post_init__(self):
        self._validate_input()
        self._validate_time()
        self._set_splits()

    def __len__(self) -> int:
        return self.num_folds

    def __iter__(self):
        for i in range(self.num_folds):
            yield self._splits[i]

    def __getitem__(self, fold_idx: int):
        return self._splits[fold_idx]

    # ======= METHODS ======== #
    def get_all_splits(self) -> Dict[int, Dict[str, str]]:
        return dict(self._splits)

    # ======== HELPERS ========= #    
    def _set_splits(self):
        """set split - info in `.splits_`"""
        min_dt = pd.Timestamp(self.min_date)
        max_dt = pd.Timestamp(self.max_date)

        for fold_idx in range(self.num_folds):

            if self.expansion_method == 'fixed_origin':
                train_start = min_dt
                train_end   = min_dt + self._offset(self.train_periods + fold_idx * self.fold_step)
            elif self.expansion_method == 'rolling_window':
                train_start = min_dt + self._offset(fold_idx * self.fold_step)
                train_end   = train_start + self._offset(self.train_periods)

            else:
                assert_never(self.expansion_method)

            fold_min_date       = train_start
            fold_trainval_split = train_end
            fold_valtest_split  = train_end + self._offset(self.val_periods)
            fold_max_date       = fold_valtest_split + self._offset(self.test_periods)

            if fold_max_date > max_dt:
                raise ValueError(
                    f"Fold {fold_idx} exceeds max_date. Reduce num_folds."
                )

            self._splits[fold_idx] = {
                'min_date'       : fold_min_date.strftime('%Y-%m-%d'),
                'trainval_split' : fold_trainval_split.strftime('%Y-%m-%d'),
                'valtest_split'  : fold_valtest_split.strftime('%Y-%m-%d'),
                'max_date'       : fold_max_date.strftime('%Y-%m-%d'),
            }
    
    def _offset(self, n: int) -> pd.DateOffset:
        """Return a DateOffset of n units."""
        match self.unit:
            case "y":
                return pd.DateOffset(years=n)
            case "m":
                return pd.DateOffset(months=n)  
            case "w":
                return pd.DateOffset(weeks=n)
            case "d":
                return pd.DateOffset(days=n)
            case _:
                assert_never(self.unit)        

    # ======= VALIDATION ========= #
    def _validate_input(self):
        """validate restricted inputs"""
        allowed_units = ['y','m','w','d']
        if self.unit not in allowed_units:
            raise CrossValidationEpiConfigError(f'Invalid input for unit. Must be one of {allowed_units}. Got {self.unit}')
        
        allowed_methods = ['rolling_window','fixed_origin']
        if self.expansion_method not in allowed_methods:
            raise CrossValidationEpiConfigError(f'Invalid input for expansion_method. Must be one of {allowed_methods}. Got {self.expansion_method}')        

        if self.fold_step < 1:
            raise CrossValidationEpiConfigError(
                f'fold_step must be >= 1. Got {self.fold_step}'
            )            
        
    def _validate_time(self):
        """validate input of temporal nature"""
        min_dt = pd.Timestamp(self.min_date)
        max_dt = pd.Timestamp(self.max_date)

        # Check a single fold fits within the date range
        single_fold_end = (
            min_dt
            + self._offset(self.train_periods)
            + self._offset(self.val_periods)
            + self._offset(self.test_periods)
        )
        if single_fold_end > max_dt:
            raise CrossValidationEpiConfigError(
                f"Date range {self.min_date} to {self.max_date} is too short "
                f"for train/val/test = "
                f"{self.train_periods}/{self.val_periods}/{self.test_periods} {self.unit}."
            )

        if self.expansion_method == 'fixed_origin':
            last_test_end = (
                min_dt
                + self._offset(self.train_periods + (self.num_folds - 1) * self.fold_step)
                + self._offset(self.val_periods)
                + self._offset(self.test_periods)
            )
        elif self.expansion_method == 'rolling_window':
            last_test_end = (
                min_dt
                + self._offset((self.num_folds - 1) * self.fold_step)
                + self._offset(self.train_periods)
                + self._offset(self.val_periods)
                + self._offset(self.test_periods)
            )
        else:
            assert_never(self.expansion_method)

        if last_test_end > max_dt:
            raise CrossValidationEpiConfigError(
                f"{self.num_folds} folds with these window sizes exceed the available "
                f"date range. Last fold would end {last_test_end.date()}, "
                f"but max_date is {self.max_date}."
            )    

    # ======= REPRESENTATION ====== #
    def summary(self) -> str:
        alignment = "^"
        column_sizes = [6, 14, 14, 14, 14]

        # Header
        header = (
            f"{'Fold':{alignment}{column_sizes[0]}} | "
            f"{'Min Date':{alignment}{column_sizes[1]}} | "
            f"{'Train/Val':{alignment}{column_sizes[2]}} | "
            f"{'Val/Test':{alignment}{column_sizes[3]}} | "
            f"{'Max Date':{alignment}{column_sizes[4]}}"
        )

        table_repr = [header]
        table_repr.append("-" * len(header))

        # Data rows
        for fold_idx, split_dict in self._splits.items():
            values = [fold_idx] + list(split_dict.values())
            line = " | ".join(
                f"{str(val):{alignment}{width}}" 
                for val, width in zip(values, column_sizes)
            )
            table_repr.append(line)

        table = "\n".join(table_repr)
        return table

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        """returns aligned repr of input"""
        all_keys = ['num_folds','fold_step','train_periods','val_periods','test_periods','unit','expansion_method']

        width           = max(len(k) for k in all_keys)
        indent          = 4

        lines = [f"<{self.__class__.__name__}(\n"]

        for attr_name in all_keys:
            attr_value = getattr(self, attr_name)
            lines.append(align(attr_name, attr_value, width, indent = indent, newline = True))

        lines.append(")>")
        representation = "".join(lines)
        return representation
