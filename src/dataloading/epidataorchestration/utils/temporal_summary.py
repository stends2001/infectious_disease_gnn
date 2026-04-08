from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Literal
import pandas as pd

from .issues import TemporalError

# ==== helper functions ===== #

def convert_to_next_monday(date: datetime, day_int = 0) -> datetime:
    """returns datetime object thats just shifted to the next version of day int where 0 means Monday"""
    if date.weekday() != day_int:
        days_ahead = (day_int - date.weekday()) % 7
        if days_ahead == 0:  # If we want same day, go to next week
            days_ahead = 7
        shifted_date = date + timedelta(days=days_ahead)
        return shifted_date
    else:
        return date

def convert_to_month_start(date: datetime) -> datetime:
    """Convert date to first day of the month"""
    return datetime(date.year, date.month, 1)

class EpiDataTemporalSummary:
    """
    Stores all the temporal information from EpiConfig, and deals with it in terms of splitting logic, extending dates, 
    and shifting to the future vs past
    """
    def __init__(self, 
                 temporal_frequency: str,
                 min_date:           str,
                 max_date:           str,
                 split_trainval:     str,
                 split_valtest:      str,

                 horizon_size:       int,
                 horizon_leadtime:   int,
                 num_lags:           int, 
                 sequence_length:    int
                 ):
        
        self.temporal_frequency     = temporal_frequency

        # input
        self.min_date                = datetime.strptime(min_date, "%Y-%m-%d") 
        self.max_date                = datetime.strptime(max_date, "%Y-%m-%d") 
        self.split_trainval          = datetime.strptime(split_trainval, "%Y-%m-%d") 
        self.split_valtest           = datetime.strptime(split_valtest, "%Y-%m-%d")                         

        self.horizon_size            = horizon_size        
        self.horizon_leadtime        = horizon_leadtime
        self.num_lags                = num_lags 
        self.sequence_length         = sequence_length

        self._resample()
        self._set_extended_timepoints()
        self._set_backwarded_timestamps()
        self._validate_dates_order()

    def _resample(self) -> None:
        """Align dates to temporal frequency (Mondays for weekly, 1st for monthly)"""
        if self.temporal_frequency == 'w':
            self.min_date       = convert_to_next_monday(self.min_date)
            self.split_trainval = convert_to_next_monday(self.split_trainval)
            self.split_valtest  = convert_to_next_monday(self.split_valtest)
            self.max_date       = convert_to_next_monday(self.max_date)

        elif self.temporal_frequency == 'm':
            self.min_date       = convert_to_month_start(self.min_date)
            self.split_trainval = convert_to_month_start(self.split_trainval)
            self.split_valtest  = convert_to_month_start(self.split_valtest)
            self.max_date       = convert_to_month_start(self.max_date)

    def _set_extended_timepoints(self) -> None:
        """extends timepoints based on all config input for the data loading / filtering"""
        # lookback periods for lags:
        self.lookback_periods = (self.num_lags - 1 )+ (self.sequence_length - 1) + (self.horizon_leadtime)
        self.forward_periods  = self.horizon_leadtime + (self.horizon_size - 1)

        self.min_date_extended = self._shift(self.min_date, -self.lookback_periods)
        self.max_date_extended = self._shift(self.max_date,  1)

    def _set_backwarded_timestamps(self):
        # Calculate target splits (shifted forward by horizon)
        self.split_trainval_bwd = self._shift(self.split_trainval, -self.horizon_leadtime)
        self.split_valtest_bwd  = self._shift(self.split_valtest, -self.horizon_leadtime)        

    def _shift(self, date: datetime, steps: int) -> datetime:
        """Shift date by steps (positive=forward, negative=backward)"""
        if self.temporal_frequency == 'd':
            return date + timedelta(days=steps)
        elif self.temporal_frequency == 'w':
            return date + timedelta(weeks=steps)
        elif self.temporal_frequency == 'm':
            return date + relativedelta(months=steps)
        else:
            raise TemporalError(f"Unknown frequency: {self.temporal_frequency}")

    def _validate_dates_order(self):
        if not self.min_date < self.split_trainval < self.split_valtest < self.max_date:
            raise TemporalError('Incorrect order of date-values')

    # ======= GETTER METHODs ====== #
    def get_extended_dates(self) -> Dict[str, pd.Timestamp]:
        """Get extended min/max dates for initial data loading"""
        return {
            'min': pd.Timestamp(self.min_date_extended),
            'max': pd.Timestamp(self.max_date_extended)
        }
    
    def get_input_splits(self) -> Dict[str, pd.Timestamp]:
        """Get INPUT split timestamps for creating train/val/test columns"""
        return {
            'trainval': pd.Timestamp(self.split_trainval),
            'valtest':  pd.Timestamp(self.split_valtest)
        }
    
    def get_target_splits(self) -> Dict[str, pd.Timestamp]:
        """Get TARGET split timestamps (for reference/plotting)"""
        return {
            'trainval': pd.Timestamp(self.split_trainval_bwd),
            'valtest': pd.Timestamp(self.split_valtest_bwd)
        }    
    
    def get_daterange_dataset(self, dataset: Literal['train','val','test'], reference: Literal['t0','target'] = 'target') -> List[datetime]:
        if dataset == 'train':
            min = self._shift(self.min_date_extended, steps = self.sequence_length - 1)
            max = self._shift(self.split_trainval_bwd, steps = -1) # not in actual data     
        elif dataset == 'val':
            min = self.split_trainval_bwd
            max = self._shift(self.split_valtest_bwd, steps = -1) # not in actual data     
        else: 
            #dataset == 'test':
            min = self.split_valtest_bwd
            max = self._shift(self.max_date, steps = -self.horizon_leadtime) # not in actual data     

        if reference == 't0':
            daterange = [min,max]
        else:
            daterange = [self._shift(min, steps = self.horizon_leadtime),
                         self._shift(max, steps = self.horizon_leadtime)]
        return daterange

    def minimal_summary(self) -> str: 
        """small - scale summary: selection of attributes displayed"""
        summary =(
            f"<{self.__class__.__name__}(temporal_frequency={self.temporal_frequency}, "
                f"min_date={self.min_date.date()}, "             
                f"max_date={self.max_date.date()}, "
                f"min_date_extended={self.min_date_extended.date()}, "                   
                f"max_date_extended={self.max_date_extended.date()}, "                   
                f"split_trainval={self.split_trainval.date()}, "                
                f"split_valtest={self.split_valtest.date()}, "
                f"horizon_size={self.horizon_size}, "  
                f"horizon_leadtime={self.horizon_leadtime}, "  
                f"num_lags={self.num_lags}, "  
                f"sequence_length={self.sequence_length})"                                                  
        )      
        return summary

    def __repr__(self) -> str: 
        return self.minimal_summary()