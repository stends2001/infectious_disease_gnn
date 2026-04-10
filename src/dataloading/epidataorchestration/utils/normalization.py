import pandas as pd
import numpy as np

from ...columnregistration import (
    LogParams, ZScoreParams, MinMaxParams
)

# Three types of Functions here:

# ======== 1. PARAMETERS - COMPUTING FUNCTIONS ========= #
def compute_zscore_params(train_df: pd.DataFrame, column: str) -> ZScoreParams:
    mean = float(train_df[column].mean())
    std  = float(train_df[column].std())
    return ZScoreParams(mean=mean, std=std)

def compute_minmax_params(train_df: pd.DataFrame, column: str) -> MinMaxParams:
    return MinMaxParams(
        min=float(train_df[column].min()),
        max=float(train_df[column].max())
    )

# ======== 2. TRANSFORMATION / NORMALIZATION - APPLYING FUNCTIONS ========= #
def apply_log(df, column, params: LogParams):
    out = df.copy()
    out[column] = np.log(df[column] + params.shift)
    return out

def apply_zscore(df: pd.DataFrame, column: str, params: ZScoreParams) -> pd.DataFrame:
    out = df.copy()
    out[column] = 0.0 if params.std == 0 else (df[column] - params.mean) / params.std
    return out

def apply_minmax(df: pd.DataFrame, column: str, params: MinMaxParams) -> pd.DataFrame:
    out = df.copy()
    rng = params.max - params.min
    out[column] = 0.0 if rng == 0 else (df[column] - params.min) / rng
    return out

# ======== 3. TRANSFORMATION / NORMALIZATION - REVERSE APPLYING FUNCTIONS ========= #
def reverse_log(df, column, params: LogParams):
    out = df.copy()
    out[column] = np.exp(df[column]) - params.shift
    return out

def reverse_zscore(df: pd.DataFrame, column: str, params: ZScoreParams) -> pd.DataFrame:
    out = df.copy()
    out[column] = df[column] * params.std + params.mean
    return out

def reverse_minmax(df: pd.DataFrame, column: str, params: MinMaxParams) -> pd.DataFrame:
    out = df.copy()
    out[column] = df[column] * (params.max - params.min) + params.min
    return out