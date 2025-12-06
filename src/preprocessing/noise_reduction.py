import numpy as np
import pandas as pd


def reduce_noise(df: pd.DataFrame, window_size: int = 3) -> pd.DataFrame:
    """
    Apply simple rolling mean smoothing to reduce noise on numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_smoothed = df.copy()
    df_smoothed[numeric_cols] = df_smoothed[numeric_cols].rolling(
        window=window_size, min_periods=1
    ).mean()
    return df_smoothed
