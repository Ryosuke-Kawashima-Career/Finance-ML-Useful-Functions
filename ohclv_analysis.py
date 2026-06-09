import numpy as np
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
def shape(returns: pd.Series, periods_per_year: int = 365 * 24) -> float:
    standard_deviation = returns.std()
    if standard_deviation < 1e-12:
        return 0.0
    return returns.mean() / standard_deviation * np.sqrt(periods_per_year)

def walk_forward_splits(n: int, n_splits: int = 5, min_train_frac: float = 0.5):
    """Yields (trian_idx, val_idx) pairs for expanding-window walk-forward CV.
    Args:
        n(int): number of time data
        n_splits(int): number of test and validation sections
        min_train_frac(float): minimum fraction of training data
    """
    val_size = int(n * (1 - min_train_frac) / n_splits)
    val_size = max(val_size, 1)
    for i in range(n_splits):
        val_end = n - (n_splits - 1 - i) * val_size
        val_start = val_end - val_size
        train_end = val_start
        if train_end < int(n * min_train_frac):
            continue
        yield (np.arange(0, train_end), np.arange(val_start, val_end))

def causality_check(get_signal_fn, full_df, models, symbols, ic_table, cutoff_ts: int) -> bool:
    """Checks if there is no data leakage"""
    full_signal = get_signal_fn(full_df, models, symbols, ic_table)
    truncated_signal = get_signal_fn(full_df.loc[:cutoff_ts], models, symbols, ic_table)
    ok = True
    for sym in symbols:
        v_full = full_signal.loc[(cutoff_ts, sym), 'signal']
        v_truncated = truncated_signal.loc[(cutoff_ts, sym), 'signal']
        matched = np.isclose(v_full, v_truncated, rtol=1e-6)
        ok &= matched
    return ok