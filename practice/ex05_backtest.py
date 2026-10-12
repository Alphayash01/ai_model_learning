"""Exercise 5: rolling-origin backtest windows.

Run:  pytest -k ex05 -q
"""
import pandas as pd


def make_cutoffs(last_date, h: int, n_windows: int, step: int) -> list[pd.Timestamp]:
    """Return cut-off dates, OLDEST FIRST.

    The newest window's test period must end exactly on last_date, so the newest
    cut-off is last_date minus h days. Each older cut-off is `step` days earlier.

    last_date = 2026-06-30, h = 28, n_windows = 4, step = 28
      -> [2026-03-10, 2026-04-07, 2026-05-05, 2026-06-02]
    """
    raise NotImplementedError


def split_window(df: pd.DataFrame, cutoff, h: int):
    """Return (train, test).

    train: rows with ds <= cutoff
    test:  rows with cutoff < ds <= cutoff + h days
    """
    raise NotImplementedError
