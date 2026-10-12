"""Exercise 1: pandas warm-up (shift, rolling, groupby).

These three tools build almost every forecasting feature you will ever write.
Fill in each function. Run:  pytest -k ex01 -q
"""
import pandas as pd


def make_sales() -> pd.DataFrame:
    """Given. 14 days of sales starting Thursday 2026-01-01."""
    ds = pd.date_range("2026-01-01", periods=14, freq="D")
    y = [10, 12, 11, 13, 20, 30, 25, 11, 12, 10, 14, 22, 31, 26]
    return pd.DataFrame({"ds": ds, "y": y})


def weekday_mean(df: pd.DataFrame) -> pd.Series:
    """Average y for each weekday number (0=Monday ... 6=Sunday).

    Hint: df["ds"].dt.dayofweek gives the weekday number; then groupby.
    """
    df["weekday"] = df["ds"].dt.dayofweek
    return df.groupby("weekday")["y"].mean()
    raise NotImplementedError


def add_lag(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return a COPY with a new column f"lag{n}" holding y from n days earlier.

    The first n rows must be NaN. Do not change the input frame.
    Hint: Series.shift(n)
    """
    out = df.copy()
    out[f"lag{n}"] = out["y"].shift(n)
    return out

    raise NotImplementedError


def add_rolling_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return a COPY with column f"roll{window}": mean of the PREVIOUS `window` days.

    Today's value must NOT be included (that would leak the answer into the feature).
    Hint: shift by 1 first, then .rolling(window).mean()
    """
    out = df.copy()
    out[f"roll{window}"] = out["y"].shift(1).rolling(window).mean()
    return out
    raise NotImplementedError
