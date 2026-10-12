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
    return df.groupby(df["ds"].dt.dayofweek)["y"].mean()


def add_lag(df: pd.DataFrame, n: int) -> pd.DataFrame:
    out = df.copy()
    out[f"lag{n}"] = out["y"].shift(n)
    return out


def add_rolling_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    out[f"roll{window}"] = out["y"].shift(1).rolling(window).mean()
    return out
