import pandas as pd


def make_cutoffs(last_date, h: int, n_windows: int, step: int) -> list[pd.Timestamp]:
    newest = pd.Timestamp(last_date) - pd.Timedelta(days=h)
    cuts = [newest - pd.Timedelta(days=step * k) for k in range(n_windows)]
    return sorted(cuts)


def split_window(df: pd.DataFrame, cutoff, h: int):
    cutoff = pd.Timestamp(cutoff)
    train = df[df["ds"] <= cutoff]
    test = df[(df["ds"] > cutoff) & (df["ds"] <= cutoff + pd.Timedelta(days=h))]
    return train, test
