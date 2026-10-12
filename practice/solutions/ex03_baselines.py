import numpy as np
import pandas as pd


def seasonal_naive(y, h: int, season: int = 7) -> np.ndarray:
    last = np.asarray(y)[-season:]
    return np.array([last[i % season] for i in range(h)])


def base_lift(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    dow = train["ds"].dt.dayofweek
    normal = train[train["promo_flag"] == 0]
    profile = normal.groupby(normal["ds"].dt.dayofweek)["y"].mean()
    promo = train[train["promo_flag"] == 1]
    if len(promo):
        expected = promo["ds"].dt.dayofweek.map(profile).sum()
        lift = promo["y"].sum() / expected
    else:
        lift = 1.0
    f_dow = future["ds"].dt.dayofweek.map(profile).to_numpy()
    return f_dow * np.where(future["promo_flag"].to_numpy() == 1, lift, 1.0)
