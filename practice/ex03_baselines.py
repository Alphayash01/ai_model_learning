"""Exercise 3: the two baselines every model must beat.

Run:  pytest -k ex03 -q
"""
import numpy as np
import pandas as pd


def seasonal_naive(y, h: int, season: int = 7) -> np.ndarray:
    """Forecast the next h days by repeating the last `season` observed values.

    y = [1..14], h = 10, season = 7  ->  [8, 9, 10, 11, 12, 13, 14, 8, 9, 10]
    """
    raise NotImplementedError


def base_lift(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Base-lift forecast.

    train columns:  ds (datetime), y, promo_flag (0/1)
    future columns: ds (datetime), promo_flag (0/1)

    Steps:
      1. baseline profile = mean of y per weekday, using ONLY non-promo training days
      2. lift = (sum of y on promo training days) / (sum of the profile values on those same days)
         use lift = 1.0 if the training data has no promo days
      3. forecast for each future day = profile[its weekday] * (lift if promo_flag == 1 else 1)

    Return a numpy array in the same order as `future`.
    """
    raise NotImplementedError
