"""Exercise 7: uplift (two scenarios) and the cost-based stock decision.

Run:  pytest -k ex07 -q
"""
import numpy as np
import pandas as pd

PROMO_COLS = ["promo_flag", "discount_pct", "price_ratio", "days_into_promo",
              "days_since_promo_end", "post_promo_week"]


def promo_off_frame(future: pd.DataFrame) -> pd.DataFrame:
    """Return a COPY of `future` as if no promotion happened.

    promo_flag = 0, discount_pct = 0, price_ratio = 1.0, days_into_promo = 0,
    days_since_promo_end = 30, post_promo_week = 0.
    Columns that are not promo columns (unique_id, ds ...) stay unchanged.
    """
    raise NotImplementedError


def uplift(pred_on, pred_off) -> np.ndarray:
    """Units added (or lost) by the promotion: pred_on - pred_off, as a numpy array."""
    raise NotImplementedError


def stock_level(quantile_levels, quantile_values, cu: float, co: float) -> float:
    """Newsvendor rule. Stock up to the forecast quantile at ratio = cu / (cu + co).

    cu = margin lost per unit you could not sell
    co = cost per leftover unit
    Interpolate between the given quantiles (numpy has a function for this).
    Raise ValueError if cu <= 0 or co <= 0.
    """
    raise NotImplementedError
