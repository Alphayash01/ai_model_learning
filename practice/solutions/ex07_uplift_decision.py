import numpy as np
import pandas as pd

PROMO_COLS = ["promo_flag", "discount_pct", "price_ratio", "days_into_promo",
              "days_since_promo_end", "post_promo_week"]


def promo_off_frame(future: pd.DataFrame) -> pd.DataFrame:
    off = future.copy()
    defaults = {"promo_flag": 0, "discount_pct": 0.0, "price_ratio": 1.0,
                "days_into_promo": 0, "days_since_promo_end": 30, "post_promo_week": 0}
    for col, val in defaults.items():
        if col in off.columns:
            off[col] = val
    return off


def uplift(pred_on, pred_off) -> np.ndarray:
    return np.asarray(pred_on, float) - np.asarray(pred_off, float)


def stock_level(quantile_levels, quantile_values, cu: float, co: float) -> float:
    if cu <= 0 or co <= 0:
        raise ValueError("costs must be positive")
    ratio = cu / (cu + co)
    return float(np.interp(ratio, quantile_levels, quantile_values))
