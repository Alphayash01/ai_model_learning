"""Exercise 6: a LightGBM forecaster with promo features, using MLForecast.

Run:  pytest -k ex06 -q
Docs: https://github.com/Nixtla/mlforecast
"""
import lightgbm as lgb
import pandas as pd
from mlforecast import MLForecast


def fit_predict(train: pd.DataFrame, future: pd.DataFrame, h: int) -> pd.DataFrame:
    """Train on `train`, forecast h days using the known future promo plan.

    train columns:  unique_id, ds, y, promo_flag
    future columns: unique_id, ds, promo_flag      (h rows per series)

    Return columns: unique_id, ds, lgbm   with lgbm never below 0.

    Steps (one TODO each):
      1. Build MLForecast with models={"lgbm": lgb.LGBMRegressor(...)}, freq="D",
         lags=[7, 14], date_features=["dayofweek"].
         Use n_estimators=150, learning_rate=0.05, random_state=0, verbosity=-1.
      2. fit(train, static_features=[])
         WHY static_features=[]: by default MLForecast treats every extra column as STATIC
         (one value per series). promo_flag changes every day, so we say "no static features".
      3. predict(h, X_df=future)
         WHY X_df: the model needs the future promo plan to know which days are promoted.
      4. clip the forecast at 0 (you cannot sell negative units).
    """
    raise NotImplementedError
