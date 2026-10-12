import lightgbm as lgb
import pandas as pd
from mlforecast import MLForecast


def fit_predict(train: pd.DataFrame, future: pd.DataFrame, h: int) -> pd.DataFrame:
    fc = MLForecast(
        models={"lgbm": lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05,
                                          random_state=0, verbosity=-1)},
        freq="D", lags=[7, 14], date_features=["dayofweek"],
    )
    fc.fit(train, static_features=[])
    out = fc.predict(h, X_df=future)
    out["lgbm"] = out["lgbm"].clip(lower=0)
    return out
