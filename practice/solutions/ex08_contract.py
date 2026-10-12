import pandas as pd


def check_feed(df: pd.DataFrame, prev_rows: int, horizon: int = 28) -> list[str]:
    errors = []
    if df.duplicated(["sku_id", "store_id", "date"]).any():
        errors.append("duplicate keys")
    if not ((df["p10"] <= df["p50"]) & (df["p50"] <= df["p90"])).all():
        errors.append("quantiles not ordered")
    if (df[["p10", "p50", "p90"]] < 0).any().any():
        errors.append("negative values")
    days = df.groupby(["sku_id", "store_id"])["date"].nunique()
    if (days < horizon).any():
        errors.append("short horizon")
    if prev_rows > 0 and abs(len(df) - prev_rows) / prev_rows > 0.10:
        errors.append("row count jump")
    return errors
