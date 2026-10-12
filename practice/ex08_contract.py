"""Exercise 8: the feed contract. A bad forecast must never be published.

Run:  pytest -k ex08 -q
"""
import pandas as pd


def check_feed(df: pd.DataFrame, prev_rows: int, horizon: int = 28) -> list[str]:
    """Return a list of error messages. An empty list means the feed is OK.

    df columns: sku_id, store_id, date, p10, p50, p90
    Checks, with these exact messages:
      "duplicate keys"                      same (sku_id, store_id, date) twice
      "quantiles not ordered"               any row where not p10 <= p50 <= p90
      "negative values"                     any negative p10, p50 or p90
      "short horizon"                       any (sku_id, store_id) with fewer than `horizon` distinct dates
      "row count jump"                      prev_rows > 0 and rows changed by more than 10 percent
    """
    raise NotImplementedError
