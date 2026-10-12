"""Exercise 2: promotion counters (the features that let a model see the post-promo dip).

Run:  pytest -k ex02 -q
"""
import pandas as pd


def promo_counters(flags: list[int]) -> pd.DataFrame:
    """Build three columns from a daily 0/1 promo plan.

    days_into_promo:      0 when there is no promo; 1 on the first promo day, 2 on the second ...
    days_since_promo_end: 0 DURING a promo. On the first day AFTER a promo ends it is 1, then 2, 3 ...
                          Capped at 30. It is 30 until the first promo has ended.
    post_promo_week:      1 when days_since_promo_end is between 1 and 7 (inclusive), else 0.

    Example: flags = [0, 1, 1, 0, 0]
      days_into_promo      = [0, 1, 2, 0, 0]
      days_since_promo_end = [30, 0, 0, 1, 2]
      post_promo_week      = [0, 0, 0, 1, 1]

    Return a DataFrame with exactly those three columns, in that order.
    Hint: loop over the days and keep two variables: how long the current promo has run,
    and the index where the last promo ended.
    """
    raise NotImplementedError
