import pandas as pd


def promo_counters(flags: list[int]) -> pd.DataFrame:
    n = len(flags)
    into = [0] * n
    since = [30] * n
    post = [0] * n
    run = 0
    last_end = None
    for k in range(n):
        if flags[k] == 1:
            run += 1
            into[k] = run
            since[k] = 0
        else:
            if k > 0 and flags[k - 1] == 1:
                last_end = k
            run = 0
            if last_end is not None:
                since[k] = min(k - last_end + 1, 30)
            post[k] = 1 if 1 <= since[k] <= 7 else 0
    return pd.DataFrame({"days_into_promo": into,
                         "days_since_promo_end": since,
                         "post_promo_week": post})
