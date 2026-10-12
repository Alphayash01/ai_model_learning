"""Lab: why 'this week minus last week' is a bad uplift estimate.

Run from the folder that contains demo/:   python practice/lab_naive_uplift.py
Part A: compare the naive estimate with the TRUE uplift (possible only in a simulation).
Part C: where it breaks (by discount depth, and with a disturbed last week). Part D: the PLACEBO test, which you can run on real data where truth is unknown.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))
import pipeline_demo as d  # noqa: E402

df = d.simulate()  # has true_baseline and true_mean columns (known only in simulation)

# ---------- find promotion events: (product, start index, length) ----------
events = []
for uid, g in df.groupby("unique_id"):
    f = g["promo_flag"].to_numpy()
    k = 0
    while k < len(f):
        if f[k] == 1:
            j = k
            while j < len(f) and f[j] == 1:
                j += 1
            events.append((uid, k, j - k))
            k = j
        else:
            k += 1

rows = []
for uid, start, length in events:
    g = df[df.unique_id == uid].reset_index(drop=True)
    if start - length < 0 or start + length > len(g):
        continue
    promo = g.iloc[start:start + length]
    before = g.iloc[start - length:start]                      # the "last week" window
    naive = promo["y"].sum() - before["y"].sum()               # naive uplift
    true = (promo["true_mean"] - promo["true_baseline"]).sum() # true uplift
    rows.append((uid, length, naive, true, promo["discount_pct"].mean(), before["y"].sum()))
ev = pd.DataFrame(rows, columns=["unique_id", "length", "naive", "true", "discount", "before_sum"])
ev["error"] = ev["naive"] - ev["true"]

print("PART A: naive uplift vs true uplift, per promotion event")
print(f"  events compared:            {len(ev)}")
print(f"  mean true uplift:           {ev['true'].mean():8.1f} units")
print(f"  mean naive uplift:          {ev['naive'].mean():8.1f} units")
print(f"  mean absolute error:        {ev['error'].abs().mean():8.1f} units "
      f"({ev['error'].abs().mean() / ev['true'].mean():.0%} of true)")
print(f"  events with error > 25%:    {(ev['error'].abs() / ev['true'] > 0.25).mean():.0%}")
print(f"  events with wrong sign:     {(ev['naive'] * ev['true'] < 0).mean():.0%}")

# ---------- Part C: where the naive method breaks ----------
print("\nPART C1: error by discount depth (small promotions drown in noise)")
ev["rel_err"] = ev["error"].abs() / ev["true"]
print(ev.groupby(ev["discount"].round(1))["rel_err"].agg(["count", "mean"]).rename(
    columns={"count": "events", "mean": "mean abs error as share of true"}).round(2).to_string())

# C2: inject a realistic disturbance into the 'last week' window for 20 percent of events:
# a stock-out (sales cut by 60 percent) or a holiday spike (sales up 50 percent).
rng2 = np.random.default_rng(1)
hit = rng2.random(len(ev)) < 0.20
shock = np.where(rng2.random(len(ev)) < 0.5, 0.4, 1.5)          # 0.4 = stock-out, 1.5 = holiday spike
before_shocked = np.where(hit, ev["before_sum"] * shock, ev["before_sum"])
naive_shocked = (ev["naive"] + ev["before_sum"]) - before_shocked
err2 = (naive_shocked - ev["true"]).abs()
print("\nPART C2: same events, but 20 percent had a stock-out or holiday spike in the 'last week' window (injected)")
print(f"  mean absolute error: {err2.mean():6.1f} units ({err2.mean() / ev['true'].mean():.0%} of true), "
      f"was {ev['error'].abs().mean():.1f}")
print(f"  events with error > 25%: {(err2 / ev['true'] > 0.25).mean():.0%}, was {(ev['error'].abs() / ev['true'] > 0.25).mean():.0%}")
print(f"  events with wrong sign:  {(naive_shocked * ev['true'] < 0).mean():.0%}")

# ---------- placebo: same naive formula where NO promotion happened ----------
plc = []
rng = np.random.default_rng(0)
for uid, g in df.groupby("unique_id"):
    g = g.reset_index(drop=True)
    ok = (g["promo_flag"] == 0) & (g["promo_flag"].shift(7, fill_value=0) == 0)
    for _ in range(40):
        s = int(rng.integers(14, len(g) - 14))
        w, p = g.iloc[s:s + 7], g.iloc[s - 7:s]
        if w["promo_flag"].sum() == 0 and p["promo_flag"].sum() == 0:
            plc.append(w["y"].sum() - p["y"].sum())
plc = np.array(plc)
print("\nPART D: placebo, 'uplift' measured in weeks with NO promotion")
print(f"  windows tested:             {len(plc)}")
print(f"  average 'uplift':           {plc.mean():8.1f} units  (should be about 0)")
print(f"  typical size (abs, median): {np.median(np.abs(plc)):8.1f} units  (pure noise, season, weekday mix)")
print(f"  95th percentile of |noise|: {np.percentile(np.abs(plc), 95):8.1f} units")
print("\nRead it: if real promotions add only about as much as the placebo noise, the naive number cannot be trusted.")
