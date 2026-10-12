"""Self-checks for the practice exercises.

Run all:        pytest -q
Run one:        pytest -k ex04 -q
Check solutions (only AFTER you tried):  USE_SOLUTIONS=1 pytest -q
"""
import numpy as np
import pandas as pd
import pytest

import ex01_pandas_warmup as ex01
import ex02_promo_counters as ex02
import ex03_baselines as ex03
import ex04_metrics as ex04
import ex05_backtest as ex05
import ex06_model as ex06
import ex07_uplift_decision as ex07
import ex08_contract as ex08


# ---------------- ex01 ----------------
def test_ex01_weekday_mean():
    df = ex01.make_sales()
    got = ex01.weekday_mean(df)
    # 2026-01-01 is a Thursday (3). Thursdays are rows 0 and 7: (10 + 11) / 2
    assert got.loc[3] == pytest.approx(10.5)
    assert len(got) == 7


def test_ex01_lag():
    df = ex01.make_sales()
    out = ex01.add_lag(df, 7)
    assert "lag7" not in df.columns, "do not change the input frame"
    assert out["lag7"].iloc[:7].isna().all()
    assert out["lag7"].iloc[7] == df["y"].iloc[0]


def test_ex01_rolling_excludes_today():
    df = ex01.make_sales()
    out = ex01.add_rolling_mean(df, 3)

    print(out["roll3"].iloc[0])
    print(type(out["roll3"].iloc[0]))
    print(out["roll3"].dtype)
    assert np.isnan(out["roll3"].iloc[0])
    assert out["roll3"].iloc[3] == pytest.approx(np.mean(df["y"].iloc[0:3]))
    df2 = df.copy()
    df2.loc[5, "y"] = 9999
    out2 = ex01.add_rolling_mean(df2, 3)
    assert out2["roll3"].iloc[5] == out["roll3"].iloc[5], "feature for day 5 must not use day 5"


# ---------------- ex02 ----------------
def test_ex02_example():
    got = ex02.promo_counters([0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    assert list(got.columns) == ["days_into_promo", "days_since_promo_end", "post_promo_week"]
    assert got["days_into_promo"].tolist() == [0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert got["days_since_promo_end"].tolist() == [30, 30, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
    assert got["post_promo_week"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]


def test_ex02_cap_at_30():
    got = ex02.promo_counters([1, 0] + [0] * 45)
    assert got["days_since_promo_end"].max() == 30
    assert got["days_since_promo_end"].iloc[1] == 1


# ---------------- ex03 ----------------
def test_ex03_seasonal_naive():
    got = ex03.seasonal_naive(list(range(1, 15)), h=10, season=7)
    assert got.tolist() == [8, 9, 10, 11, 12, 13, 14, 8, 9, 10]


def test_ex03_base_lift_recovers_known_lift():
    ds = pd.date_range("2026-01-05", periods=56, freq="D")  # starts Monday
    profile = np.array([10, 10, 12, 12, 20, 30, 25], float)
    base = profile[ds.dayofweek]
    promo = np.zeros(56, int)
    promo[[10, 11, 30, 31, 45]] = 1
    y = base * np.where(promo == 1, 2.0, 1.0)
    train = pd.DataFrame({"ds": ds, "y": y, "promo_flag": promo})
    fut_ds = pd.date_range("2026-03-02", periods=7, freq="D")  # Monday
    future = pd.DataFrame({"ds": fut_ds, "promo_flag": [0, 1, 0, 0, 0, 0, 0]})
    got = ex03.base_lift(train, future)
    assert got[0] == pytest.approx(10.0)
    assert got[1] == pytest.approx(20.0)   # Tuesday profile 10, lift 2
    assert got[5] == pytest.approx(30.0)


# ---------------- ex04 ----------------
def test_ex04_wape_bias():
    y, p = [10, 20, 30], [12, 18, 33]
    assert ex04.wape(y, p) == pytest.approx(7 / 60)
    assert ex04.bias(y, p) == pytest.approx(3 / 60)


def test_ex04_pinball():
    assert ex04.pinball([10], [8], 0.9) == pytest.approx(1.8)    # under-predicted, high quantile hurts
    assert ex04.pinball([10], [12], 0.9) == pytest.approx(0.2)   # over-predicted, cheap


def test_ex04_coverage():
    assert ex04.coverage([1, 5, 9, 12], [0, 0, 0, 0], [10, 10, 10, 10]) == pytest.approx(0.75)


# ---------------- ex05 ----------------
def test_ex05_cutoffs():
    got = ex05.make_cutoffs("2026-06-30", h=28, n_windows=4, step=28)
    assert [d.strftime("%Y-%m-%d") for d in got] == ["2026-03-10", "2026-04-07", "2026-05-05", "2026-06-02"]


def test_ex05_split_has_no_overlap():
    df = pd.DataFrame({"ds": pd.date_range("2026-01-01", "2026-06-30", freq="D")})
    cut = pd.Timestamp("2026-06-02")
    train, test = ex05.split_window(df, cut, 28)
    assert train["ds"].max() <= cut < test["ds"].min()
    assert len(test) == 28
    assert test["ds"].max() == pd.Timestamp("2026-06-30")


# ---------------- ex06 ----------------
def _toy_sales():
    rng = np.random.default_rng(0)
    rows = []
    for u in ["a", "b", "c", "d"]:
        ds = pd.date_range("2025-01-01", periods=308, freq="D")
        promo = np.zeros(308, int)
        for s in range(15, 280, 30):
            promo[s:s + 7] = 1
        promo[290:297] = 1                       # a promo inside the forecast window
        dow = np.array([0.9, 0.9, 1.0, 1.0, 1.2, 1.4, 1.2])[ds.dayofweek]
        y = rng.poisson(20 * dow * (1 + 1.5 * promo)).astype(float)
        rows.append(pd.DataFrame({"unique_id": u, "ds": ds, "y": y, "promo_flag": promo}))
    df = pd.concat(rows, ignore_index=True)
    cut = pd.Timestamp("2025-01-01") + pd.Timedelta(days=280)
    return df[df.ds < cut], df[df.ds >= cut][["unique_id", "ds", "promo_flag"]]


def test_ex06_model_learns_promo_effect():
    train, future = _toy_sales()
    out = ex06.fit_predict(train, future, h=28)
    assert set(["unique_id", "ds", "lgbm"]).issubset(out.columns)
    assert (out["lgbm"] >= 0).all()
    m = out.merge(future, on=["unique_id", "ds"])
    on = m.loc[m.promo_flag == 1, "lgbm"].mean()
    off = m.loc[m.promo_flag == 0, "lgbm"].mean()
    assert on > 1.4 * off, f"promo days {on:.1f} vs normal {off:.1f}: did you pass X_df and static_features=[]?"


# ---------------- ex07 ----------------
def test_ex07_promo_off_frame():
    fut = pd.DataFrame({"unique_id": ["a"] * 3, "ds": pd.date_range("2026-07-01", periods=3),
                        "promo_flag": [1, 1, 0], "discount_pct": [0.3, 0.3, 0.0],
                        "price_ratio": [0.7, 0.7, 1.0], "days_into_promo": [1, 2, 0],
                        "days_since_promo_end": [0, 0, 1], "post_promo_week": [0, 0, 1]})
    off = ex07.promo_off_frame(fut)
    assert off["promo_flag"].sum() == 0 and off["discount_pct"].sum() == 0
    assert (off["price_ratio"] == 1.0).all() and (off["days_since_promo_end"] == 30).all()
    assert off["post_promo_week"].sum() == 0 and off["days_into_promo"].sum() == 0
    assert fut["promo_flag"].sum() == 2, "do not change the input frame"
    assert off["unique_id"].tolist() == ["a", "a", "a"]


def test_ex07_uplift_and_stock():
    assert ex07.uplift([50, 20], [30, 25]).tolist() == [20.0, -5.0]
    assert ex07.stock_level([0.1, 0.5, 0.9], [52, 71, 98], cu=4.0, co=1.0) == pytest.approx(91.25)
    with pytest.raises(ValueError):
        ex07.stock_level([0.1, 0.9], [1, 2], cu=0, co=1)


# ---------------- ex08 ----------------
def _feed(days=28):
    d = pd.date_range("2026-11-01", periods=days)
    return pd.DataFrame({"sku_id": "a", "store_id": "s", "date": d, "p10": 1.0, "p50": 2.0, "p90": 3.0})


def test_ex08_good_feed():
    assert ex08.check_feed(_feed(), prev_rows=28) == []


def test_ex08_each_failure():
    f = _feed()
    assert "duplicate keys" in ex08.check_feed(pd.concat([f, f.iloc[:1]]), prev_rows=0)
    bad = f.copy(); bad.loc[0, "p50"] = 0.5
    assert "quantiles not ordered" in ex08.check_feed(bad, prev_rows=0)
    neg = f.copy(); neg.loc[0, ["p10", "p50", "p90"]] = [-1.0, 0.0, 1.0]
    assert "negative values" in ex08.check_feed(neg, prev_rows=0)
    assert "short horizon" in ex08.check_feed(_feed(20), prev_rows=0)
    assert "row count jump" in ex08.check_feed(f, prev_rows=100)
