"""End-to-end demo of the promo demand forecasting flow on SYNTHETIC data.

Run:  python pipeline_demo.py
Output: figures/fig01..fig08 PNG files and printed tables.

Purpose: let you SEE each step of the pipeline (data -> features -> baselines ->
LightGBM -> backtest -> uplift -> intervals -> stock decision). Numbers come from
simulated data, so they show how the method behaves, not what your real accuracy will be.
"""
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean, RollingStd
from mlforecast.utils import PredictionIntervals
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
H, N_WINDOWS, SEED = 28, 4, 42
CAL_WINDOWS = 10   # windows used to calibrate the conformal interval

# ---- chart style (validated reference palette, light mode) -----------------
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})


def short_dates(ax, n=6):
    import matplotlib.dates as md
    ax.xaxis.set_major_locator(md.AutoDateLocator(maxticks=n))
    ax.xaxis.set_major_formatter(md.DateFormatter("%d %b"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("saved", name)


# =============================================================================
# STEP 1. Simulated data: 24 products, daily, with promotions, a post-promo dip
# =============================================================================
def simulate(n_series=24, start="2024-01-01", end="2026-06-30"):
    rng = np.random.default_rng(SEED)
    days = pd.date_range(start, end, freq="D")
    rows = []
    for i in range(n_series):
        uid = f"SKU{i:02d}"
        cat = ["snacks", "drinks", "care"][i % 3]
        base = rng.uniform(6, 40)
        reg_price = rng.uniform(2, 9)
        elasticity = rng.uniform(1.8, 4.5)        # how strongly this SKU reacts to discounts
        dow = np.array([0.85, 0.85, 0.9, 0.95, 1.1, 1.3, 1.15])
        # promotions: 7-14 days long, about 7 per year, discount 10-40 percent
        promo = np.zeros(len(days)); disc = np.zeros(len(days))
        t = int(rng.integers(20, 50))
        while t < len(days) - 14:
            length = int(rng.integers(7, 15)); d = rng.choice([0.1, 0.2, 0.3, 0.4])
            promo[t:t + length] = 1; disc[t:t + length] = d
            t += length + int(rng.integers(25, 70))
        # post-promo dip: 7 days after a promo end, demand falls (stock-piling)
        post = np.zeros(len(days))
        for k in range(1, len(days)):
            if promo[k] == 0 and promo[k - 1] == 1:
                post[k:k + 7] = 1
        season = 1 + 0.15 * np.sin(2 * np.pi * (np.arange(len(days)) - 60) / 365.25)
        trend = 1 + 0.0002 * np.arange(len(days))
        baseline = base * dow[days.dayofweek] * season * trend           # true no-promo demand
        lift = 1 + elasticity * disc                                      # true promo effect
        dip = 1 - 0.25 * post
        mean = baseline * lift * dip
        y = rng.poisson(mean).astype(float)
        rows.append(pd.DataFrame({
            "unique_id": uid, "ds": days, "y": y, "category": cat,
            "promo_flag": promo.astype(int), "discount_pct": disc,
            "price": reg_price * (1 - disc), "reg_price": reg_price,
            "true_baseline": baseline, "true_mean": mean,
        }))
    df = pd.concat(rows, ignore_index=True)
    df["category"] = df["category"].astype("category")
    return df


# =============================================================================
# STEP 2. Feature engineering that must be knowable for FUTURE dates
# =============================================================================
def add_promo_features(df, flag="promo_flag", disc="discount_pct", price="price", reg="reg_price"):
    """Features derived from the PLANNED promo calendar, so they exist for future days too."""
    df = df.sort_values(["unique_id", "ds"]).copy()
    g = df.groupby("unique_id", group_keys=False)

    def per_series(s):
        f = s[flag].to_numpy()
        into = np.zeros(len(f)); since = np.full(len(f), 30.0)
        c = 0; last_end = None
        for k in range(len(f)):
            c = c + 1 if f[k] else 0
            into[k] = c
            if k > 0 and f[k - 1] == 1 and f[k] == 0:
                last_end = k
            since[k] = min(k - last_end + 1, 30) if (last_end is not None and not f[k]) else (0 if f[k] else 30)
        out = s.copy()
        out["days_into_promo"] = into
        out["days_since_promo_end"] = since
        out["post_promo_week"] = ((since >= 1) & (since <= 7) & (f == 0)).astype(int)
        return out

    df = g.apply(per_series, include_groups=True)
    df["price_ratio"] = df[price] / df[reg]
    return df.reset_index(drop=True)


DYN = ["promo_flag", "discount_pct", "price_ratio", "days_into_promo", "days_since_promo_end", "post_promo_week"]


# =============================================================================
# STEP 3. Models
# =============================================================================
def make_lgbm():
    return MLForecast(
        models={"lgbm": lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2,
                                          n_estimators=400, learning_rate=0.05, num_leaves=31,
                                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                          random_state=SEED, verbosity=-1)},
        freq="D", lags=[7, 14, 28],
        lag_transforms={7: [RollingMean(window_size=28), RollingStd(window_size=28)], 1: [ExpandingMean()]},
        date_features=["dayofweek", "month"],
    )


def base_lift_forecast(train, future):
    """Classic base-lift: baseline = day-of-week profile of recent non-promo days,
    lift = average ratio of promo-day sales to that baseline, per series."""
    out = []
    for uid, tr in train.groupby("unique_id"):
        recent = tr.tail(84)
        clean = recent[(recent.promo_flag == 0) & (recent.post_promo_week == 0)]
        prof = clean.groupby(clean.ds.dt.dayofweek)["y"].mean()
        prof = prof.reindex(range(7)).fillna(clean["y"].mean())
        hist = tr[tr.promo_flag == 1]
        hb = hist.ds.dt.dayofweek.map(prof)
        lift = float((hist.y.sum() / hb.sum())) if len(hist) and hb.sum() > 0 else 1.0
        fu = future[future.unique_id == uid].copy()
        b = fu.ds.dt.dayofweek.map(prof).to_numpy()
        fu["base_lift"] = b * np.where(fu.promo_flag == 1, lift, 1.0)
        out.append(fu[["unique_id", "ds", "base_lift"]])
    return pd.concat(out)


# =============================================================================
# STEP 4. Metrics
# =============================================================================
def wape(y, p): return float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-9))
def bias(y, p): return float((p - y).sum() / max(y.sum(), 1e-9))


def main():
    # ---------------- step 1
    raw = simulate()
    print("STEP 1 raw data sample:\n", raw[["unique_id", "ds", "y", "promo_flag", "discount_pct", "price"]].head(3))
    one = raw[raw.unique_id == "SKU03"].iloc[:240]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(one.ds, one.y, color=BLUE, lw=1.5, label="Units sold")
    on = one.promo_flag.to_numpy() == 1
    ax.fill_between(one.ds, 0, one.y.max() * 1.05, where=on, color=ORANGE, alpha=0.18, lw=0, label="Promotion running")
    ax.set_title("Stage 1. Raw data: one product, daily sales with promotion periods")
    ax.set_ylabel("Units per day"); ax.legend(loc="upper left", ncol=2)
    save(fig, "fig01_raw_series.png")

    # ---------------- step 2
    df = add_promo_features(raw)
    sample = df[df.unique_id == "SKU03"].iloc[30:75]
    fig, axs = plt.subplots(3, 1, figsize=(10, 5.2), sharex=True)
    axs[0].plot(sample.ds, sample.y, color=BLUE, lw=1.5); axs[0].set_ylabel("Units")
    axs[0].set_title("Stage 2. Features: what the model sees for each day")
    axs[1].step(sample.ds, sample.days_into_promo, color=ORANGE, lw=1.5, where="post"); axs[1].set_ylabel("Days into\npromo")
    axs[2].step(sample.ds, sample.days_since_promo_end, color=AQUA, lw=1.5, where="post"); axs[2].set_ylabel("Days since\npromo ended")
    pp = sample.post_promo_week.to_numpy() == 1
    for a in axs:
        a.fill_between(sample.ds, a.get_ylim()[0], a.get_ylim()[1], where=pp, color=GRAY, alpha=0.18, lw=0)
    axs[0].text(sample.ds[pp].iloc[0], axs[0].get_ylim()[1] * 0.92, " post-promo\n week", color=INK2, fontsize=9, va="top")
    save(fig, "fig02_features.png")

    # ---------------- step 3: rolling-origin backtest picture
    last = df.ds.max()
    cutoffs = [last - pd.Timedelta(days=int(H * (N_WINDOWS - k))) for k in range(N_WINDOWS)]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    start = df.ds.min()
    for k, c in enumerate(cutoffs):
        ax.barh(k, (c - start).days, left=mdates_num(start), color=BLUE, height=0.55)
        ax.barh(k, H, left=mdates_num(c), color=ORANGE, height=0.55)
    ax.set_yticks(range(N_WINDOWS)); ax.set_yticklabels([f"Window {k+1}" for k in range(N_WINDOWS)])
    ax.invert_yaxis(); ax.set_xlim(mdates_num(start) + 500, mdates_num(last) + 5)
    ax.xaxis_date(); ax.grid(axis="y", visible=False)
    ax.set_title("Stage 3. Rolling-origin backtest: train on the past (blue), test on the next 28 days (orange)")
    save(fig, "fig03_backtest_windows.png")

    # ---------------- step 4/5: backtest all models on the same windows
    cv_base = df[["unique_id", "ds", "y"]]
    sf = StatsForecast(models=[SeasonalNaive(season_length=7)], freq="D", n_jobs=1)
    cv_sn = sf.cross_validation(df=cv_base, h=H, n_windows=N_WINDOWS, step_size=H).reset_index(drop=True)

    fc = make_lgbm()
    cv_lg = fc.cross_validation(df[["unique_id", "ds", "y", "category"] + DYN], n_windows=N_WINDOWS, h=H,
                                step_size=H, static_features=["category"],
                                prediction_intervals=PredictionIntervals(n_windows=CAL_WINDOWS, h=H), level=[80])
    cv = cv_lg.merge(cv_sn[["unique_id", "ds", "SeasonalNaive"]], on=["unique_id", "ds"])
    # base-lift per window
    bl = []
    for c in cv.cutoff.unique():
        tr = df[df.ds <= c]; fu = df[(df.ds > c) & (df.ds <= c + pd.Timedelta(days=int(H)))]
        bl.append(base_lift_forecast(tr, fu))
    cv = cv.merge(pd.concat(bl), on=["unique_id", "ds"])
    cv = cv.merge(df[["unique_id", "ds", "promo_flag", "post_promo_week", "true_baseline"]], on=["unique_id", "ds"])
    cv["lgbm"] = cv["lgbm"].clip(lower=0)

    models = {"Seasonal naive": "SeasonalNaive", "Base-lift": "base_lift", "LightGBM + promo features": "lgbm"}
    segs = {"All days": cv.index == cv.index,
            "Promo days": cv.promo_flag == 1,
            "Post-promo week": cv.post_promo_week == 1,
            "Normal days": (cv.promo_flag == 0) & (cv.post_promo_week == 0)}
    res = pd.DataFrame({m: {s: wape(cv.y[mask], cv[c][mask]) for s, mask in segs.items()} for m, c in models.items()})
    bres = pd.DataFrame({m: {s: bias(cv.y[mask], cv[c][mask]) for s, mask in segs.items()} for m, c in models.items()})
    print("\nWAPE by segment (lower is better):\n", res.round(3))
    print("\nBias by segment:\n", bres.round(3))
    cover = float(((cv.y >= cv["lgbm-lo-80"]) & (cv.y <= cv["lgbm-hi-80"])).mean())
    print(f"\n80% interval coverage (LightGBM, conformal): {cover:.3f}")
    res.round(4).to_csv(OUT.parent / "demo_wape_by_segment.csv")

    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.26; x = np.arange(len(res.index)); cols = [GRAY, AQUA, BLUE]
    for k, (m, col) in enumerate(zip(res.columns, cols)):
        ax.bar(x + (k - 1) * w, res[m] * 100, width=w * 0.9, color=col, label=m)
    ax.set_xticks(x); ax.set_xticklabels(res.index); ax.set_ylabel("WAPE, % (lower is better)")
    ax.legend(loc="upper left", ncol=3); ax.grid(axis="x", visible=False)
    ax.set_title("Stage 4. Scorecard: report error by segment, not one global number")
    save(fig, "fig04_scorecard_by_segment.png")

    # ---------------- step 6: forecast with interval on the last window for the busiest promo series
    last_w = cv[cv.cutoff == cv.cutoff.max()]
    pick = last_w.groupby("unique_id").promo_flag.sum().idxmax()
    s = last_w[last_w.unique_id == pick].sort_values("ds")
    hist = df[(df.unique_id == pick) & (df.ds > s.ds.min() - pd.Timedelta(days=56)) & (df.ds < s.ds.min())]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist.ds, hist.y, color=GRAY, lw=1.3, label="History")
    ax.fill_between(s.ds, s["lgbm-lo-80"], s["lgbm-hi-80"], color=BLUE, alpha=0.18, lw=0, label="80% interval")
    ax.plot(s.ds, s.y, color=INK, lw=1.3, label="Actual")
    ax.plot(s.ds, s.lgbm, color=BLUE, lw=2, label="Forecast (median)")
    on = s.promo_flag.to_numpy() == 1
    ax.fill_between(s.ds, 0, max(s.y.max(), s["lgbm-hi-80"].max()) * 1.05, where=on, color=ORANGE, alpha=0.14, lw=0, label="Promotion")
    ax.set_title(f"Stage 5. Forecast with range for {pick} (28 days after the cut-off)")
    ax.set_ylabel("Units per day"); ax.legend(ncol=5, loc="upper left")
    short_dates(ax)
    save(fig, "fig05_forecast_with_interval.png")

    # ---------------- step 7: uplift = forecast with promo minus forecast with promo off
    cut = df.ds.max() - pd.Timedelta(days=int(H))
    train = df[df.ds <= cut]; fut = df[df.ds > cut]
    fcs = make_lgbm()
    fcs.fit(train[["unique_id", "ds", "y", "category"] + DYN], static_features=["category"])
    # scenario A: real promo plan
    pa = fcs.predict(H, X_df=fut[["unique_id", "ds"] + DYN])
    # scenario B: promo switched OFF (all promo features reset)
    off = fut[["unique_id", "ds"] + DYN].copy()
    off[["promo_flag", "discount_pct", "days_into_promo", "post_promo_week"]] = 0
    off["price_ratio"] = 1.0; off["days_since_promo_end"] = 30.0
    pb = fcs.predict(H, X_df=off)
    up = pa.merge(pb, on=["unique_id", "ds"], suffixes=("_promo", "_off"))
    up = up.merge(fut[["unique_id", "ds", "y", "promo_flag", "true_baseline", "true_mean"]], on=["unique_id", "ds"])
    up["uplift"] = up.lgbm_promo - up.lgbm_off
    up["true_uplift"] = up.true_mean - up.true_baseline
    print("\nUplift check, units over the 28 days (all promoted days):")
    pm = up.promo_flag == 1
    print(f"  predicted uplift {up.uplift[pm].sum():.0f}  vs true uplift {up.true_uplift[pm].sum():.0f}")

    focus = up[up.unique_id == pick].sort_values("ds")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(focus.ds, focus.lgbm_promo, color=BLUE, lw=2, label="Forecast with promotion")
    ax.plot(focus.ds, focus.lgbm_off, color=GRAY, lw=2, ls="--", label="Baseline: promotion switched off")
    ax.fill_between(focus.ds, focus.lgbm_off, focus.lgbm_promo, where=focus.lgbm_promo >= focus.lgbm_off,
                    color=ORANGE, alpha=0.3, lw=0, label="Promo uplift")
    ax.fill_between(focus.ds, focus.lgbm_off, focus.lgbm_promo, where=focus.lgbm_promo < focus.lgbm_off,
                    color=AQUA, alpha=0.35, lw=0, label="Post-promo dip")
    ax.set_title(f"Stage 6. Uplift for {pick}: same model run twice, promo on vs promo off")
    ax.set_ylabel("Units per day"); ax.legend(ncol=2, loc="upper left")
    short_dates(ax)
    save(fig, "fig06_uplift_on_vs_off.png")

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    by = up[pm].groupby("unique_id")[["uplift", "true_uplift"]].sum()
    ax.scatter(by.true_uplift, by.uplift, color=BLUE, s=36, edgecolor=SURF, linewidth=1.2)
    mx = float(max(by.max())) * 1.05
    ax.plot([0, mx], [0, mx], color=GRAY, lw=1, ls="--")
    ax.set_xlabel("True uplift (known only in simulation)"); ax.set_ylabel("Predicted uplift")
    ax.set_title("Stage 6b. Does predicted uplift match truth?")
    save(fig, "fig07_uplift_check.png")

    # ---------------- step 8: stock decision from quantiles with costs
    fig, ax = plt.subplots(figsize=(8, 3.8))
    sub = cv[(cv.cutoff == cv.cutoff.max())].groupby("unique_id")[["y", "lgbm", "lgbm-hi-80"]].sum()
    ax.scatter(sub.lgbm, sub.y, color=BLUE, s=34, edgecolor=SURF, linewidth=1.2, label="Product (28-day total)")
    mx = float(sub[["y", "lgbm"]].values.max()) * 1.05
    ax.plot([0, mx], [0, mx], color=GRAY, ls="--", lw=1)
    ax.set_xlabel("Forecast, 28-day total (median)"); ax.set_ylabel("Actual, 28-day total")
    ax.set_title("Stage 7. Aggregated forecast vs actual per product"); ax.legend(loc="upper left")
    save(fig, "fig08_forecast_vs_actual_totals.png")

    # feature importance
    gain = fcs.models_["lgbm"].booster_.feature_importance(importance_type="gain")
    imp = pd.Series(gain / gain.sum() * 100, index=fcs.ts.features_order_).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh(imp.index, imp.values, color=BLUE, height=0.65); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of total gain, %"); ax.set_title("Stage 8. What drives the LightGBM forecast (gain, not split count)")
    save(fig, "fig09_feature_importance.png")
    print("\nTop features:", list(imp.sort_values(ascending=False).head(6).index))


def mdates_num(ts):
    import matplotlib.dates as md
    return md.date2num(ts)


if __name__ == "__main__":
    main()
