# core/how_baseline.py
import numpy as np
import pandas as pd

from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import OPEN_HOUR, CLOSE_HOUR

"""
If you decide to squeeze forecast error down after you see the first schedules:


Hour‑of‑Week baseline + month/holiday level multipliers

Keep HoW, but compute separate level multipliers for Nov–Dec vs other months.
Expand history from 8 → 12 weeks in Q4.



LightGBM Quantile model (p10/p50/p90)

Features: hour, dow, doy, is_holiday, lag‑24, lag‑168, rolling 24/168 means.
This often cuts pooled wMAPE by 10–20% on spiky hourly data.



Anomaly down‑weighting

Cap extreme hours (e.g., top 99.5th percentile) during training to prevent single spikes from distorting profiles.
"""


def _restrict_open_hours(df: pd.DataFrame) -> pd.DataFrame:
    h = df["timestamp_local"].dt.hour
    return df[(h >= OPEN_HOUR) & (h < CLOSE_HOUR)].copy()

def build_how_quantiles_smoothed(
    df_hourly: pd.DataFrame,
    history_weeks: int = 8,
    recent_days: int = 14,
    quantiles = (0.1, 0.5, 0.9),
    k: float = 8.0,                # shrinkage strength toward mean
    min_floor: float = 0.10,       # tiny floor to avoid silent zeros
    p90_margin: float = 0.50       # ensure p90 >= p50 + margin
) -> pd.DataFrame:
    """
    Returns a dataframe indexed by (dow, hour) with smoothed q10/q50/q90,
    trend-adjusted and clipped to be monotone with a minimum floor.
    """
    df = _restrict_open_hours(df_hourly).sort_values("timestamp_local").copy()

    # history window
    cutoff_hist = df["timestamp_local"].max() - pd.Timedelta(weeks=history_weeks)
    hist = df[df["timestamp_local"] >= cutoff_hist].copy()
    hist["dow"] = hist["timestamp_local"].dt.dayofweek
    hist["hour"] = hist["timestamp_local"].dt.hour

    # raw quantiles per (dow,hour)
    q_raw = hist.groupby(["dow","hour"])["demand_count"].quantile(quantiles).unstack()
    q_raw.columns = [f"q{int(q*100)}" for q in quantiles]

    # per-bin mean and count
    g = hist.groupby(["dow","hour"])["demand_count"]
    mean_how = g.mean().rename("mean")
    n_how    = g.size().rename("n")

    q = q_raw.join(mean_how, how="outer").join(n_how, how="outer").fillna(0.0)

    # recent vs prior level (trend multiplier)
    cutoff_recent = df["timestamp_local"].max() - pd.Timedelta(days=recent_days)
    prior  = df[df["timestamp_local"] < cutoff_recent]["demand_count"]
    recent = df[df["timestamp_local"] >= cutoff_recent]["demand_count"]
    lvl_prior  = max(1.0, float(prior.mean()))   # guard
    lvl_recent = max(1.0, float(recent.mean()))
    trend = float(np.clip(lvl_recent / lvl_prior if lvl_prior > 0 else 1.0, 0.7, 1.3))

    # empirical-Bayes shrinkage toward the mean_how
    def _shrink(raw, mean, n, k):
        return (raw * n + mean * k) / np.maximum(n + k, 1e-6)

    q["q10_s"] = _shrink(q.get("q10", 0.0), q["mean"], q["n"], k)
    q["q50_s"] = _shrink(q.get("q50", 0.0), q["mean"], q["n"], k)
    q["q90_s"] = _shrink(q.get("q90", 0.0), q["mean"], q["n"], k)

    # enforce monotonicity and floors after applying the trend
    q["q50_s"] = np.maximum(q["q50_s"], q["q10_s"])
    q["q90_s"] = np.maximum(q["q90_s"], q["q50_s"] + p90_margin)

    for c in ["q10_s","q50_s","q90_s"]:
        q[c] = np.maximum(q[c] * trend, min_floor)

    # final columns
    q = q[["q10_s","q50_s","q90_s"]].rename(columns={"q10_s":"q10","q50_s":"q50","q90_s":"q90"})
    return q

def forecast_next_7d_from_how(
    df_hourly: pd.DataFrame,
    how_quantiles: pd.DataFrame,
    start_from: pd.Timestamp | None = None
) -> pd.DataFrame:
    df = _restrict_open_hours(df_hourly)
    last_ts = df["timestamp_local"].max() if start_from is None else pd.Timestamp(start_from)
    start = (last_ts + pd.Timedelta(hours=1)).floor("h")
    end   = start + pd.Timedelta(days=7)

    future = pd.date_range(start, end, freq="h")
    future = future[(future.hour >= OPEN_HOUR) & (future.hour < CLOSE_HOUR)]

    rows = []
    for ts in future:
        key = (ts.dayofweek, ts.hour)
        if key in how_quantiles.index:
            r = how_quantiles.loc[key]
            p10, p50, p90 = r["q10"], r["q50"], r["q90"]
        else:
            # very rare fallback: use overall medians guarded by floors
            p10 = p50 = p90 = max(0.10, df["demand_count"].median())
            p90 = max(p90, p50 + 0.50)
        rows.append({"timestamp_local": ts, "yhat_p10": float(p10),
                     "yhat_p50": float(p50), "yhat_p90": float(p90)})
    return pd.DataFrame(rows).sort_values("timestamp_local")
