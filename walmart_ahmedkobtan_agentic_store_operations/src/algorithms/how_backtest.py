# scripts/backtest_how.py
import pandas as pd
import numpy as np
from walmart_ahmedkobtan_agentic_store_operations.src.algorithms.how_baseline import build_how_quantiles_smoothed, forecast_next_7d_from_how
from walmart_ahmedkobtan_agentic_store_operations.src.utils.metrics import wmape, smape, pinball_loss
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import OPEN_HOUR, CLOSE_HOUR, PROCESSED_OUT_PATH, ARTIFACT_OUT_DIR, MIN_VOL

def _restrict_open_hours(df):
    h = df["timestamp_local"].dt.hour
    return df[(h >= OPEN_HOUR) & (h < CLOSE_HOUR)].copy()

def main():
    df = pd.read_parquet(PROCESSED_OUT_PATH).sort_values("timestamp_local")
    df = _restrict_open_hours(df)

    history_weeks = 8
    step_hours = (CLOSE_HOUR - OPEN_HOUR) * 7

    per_window = []
    pooled_abs_err = 0.0
    pooled_abs_y   = 0.0

    start_idx = history_weeks * step_hours
    end_idx   = len(df) - step_hours

    for i in range(start_idx, end_idx, step_hours):
        hist = df.iloc[:i].copy()
        test = df.iloc[i:i+step_hours].copy()

        q  = build_how_quantiles_smoothed(hist, history_weeks=history_weeks)
        fc = forecast_next_7d_from_how(hist, q, start_from=hist["timestamp_local"].max())

        merged = test.merge(fc, on="timestamp_local", how="inner").dropna(subset=["yhat_p50"])
        y   = merged["demand_count"].values.astype(float)
        p50 = merged["yhat_p50"].values.astype(float)
        p10 = merged["yhat_p10"].values.astype(float)
        p90 = merged["yhat_p90"].values.astype(float)

        # Pooled (for robust overall number)
        pooled_abs_err += np.abs(y - p50).sum()
        pooled_abs_y   += np.abs(y).sum()

        # Per-window (skip degenerate weeks)
        total_y = np.abs(y).sum()
        if total_y >= MIN_VOL:
            per_window.append({
                "window_start": test["timestamp_local"].min(),
                "wMAPE": wmape(y, p50),
                "sMAPE": smape(y, p50),
                "Pinball@0.1": pinball_loss(y, p10, 0.1),
                "Pinball@0.5": pinball_loss(y, p50, 0.5),
                "Pinball@0.9": pinball_loss(y, p90, 0.9),
                "TotalActuals": total_y
            })

    pooled_wMAPE = 100.0 * pooled_abs_err / max(pooled_abs_y, 1e-6)
    print(f"Pooled wMAPE across all windows: {pooled_wMAPE:.2f}%")
    res = pd.DataFrame(per_window)
    print(res.describe(include='all'))
    out = ARTIFACT_OUT_DIR / "how_backtest_summary_smoothed.csv"
    res.to_csv(out, index=False)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
