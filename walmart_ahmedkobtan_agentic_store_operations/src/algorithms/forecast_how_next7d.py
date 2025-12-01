import pandas as pd
from pathlib import Path
from typing import Optional

from walmart_ahmedkobtan_agentic_store_operations.src.algorithms.how_baseline import (
    build_how_quantiles_smoothed,
    forecast_next_7d_from_how,
)
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import (
    PROCESSED_OUT_PATH,
    ARTIFACT_OUT_DIR,
)

OUT_PATH_PARQUET = ARTIFACT_OUT_DIR / Path("df_forecast_next7d_how.parquet")


def forecast_next7d_how(horizon_days: int = 7, seed: Optional[int] = None) -> pd.DataFrame:
    """Return hour-of-week forecast for next N days.

    Reads processed hourly demand, builds smoothed quantiles, and forecasts the next
    7 days, then truncates to the requested horizon.
    """
    df = pd.read_parquet(PROCESSED_OUT_PATH).sort_values("timestamp_local")
    q = build_how_quantiles_smoothed(df, history_weeks=8, recent_days=14)
    fc = forecast_next_7d_from_how(df, q)

    # Ensure expected columns exist; create yhat_p50 if only yhat is present
    if "yhat_p50" not in fc.columns and "yhat" in fc.columns:
        fc = fc.rename(columns={"yhat": "yhat_p50"})
    if "yhat_p10" not in fc.columns:
        fc["yhat_p10"] = fc["yhat_p50"] * 0.9
    if "yhat_p90" not in fc.columns:
        fc["yhat_p90"] = fc["yhat_p50"] * 1.1

    # Limit to requested horizon (assumes chronological order)
    rows = horizon_days * 24
    fc_out = fc.iloc[:rows].copy()
    return fc_out


def main():
    fc = forecast_next7d_how(horizon_days=7)
    OUT_PATH_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(OUT_PATH_PARQUET, index=False)
    print(f"Wrote {OUT_PATH_PARQUET} ({len(fc)} rows)")


if __name__ == "__main__":
    main()
