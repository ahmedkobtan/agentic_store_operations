
# algorithms/make_role_targets.py
# Patch 1 applied: regularize to a full 7-day open-hours grid and join forecast onto it.
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import (
    ARTIFACT_OUT_DIR,
    OPEN_HOUR,
    CLOSE_HOUR,
    TX_PER_CASHIER,
    CASHIER_TRIGGER,
    PROMOTE_TO_P90_THRESHOLD,
    WEEKDAYS,
    ALWAYS_P90_CFG,
    LEAD_ALL_OPEN,
    BASE_Q,
)
import uuid

# I/O paths
FORECAST_PATH = ARTIFACT_OUT_DIR / "df_forecast_next7d_how.parquet"
OUTPUT_PATH   = ARTIFACT_OUT_DIR / "role_targets_next7d.parquet"


# --------------------------- helpers ---------------------------

def load_forecast(path: Path) -> pd.DataFrame:
    """
    Load forecast (parquet preferred, CSV fallback) with a timestamp column 'timestamp_local'.
    Returns a DataFrame sorted by timestamp.
    """
    if path.exists() and path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path.with_suffix(".csv"), parse_dates=["timestamp_local"])
    # Defensive: ensure dtype and sort
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])
    df = df.sort_values("timestamp_local").reset_index(drop=True)
    return df


def in_always_p90(ts: pd.Timestamp, always_p90_cfg: Optional[list]) -> bool:
    """
    Check if a timestamp falls in any 'always_p90' window:
    [{'dow':'fri','start_hour':10,'end_hour':14}, ...]
    """
    if not always_p90_cfg:
        return False
    d = WEEKDAYS[ts.weekday()]  # 'mon'...'sun'
    h = ts.hour
    for w in always_p90_cfg:
        if d == w.get("dow") and (h >= int(w["start_hour"])) and (h < int(w["end_hour"])):
            return True
    return False


def build_full_open_hours_grid(first_ts: pd.Timestamp,
                               open_h: int,
                               close_h: int,
                               days: int = 7) -> pd.DataFrame:
    """
    Build a complete hourly grid for [open_h, close_h) over 'days' starting
    at the first forecast day (normalized to midnight + open_h).
    """
    first_day = first_ts.normalize()
    start = first_day + pd.Timedelta(hours=open_h)
    end   = start + pd.Timedelta(days=days)   # exclusive
    full_idx = pd.date_range(start, end, freq="h")
    # keep only open hours per day
    full_idx = full_idx[(full_idx.hour >= open_h) & (full_idx.hour < close_h)]
    grid = pd.DataFrame({"timestamp_local": full_idx}).set_index("timestamp_local")
    return grid


def build_targets_for_tx(fc: pd.DataFrame,
                         guardrails: Optional[Dict[str, Any]] = None,
                         tx: float = TX_PER_CASHIER) -> pd.DataFrame:
    """
    Patch 1 implementation:
      * Reindex the forecast to a full 7-day open-hour grid.
      * Fill missing forecast hours with zeros (so we get rows).
      * Apply quantile promotion (uncertainty + always_p90 windows).
      * Map demand to role targets, enforcing lead=1 at all open hours.
    """
    # Resolve configuration (constants or guardrails override)
    open_h        = int(guardrails.get("store", {}).get("open_hour", OPEN_HOUR)) if guardrails else OPEN_HOUR
    close_h       = int(guardrails.get("store", {}).get("close_hour", CLOSE_HOUR)) if guardrails else CLOSE_HOUR
    promote_thr   = float(guardrails.get("quantiles", {}).get("promote_to_p90_if_rel_uncertainty_ge",
                                                              PROMOTE_TO_P90_THRESHOLD)) if guardrails else PROMOTE_TO_P90_THRESHOLD
    trigger       = float(guardrails.get("cashier_mapping", {}).get("cashier_trigger_tx_per_hour",
                                                                    CASHIER_TRIGGER)) if guardrails else CASHIER_TRIGGER
    always_p90    = guardrails.get("always_p90", ALWAYS_P90_CFG) if guardrails else ALWAYS_P90_CFG
    lead_all_open = bool(guardrails.get("lead_policy", {}).get("lead_required_all_open_hours",
                                                               LEAD_ALL_OPEN)) if guardrails else LEAD_ALL_OPEN
    base_q        = float(guardrails.get("quantiles", {}).get("base_quantile", BASE_Q)) if guardrails else BASE_Q

    # --- Build full 7-day open-hours grid starting at the first forecast day ---
    if fc.empty:
        # Edge case: no forecast—create an empty grid for today
        first_ts = pd.Timestamp.today().normalize() + pd.Timedelta(hours=open_h)
        grid = build_full_open_hours_grid(first_ts, open_h, close_h, days=7)
        joined = grid.copy()
        joined[["yhat_p10", "yhat_p50", "yhat_p90"]] = 0.0
    else:
        grid   = build_full_open_hours_grid(fc["timestamp_local"].iloc[0], open_h, close_h, days=7)
        joined = grid.join(fc.set_index("timestamp_local")[["yhat_p10", "yhat_p50", "yhat_p90"]], how="left")
        # Missing hours → zero demand
        joined[["yhat_p10", "yhat_p50", "yhat_p90"]] = joined[["yhat_p10", "yhat_p50", "yhat_p90"]].fillna(0.0)

    # Choose base quantile column
    base_col = "yhat_p50" if base_q == 0.5 else ("yhat_p90" if base_q == 0.9 else "yhat_p10")

    # Relative uncertainty (clip denominator to avoid explosion at tiny counts)
    rel_unc = (joined["yhat_p90"] - joined["yhat_p50"]) / np.maximum(joined["yhat_p50"], 1.0)

    # Demand used for staffing: promote to p90 on uncertainty or in always_p90 windows
    idx_series = joined.index.to_series()
    joined["demand_for_staffing"] = np.where(
        (rel_unc >= promote_thr) | (idx_series.apply(lambda x: in_always_p90(x, always_p90))),
        joined["yhat_p90"],
        joined[base_col]
    )

    # Cashier mapping with trigger threshold and throughput
    joined["cashier_needed"] = np.where(
        joined["demand_for_staffing"] >= trigger,
        np.ceil(joined["demand_for_staffing"] / float(tx)),
        0
    ).astype(int)

    # Floor is 50% of cashiers (round up)
    joined["floor_needed"] = np.ceil(0.5 * joined["cashier_needed"]).astype(int)

    # Lead baseline coverage (per policy): 1 at all open hours
    joined["lead_needed"] = 1 if lead_all_open else 0

    # Return tidy frame
    out = joined.reset_index()[["timestamp_local", "lead_needed", "cashier_needed", "floor_needed"]]
    return out


# --------------------------- main ---------------------------

def main():
    run_id = str(uuid.uuid4())
    fc_forecast = load_forecast(FORECAST_PATH)
    targets     = build_targets_for_tx(fc_forecast, guardrails=None, tx=TX_PER_CASHIER)
    targets["run_id"] = run_id

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(OUTPUT_PATH, index=False)
    # write a lightweight metadata json
    meta_path = ARTIFACT_OUT_DIR / "role_targets_meta.json"
    meta = {
        "run_id": run_id,
        "rows": int(len(targets)),
        "start": str(targets["timestamp_local"].min()),
        "end": str(targets["timestamp_local"].max()),
        "tx_per_cashier": TX_PER_CASHIER,
    }
    with open(meta_path, "w") as f:
        import json; json.dump(meta, f, indent=2)

    targets["day"] = targets["timestamp_local"].dt.normalize()
    by_day = targets.groupby("day")[["lead_needed", "cashier_needed", "floor_needed"]].sum()
    print(by_day.to_string())
    print(f"Saved {OUTPUT_PATH} ({len(targets)} rows) run_id={run_id}")


if __name__ == "__main__":
    main()
