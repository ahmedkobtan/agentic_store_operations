import pandas as pd
from pathlib import Path

from walmart_ahmedkobtan_agentic_store_operations.src.algorithms.schedule_solver import propose_schedule
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import ARTIFACT_OUT_DIR


def _targets_path() -> str:
    # Prefer parquet if exists else csv
    pq = ARTIFACT_OUT_DIR / "role_targets_next7d.parquet"
    if pq.exists():
        return str(pq)
    csv = ARTIFACT_OUT_DIR / "role_targets_next7d.csv"
    return str(csv)


def _constraints_path() -> str:
    return str(Path("walmart_ahmedkobtan_agentic_store_operations") / "data" / "configs" / "constraints.yaml")


def _roster_path() -> Path:
    # Actual roster location (not under raw/)
    return Path("walmart_ahmedkobtan_agentic_store_operations") / "data" / "roster" / "employees.csv"


def test_horizon_slice(tmp_path):
    """Ensure horizon_days=1 slices targets correctly (lead demand hours == open-close)."""
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = propose_schedule(
        role_targets_path=_targets_path(),
        roster_csv=str(_roster_path()),
        constraints_yaml=_constraints_path(),
        horizon_days=1,
        out_dir=str(out_dir),
        time_limit_s=5,
    )["summary"]
    open_h = summary["open_hour"]
    close_h = summary["close_hour"]
    expected_lead_hours = close_h - open_h
    assert summary["demand_hours_by_role"]["lead"] == expected_lead_hours, (
        "Lead demand hours mismatch; horizon slice may be incorrect"
    )


def test_cashier_shortfall_detected(tmp_path):
    """Remove all cashiers to force shortfall; solver should report unmet cashier demand > 0."""
    roster_df = pd.read_csv(_roster_path())
    no_cashiers = roster_df[roster_df["role"] != "cashier"].copy()
    assert (no_cashiers["role"] == "cashier").sum() == 0, "Cashiers still present in filtered roster"
    roster_tmp = tmp_path / "roster.csv"
    no_cashiers.to_csv(roster_tmp, index=False)

    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = propose_schedule(
        role_targets_path=_targets_path(),
        roster_csv=str(roster_tmp),
        constraints_yaml=_constraints_path(),
        horizon_days=1,
        out_dir=str(out_dir),
        time_limit_s=5,
    )["summary"]

    cashier_shortfall = summary["shortfall_by_role_hours"]["cashier"]
    assert cashier_shortfall > 0, "Expected cashier shortfall > 0 when no cashiers available"
    # Lead role should remain fully covered due to enforced minimum
    assert summary["shortfall_by_role_hours"]["lead"] == 0, "Lead shortfall should be zero"
