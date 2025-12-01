from fastapi.testclient import TestClient
from pathlib import Path
import pandas as pd
import tempfile

from walmart_ahmedkobtan_agentic_store_operations.services.compliance_service import app as compliance_app
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import ARTIFACT_OUT_DIR


def _write_csv(df, path):
    df.to_csv(path, index=False)
    return path

def test_compliance_accepts_schedule_format_b():
    client = TestClient(compliance_app)
    # Use existing artifact schedule (format B)
    proposal_csv = Path(ARTIFACT_OUT_DIR) / "schedule_proposal.csv"
    resp = client.post(
        "/validate",
        params={"proposal_path": str(proposal_csv)}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "schema_version" in data
    assert "checked_rules" in data


def test_compliance_flags_break_and_consecutive_days_and_minor_late():
    client = TestClient(compliance_app)
    # Construct synthetic 6-day schedule for one employee with long shifts and a minor late shift
    days = [f"2025-11-{str(d).zfill(2)}" for d in range(1,7)]  # 6 consecutive days
    rows = []
    for d in days:
        # 6.5 hour shift triggers break rule (>=6 threshold)
        rows.append({
            "employee_id": "E100",
            "role": "cashier",
            "day": d,
            "start_hour": 10,
            "end_hour": 16,  # 6h shift (exact threshold) - ensure break rule triggers
            "length": 6
        })
    # Add minor late shift beyond cutoff hour 21
    rows.append({
        "employee_id": "E200",  # minor employee
        "role": "cashier",
        "day": days[-1],
        "start_hour": 15,
        "end_hour": 22,  # past store close / cutoff
        "length": 7
    })
    df_schedule = pd.DataFrame(rows)
    tmp_sched = Path(tempfile.gettempdir()) / "synthetic_schedule.csv"
    _write_csv(df_schedule, tmp_sched)

    # Roster with minor flag
    roster_rows = [
        {"employee_id": "E100", "role": "cashier", "is_minor": False},
        {"employee_id": "E200", "role": "cashier", "is_minor": True},
    ]
    df_roster = pd.DataFrame(roster_rows)
    tmp_roster = Path(tempfile.gettempdir()) / "synthetic_roster.csv"
    _write_csv(df_roster, tmp_roster)

    resp = client.post(
        "/validate",
        params={
            "proposal_path": str(tmp_sched),
            "roster_path": str(tmp_roster),
            "max_consecutive_days": 5,
            "minor_cutoff_hour": 21,
            "break_threshold_hours": 6.0
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    codes = {v["code"] for v in data["violations"]}
    assert "max_consecutive_days" in codes, codes
    assert "break_required" in codes, codes
    assert "minor_late" in codes, codes
    assert data["schema_version"] == "1.0"
    assert "checked_rules" in data and len(data["checked_rules"]) >= 3


def test_compliance_config_override_removes_break_violation(tmp_path):
    client = TestClient(compliance_app)
    # Schedule with 6h shifts which would trigger break at threshold 6 but config sets 7
    rows = [
        {"employee_id": "E300", "role": "cashier", "day": "2025-11-01", "start_hour": 10, "end_hour": 16, "length": 6},
        {"employee_id": "E300", "role": "cashier", "day": "2025-11-02", "start_hour": 10, "end_hour": 16, "length": 6},
        {"employee_id": "E300", "role": "cashier", "day": "2025-11-03", "start_hour": 10, "end_hour": 16, "length": 6},
        {"employee_id": "E300", "role": "cashier", "day": "2025-11-04", "start_hour": 10, "end_hour": 16, "length": 6},
    ]
    df = pd.DataFrame(rows)
    sched_path = tmp_path / "sched.csv"
    df.to_csv(sched_path, index=False)
    cfg_path = Path("walmart_ahmedkobtan_agentic_store_operations/data/configs/compliance_rules.yaml")
    resp = client.post(
        "/validate",
        params={
            "proposal_path": str(sched_path),
            "config_path": str(cfg_path)
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    codes = {v["code"] for v in data["violations"]}
    # With config override break threshold=7.0, no break_required violation expected
    assert "break_required" not in codes, codes
    assert data["schema_version"] == "1.0"
    assert "config_override" in data["checked_rules"]
    assert data.get("rules_version") == "1.0"
