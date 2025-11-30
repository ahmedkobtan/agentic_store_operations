from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import pandas as pd
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import SCHEMA_VERSION
from walmart_ahmedkobtan_agentic_store_operations.src.utils.yaml_utils import read_yaml


class Violation(BaseModel):
    code: str
    message: str


class ComplianceResponse(BaseModel):
    pass_: bool
    violations: List[Violation]
    schema_version: str
    checked_rules: List[str]


app = FastAPI(title="Compliance Service")


@app.post("/validate", response_model=ComplianceResponse)
def validate_schedule(
    proposal_path: str,
    roster_path: Optional[str] = None,
    config_path: Optional[str] = None,
    max_consecutive_days: int = 5,
    break_threshold_hours: float = 6.0,
    break_min_minutes: int = 30,
    minor_cutoff_hour: int = 21
) -> ComplianceResponse:
    """Validate schedule against structural + labor rules.

    Supports two artifact schemas:
      Format A: employee_id, role, start_ts, end_ts, hours
      Format B: employee_id, role, day, start_hour, end_hour, length

    Additional rules implemented:
      - max_consecutive_days: employees scheduled more than N successive days.
      - break_required: shifts with length >= break_threshold_hours assumed to need a break (placeholder until explicit break modeling).
      - minor_late: minor employees (is_minor==True or age<18) ending after minor_cutoff_hour.
    """
    violations: List[Violation] = []
    checked: List[str] = []
    try:
        path = Path(proposal_path)
        if path.suffix != ".csv":
            violations.append(Violation(code="format", message="Only CSV proposals supported"))
            return ComplianceResponse(pass_=False, violations=violations, schema_version=SCHEMA_VERSION, checked_rules=checked)

        df = pd.read_csv(path)

        # External rule config (YAML) override if provided
        if config_path:
            try:
                cfg = read_yaml(config_path)
                max_consecutive_days = int(cfg.get("max_consecutive_days", max_consecutive_days))
                break_threshold_hours = float(cfg.get("break_threshold_hours", break_threshold_hours))
                break_min_minutes = int(cfg.get("break_min_minutes", break_min_minutes))
                minor_cutoff_hour = int(cfg.get("minor_cutoff_hour", minor_cutoff_hour))
                checked.append("config_override")
            except Exception as e:
                violations.append(Violation(code="config_read", message=f"Config read error: {e}"))

        # Detect schema format
        format_a = {"employee_id", "role", "start_ts", "end_ts", "hours"}.issubset(df.columns)
        format_b = {"employee_id", "role", "day", "start_hour", "end_hour", "length"}.issubset(df.columns)
        if not (format_a or format_b):
            violations.append(Violation(code="schema", message="Unrecognized schedule format"))
            return ComplianceResponse(pass_=False, violations=violations, schema_version=SCHEMA_VERSION, checked_rules=checked)

        # Normalize to canonical columns start_ts/end_ts/hours
        if format_b and not format_a:
            df["start_ts"] = pd.to_datetime(df["day"] + " " + df["start_hour"].astype(str) + ":00:00")
            df["end_ts"] = pd.to_datetime(df["day"] + " " + df["end_hour"].astype(str) + ":00:00")
            df["hours"] = df.get("length", (df["end_hour"] - df["start_hour"]))
        elif format_a:
            df["start_ts"] = pd.to_datetime(df["start_ts"])
            df["end_ts"] = pd.to_datetime(df["end_ts"])
            df["hours"] = df["hours"].astype(float)

        # Basic structural checks
        checked.append("schema_basic")
        if (df["hours"] < 0).any():
            violations.append(Violation(code="hours", message="Negative hours detected"))

        # Max consecutive days rule
        checked.append("max_consecutive_days")
        df["day_norm"] = df["start_ts"].dt.date
        streak_violators = []
        for emp_id, g in df.groupby("employee_id"):
            days_sorted = sorted(set(g["day_norm"]))
            longest = 0
            current = 0
            prev = None
            for d in days_sorted:
                if prev is None or (d - prev).days == 1:
                    current += 1
                else:
                    current = 1
                longest = max(longest, current)
                prev = d
            if longest > max_consecutive_days:
                streak_violators.append((emp_id, longest))
        if streak_violators:
            violations.append(Violation(code="max_consecutive_days", message=f"Employees exceed max consecutive days: {streak_violators}"))

        # Break requirement rule
        checked.append("break_required")
        long_shifts = df[df["hours"] >= break_threshold_hours]
        if not long_shifts.empty:
            violations.append(Violation(code="break_required", message=f"{len(long_shifts)} shifts >= {break_threshold_hours}h need >= {break_min_minutes}m break (not modeled)"))

        # Minor late rule
        if roster_path:
            try:
                roster_df = pd.read_csv(roster_path)
                is_minor_col = None
                if "is_minor" in roster_df.columns:
                    is_minor_col = "is_minor"
                elif "age" in roster_df.columns:
                    roster_df["is_minor"] = roster_df["age"] < 18
                    is_minor_col = "is_minor"
                if is_minor_col:
                    checked.append("minor_late")
                    minor_ids = set(roster_df[roster_df["is_minor"] == True]["employee_id"].tolist())
                    df["end_hour_val"] = df["end_ts"].dt.hour
                    late_minors = df[(df["employee_id"].isin(minor_ids)) & (df["end_hour_val"] > minor_cutoff_hour)]
                    if not late_minors.empty:
                        violations.append(Violation(code="minor_late", message=f"Minor shifts ending after {minor_cutoff_hour}: {[ (r.employee_id, r.end_hour_val) for r in late_minors.itertuples() ]}"))
            except Exception as e:
                violations.append(Violation(code="roster_read", message=f"Roster read error: {e}"))

        passed = len(violations) == 0
        return ComplianceResponse(pass_=passed, violations=violations, schema_version=SCHEMA_VERSION, checked_rules=checked)
    except Exception as e:
        violations.append(Violation(code="error", message=str(e)))
        return ComplianceResponse(pass_=False, violations=violations, schema_version=SCHEMA_VERSION, checked_rules=checked)
