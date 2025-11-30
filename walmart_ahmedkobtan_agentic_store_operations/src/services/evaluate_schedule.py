import json
import pandas as pd

from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import ARTIFACT_OUT_DIR


def load_artifacts():
    sched_summary_path = ARTIFACT_OUT_DIR / "schedule_summary.json"
    targets_path = ARTIFACT_OUT_DIR / "role_targets_next7d.parquet"
    schedule_path = ARTIFACT_OUT_DIR / "schedule_proposal.parquet"
    summary = json.loads(sched_summary_path.read_text())
    targets = pd.read_parquet(targets_path)
    schedule = pd.read_parquet(schedule_path)
    return summary, targets, schedule


def align_targets(schedule: pd.DataFrame, targets: pd.DataFrame):
    # Prefer matching run_id, but schedule and targets may originate from separate generations.
    if "run_id" in schedule.columns and "run_id" in targets.columns:
        s_ids = schedule.run_id.unique().tolist()
        t_ids = targets.run_id.unique().tolist()
        common = [rid for rid in s_ids if rid in t_ids]
        if common:
            rid = common[-1]
            return rid, schedule[schedule.run_id == rid], targets[targets.run_id == rid]
    return None, schedule, targets


def slice_targets_to_horizon(targets: pd.DataFrame, summary: dict, schedule: pd.DataFrame, run_id):
    horizon_days = summary.get("horizon_days")
    if horizon_days is None:
        return targets
    # If we matched on run_id we assume targets already correspond; still slice to safety.
    if run_id:
        start_day = targets["timestamp_local"].min().normalize()
    else:
        # Derive start from schedule coverage days if no run_id alignment.
        if "day" in schedule.columns and len(schedule):
            start_day = schedule["day"].min().normalize()
        else:
            start_day = targets["timestamp_local"].min().normalize()
    end_day = start_day + pd.Timedelta(days=horizon_days)
    mask = (targets["timestamp_local"] >= start_day) & (targets["timestamp_local"] < end_day)
    return targets.loc[mask].copy()


def coverage_matrix(schedule: pd.DataFrame, targets: pd.DataFrame):
    hours = targets["timestamp_local"].sort_values().unique()
    roles = ["lead", "cashier", "floor"]
    rows = []
    for ts in hours:
        day_norm = ts.normalize()
        for role in roles:
            needed_col = f"{role}_needed"
            needed_row = targets.loc[targets["timestamp_local"] == ts, needed_col]
            if needed_row.empty:
                continue
            needed = int(needed_row.iloc[0])
            staffed = ((schedule["role"] == role) & (schedule["day"] == day_norm) &
                       (schedule["start_hour"] <= ts.hour) & (schedule["end_hour"] > ts.hour)).sum()
            rows.append({
                "timestamp_local": ts,
                "role": role,
                "needed": needed,
                "staffed": int(staffed),
                "shortfall": max(0, needed - staffed),
                "overstaff": max(0, staffed - needed)
            })
    return pd.DataFrame(rows)


def evaluate():
    summary, targets, schedule = load_artifacts()
    run_id, schedule_f, targets_f = align_targets(schedule, targets)
    targets_h = slice_targets_to_horizon(targets_f, summary, schedule_f, run_id)
    cov = coverage_matrix(schedule_f, targets_h)
    agg = cov.groupby("role").agg({"shortfall": "sum", "overstaff": "sum", "needed": "sum"}).reset_index()
    total_shortfall = int(cov["shortfall"].sum())
    total_overstaff = int(cov["overstaff"].sum())
    coverage_ratio = 1 - (total_shortfall / max(cov["needed"].sum(), 1))
    print("Run ID:", run_id or summary.get("run_id"))
    print("Schedule Solver Summary (raw):")
    print(json.dumps(summary, indent=2))
    print("Hours evaluated (after horizon slice):", len(cov))
    print("Derived Coverage Aggregates:")
    print(agg.to_string(index=False))
    print(f"Total shortfall hours: {total_shortfall}")
    print(f"Total overstaff hours: {total_overstaff}")
    print(f"Coverage ratio: {coverage_ratio:.3f}")
    mismatches = cov[(cov["shortfall"] > 0) | (cov["overstaff"] > 0)]
    print(f"Hours with any deviation: {len(mismatches)}")
    if len(mismatches):
        print(mismatches.head(20).to_string(index=False))


if __name__ == "__main__":
    evaluate()
