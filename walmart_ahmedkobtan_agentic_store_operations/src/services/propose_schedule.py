# scripts/propose_schedule.py
import argparse
from pathlib import Path
import json

from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import ARTIFACT_OUT_DIR, ROSTER_FILE, CONSTRAINTS_YAML_PATH
from walmart_ahmedkobtan_agentic_store_operations.src.algorithms.schedule_solver import propose_schedule
import uuid

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=str, default=ARTIFACT_OUT_DIR / "role_targets_next7d.parquet",
                    help="Role targets path (parquet or csv) with columns: timestamp_local, lead_needed, cashier_needed, floor_needed")
    ap.add_argument("--roster", type=str, default=ROSTER_FILE,
                    help="Roster CSV with columns: employee_id,role,base_wage,availability_json")
    ap.add_argument("--constraints", type=str, default=CONSTRAINTS_YAML_PATH,
                    help="Constraints YAML (open/close, min/max shift, wages, objective weights)")
    ap.add_argument("--horizon_days", type=int, default=3, help="Solve the first N days in targets")
    ap.add_argument("--out_dir", type=str, default=ARTIFACT_OUT_DIR, help="Where to write schedule & summary")
    ap.add_argument("--time_limit_s", type=int, default=30, help="CP-SAT max time in seconds")
    return ap.parse_args()

def run_schedule(horizon_days: int = 3, targets: str | Path | None = None,
                 roster: str | Path | None = None, constraints: str | Path | None = None,
                 out_dir: str | Path | None = None, time_limit_s: int = 30) -> dict:
    """Programmatic entrypoint for UI or services."""
    t_path = targets or (ARTIFACT_OUT_DIR / "role_targets_next7d.parquet")
    r_path = roster or ROSTER_FILE
    c_path = constraints or CONSTRAINTS_YAML_PATH
    o_dir = out_dir or ARTIFACT_OUT_DIR
    run_id = str(uuid.uuid4())
    res = propose_schedule(
        role_targets_path=str(t_path),
        roster_csv=str(r_path),
        constraints_yaml=str(c_path),
        horizon_days=horizon_days,
        out_dir=str(o_dir),
        time_limit_s=time_limit_s,
        run_id=run_id
    )
    return res

def main():
    args = parse_args()
    res = run_schedule(
        horizon_days=args.horizon_days,
        targets=args.targets,
        roster=args.roster,
        constraints=args.constraints,
        out_dir=args.out_dir,
        time_limit_s=args.time_limit_s,
    )
    print(json.dumps(res["summary"], indent=2))
    print(f"Artifacts stamped with run_id={res['summary']['run_id']}")

if __name__ == "__main__":
    main()
