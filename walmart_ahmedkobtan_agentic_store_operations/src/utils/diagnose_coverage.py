
# scripts/diagnose_coverage.py
import pandas as pd
from pathlib import Path

from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import ARTIFACT_OUT_DIR, OPEN_HOUR, CLOSE_HOUR

targets = pd.read_parquet(ARTIFACT_OUT_DIR / "role_targets_next7d.parquet")
schedule = pd.read_parquet(ARTIFACT_OUT_DIR / "schedule_proposal.parquet")

# Align on run_id if present in both
if "run_id" in targets.columns and "run_id" in schedule.columns:
    t_run_ids = set(targets.run_id.unique())
    s_run_ids = set(schedule.run_id.unique())
    common = list(t_run_ids.intersection(s_run_ids))
    if common:
        run_id = common[-1]
        targets = targets[targets.run_id == run_id]
        schedule = schedule[schedule.run_id == run_id]
    else:
        run_id = None
else:
    run_id = None

start = targets["timestamp_local"].min().normalize() + pd.Timedelta(hours=OPEN_HOUR)
end   = start + pd.Timedelta(days=3)
t3 = targets[(targets["timestamp_local"]>=start)&(targets["timestamp_local"]<end)].copy()

# staffed per hour/role
hours = t3["timestamp_local"].sort_values().unique()
staffed = { "lead":[], "cashier":[], "floor":[] }
for ts in hours:
    staffed["lead"].append(((schedule["role"]=="lead") & (schedule["day"]==ts.normalize()) &
                            (schedule["start_hour"]<=ts.hour) & (schedule["end_hour"]>ts.hour)).sum())
    staffed["cashier"].append(((schedule["role"]=="cashier") & (schedule["day"]==ts.normalize()) &
                               (schedule["start_hour"]<=ts.hour) & (schedule["end_hour"]>ts.hour)).sum())
    staffed["floor"].append(((schedule["role"]=="floor") & (schedule["day"]==ts.normalize()) &
                             (schedule["start_hour"]<=ts.hour) & (schedule["end_hour"]>ts.hour)).sum())

df = pd.DataFrame({
    "timestamp_local": hours,
    "lead_req": t3["lead_needed"].values,
    "cashier_req": t3["cashier_needed"].values,
    "floor_req": t3["floor_needed"].values,
    "lead_staff": staffed["lead"],
    "cashier_staff": staffed["cashier"],
    "floor_staff": staffed["floor"],
})
df["cashier_diff"] = df["cashier_staff"] - df["cashier_req"]
diffs = df[df["cashier_diff"]!=0][["timestamp_local","cashier_req","cashier_staff","cashier_diff"]]
print("Run ID:" , run_id)
print(diffs)
print(f"Total mismatched cashier hours: {len(diffs)}")
