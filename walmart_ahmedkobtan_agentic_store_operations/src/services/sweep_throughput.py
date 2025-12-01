
# scripts/sweep_throughput.py
import argparse, yaml, pandas as pd
from pathlib import Path

from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import ARTIFACT_OUT_DIR, ROSTER_FILE, CONSTRAINTS_YAML_PATH, FORECASTING_GUARDRAILS_YAML_PATH
from walmart_ahmedkobtan_agentic_store_operations.src.algorithms.schedule_solver import propose_schedule
from walmart_ahmedkobtan_agentic_store_operations.src.algorithms.make_role_targets import build_targets_for_tx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", nargs="+", type=float, default=[20,30,40,60])
    ap.add_argument("--forecast", type=str, default=ARTIFACT_OUT_DIR / "df_forecast_next7d_how.parquet")
    ap.add_argument("--guardrails", type=str, default=FORECASTING_GUARDRAILS_YAML_PATH)
    ap.add_argument("--roster", type=str, default=ROSTER_FILE)
    ap.add_argument("--constraints", type=str, default=CONSTRAINTS_YAML_PATH)
    ap.add_argument("--out_dir", type=str, default=ARTIFACT_OUT_DIR)
    ap.add_argument("--horizon_days", type=int, default=3)
    args = ap.parse_args()

    fc_path = Path(args.forecast)
    if fc_path.suffix == ".parquet": fc = pd.read_parquet(fc_path).sort_values("timestamp_local")
    else: fc = pd.read_csv(fc_path.with_suffix(".csv"), parse_dates=["timestamp_local"]).sort_values("timestamp_local")

    guardrails = yaml.safe_load(Path(args.guardrails).read_text())

    rows = []
    for tx in args.values:
        targets = build_targets_for_tx(fc, guardrails=guardrails, tx=tx)
        tpath = Path(args.out_dir) / f"role_targets_next7d_tx{int(tx)}.parquet"
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        targets.to_parquet(tpath, index=False)

        res = propose_schedule(str(tpath), args.roster, args.constraints, horizon_days=args.horizon_days, out_dir=args.out_dir, time_limit_s=30)
        s = res["summary"]
        rows.append({
            "tx_per_cashier": tx,
            "status": s["status"],
            "total_cost": s["total_cost"],
            "short_lead": s["shortfall_by_role_hours"]["lead"],
            "short_cashier": s["shortfall_by_role_hours"]["cashier"],
            "short_floor": s["shortfall_by_role_hours"]["floor"],
            "over_lead": s["overstaff_by_role_hours"]["lead"],
            "over_cashier": s["overstaff_by_role_hours"]["cashier"],
            "over_floor": s["overstaff_by_role_hours"]["floor"],
        })

    dfres = pd.DataFrame(rows)
    out = Path(args.out_dir) / "throughput_sweep_results.csv"
    dfres.to_csv(out, index=False)
    print(dfres.to_string(index=False))
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
