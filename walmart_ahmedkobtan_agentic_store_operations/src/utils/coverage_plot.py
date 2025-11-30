
# scripts/coverage_plot.py
import argparse, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import yaml

from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import OPEN_HOUR, CLOSE_HOUR, ARTIFACT_OUT_DIR, CONSTRAINTS_YAML_PATH

def load_targets(path):
    if path.__str__().endswith(".parquet") if isinstance(path, Path) else path.endswith(".parquet"): return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["timestamp_local"])

def load_schedule(path):
    if path.__str__().endswith(".parquet") if isinstance(path, Path) else path.endswith(".parquet"): return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["day"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=ARTIFACT_OUT_DIR / "role_targets_next7d.parquet")
    ap.add_argument("--schedule", default=ARTIFACT_OUT_DIR / "schedule_proposal.parquet")
    ap.add_argument("--constraints", default=CONSTRAINTS_YAML_PATH)
    ap.add_argument("--out_png", default=ARTIFACT_OUT_DIR / "coverage_plot.png")
    args = ap.parse_args()


    cons = yaml.safe_load(Path(args.constraints).read_text())
    open_h, close_h = cons["store"]["open_hour"], cons["store"]["close_hour"]
    H = close_h - open_h

    tgt = load_targets(args.targets).sort_values("timestamp_local")
    start = tgt["timestamp_local"].min().normalize() + pd.Timedelta(hours=open_h)
    end   = start + pd.Timedelta(days=3)
    tgt = tgt[(tgt["timestamp_local"] >= start) & (tgt["timestamp_local"] < end)]

    sch = load_schedule(args.schedule)

    days = sorted(list({ts.normalize() for ts in tgt["timestamp_local"]}))
    day_to_idx = {d:i for i,d in enumerate(days)}

    def mk_grid(col):
        g = pd.DataFrame(0, index=range(len(days)), columns=range(H))
        for _, r in tgt.iterrows():
            d = day_to_idx[r["timestamp_local"].normalize()]
            h = int(r["timestamp_local"].hour) - open_h
            g.loc[d, h] = int(r[col])
        return g

    req = {
      "lead": mk_grid("lead_needed"),
      "cashier": mk_grid("cashier_needed"),
      "floor": mk_grid("floor_needed")
    }





    min_windows = {}
    for role_entry in cons.get("roles", []):
        name = role_entry["name"]
        min_windows[name] = []
        for win, k in (role_entry.get("min_headcount", {}) or {}).items():
            hs, he = [int(s.split(":")[0]) for s in win.split("-")]
            min_windows[name].append((hs, he, int(k)))

    def apply_min_headcount(req_grid, role):
        for (hs, he, k) in min_windows.get(role, []):
            req_grid.loc[:, hs-open_h:he-open_h-1] = np.maximum(req_grid.loc[:, hs-open_h:he-open_h-1], k)
        return req_grid

    for r in ["lead","cashier","floor"]:
        req[r] = apply_min_headcount(req[r], r)





    staffed = { "lead": req["lead"]*0, "cashier": req["cashier"]*0, "floor": req["floor"]*0 }
    for _, r in sch.iterrows():
        d = day_to_idx[pd.Timestamp(r["day"]).normalize()]
        role = r["role"]
        for h in range(int(r["start_hour"]), int(r["end_hour"])):
            slot = h - open_h
            if 0 <= slot < H:
                staffed[role].loc[d, slot] += 1

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    roles = ["lead","cashier","floor"]
    for i, role in enumerate(roles):
        ax = axes[i]
        colors = ["tab:blue","tab:green","tab:red"]
        for d in range(len(days)):
            x = list(range(open_h, close_h))
            ax.step(x, req[role].loc[d].values, where="mid", color=colors[d%len(colors)], alpha=0.6)
            ax.step(x, staffed[role].loc[d].values, where="mid", color=colors[d%len(colors)], alpha=0.9, linestyle="--")
        # for d in range(len(days)):
        #     x = list(range(open_h, close_h))
        #     ax.step(x, req[role].loc[d].values, where="mid", color="tab:blue", alpha=0.6, label=f"req day{d+1}" if i==0 and d==0 else None)
        #     ax.step(x, staffed[role].loc[d].values, where="mid", color="tab:orange", alpha=0.9, label=f"staff day{d+1}" if i==0 and d==0 else None)
        ax.set_title(f"{role.capitalize()} — Required vs Staffed (3 days)")
        ax.set_ylabel("headcount")
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[-1].set_xlabel("hour of day")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=150)
    print(f"Saved {args.out_png}")

if __name__ == "__main__":
    main()
