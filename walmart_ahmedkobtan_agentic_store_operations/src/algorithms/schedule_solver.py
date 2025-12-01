# core/schedule_solver.py
from __future__ import annotations
import ast
import math
import json
import uuid
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# OR-Tools
from ortools.sat.python import cp_model

# Data models
from walmart_ahmedkobtan_agentic_store_operations.src.models.constraints import Constraints
from walmart_ahmedkobtan_agentic_store_operations.src.models.employee import Employee
from walmart_ahmedkobtan_agentic_store_operations.src.models.candidate_shifts import CandidateShift
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import WEEKDAYS, ARTIFACT_OUT_DIR
from walmart_ahmedkobtan_agentic_store_operations.src.utils.yaml_utils import read_yaml

# ---------- Data Structures ----------

def _parse_constraints(yaml_path: str) -> Constraints:
    y = read_yaml(yaml_path)
    # Convert min_headcount windows "HH:MM-HH:MM": k to tuples [start,end,k]
    role_min = {}
    for role_entry in y.get("roles", []):
        name = role_entry["name"]
        role_min[name] = []
        mh = role_entry.get("min_headcount", {}) or {}
        for window, k in mh.items():
            start_s, end_s = window.split("-")
            hs = int(start_s.split(":")[0])
            he = int(end_s.split(":")[0])
            role_min[name].append((hs, he, int(k)))

    role_shifts = y["employees"].get("role_shift_hours", {})

    caps = y.get("overstaff_caps", {"lead":0,"cashier":1,"floor":0})

    c = Constraints(
        open_hour = int(y["store"]["open_hour"]),
        close_hour = int(y["store"]["close_hour"]),
        min_shift_hours = int(y["employees"]["defaults"]["min_shift_hours"]),
        max_shift_hours = int(y["employees"]["defaults"]["max_shift_hours"]),
        max_hours_per_day = int(y["employees"]["defaults"]["max_hours_per_day"]),
        min_rest_hours = int(y["employees"]["defaults"]["min_rest_hours"]),
        max_hours_per_week = y["employees"]["defaults"].get("max_hours_per_week", None),
        wages = {k: float(v) for k, v in y["wages"].items()},
        role_min_headcount = role_min,
        role_min_hours = {r:int(v.get("min", y["employees"]["defaults"]["min_shift_hours"])) for r,v in role_shifts.items()},
        role_max_hours = {r:int(v.get("max", y["employees"]["defaults"]["max_shift_hours"])) for r,v in role_shifts.items()},
        over_caps = {k:int(v) for k,v in caps.items()},
        obj_w_shortfall = float(y["objective"]["weights"]["shortfall"]),
        obj_w_overstaff = float(y["objective"]["weights"]["overstaff"]),
        obj_w_cost = float(y["objective"]["weights"]["cost"]),
        obj_w_fairness = float(y["objective"]["weights"]["fairness"]),
    )
    return c

def _load_roster(roster_csv: str) -> List[Employee]:
    df = pd.read_csv(roster_csv)
    emps: List[Employee] = []
    for _, r in df.iterrows():
        # availability_json is Python-literal dict like {'mon':[6,21], 'sat':[7,21], ...}
        avail_raw = ast.literal_eval(str(r["availability_json"]))
        availability = {}
        for wd in WEEKDAYS:
            if wd in avail_raw:
                s, e = avail_raw[wd]
                availability[wd] = (int(s), int(e))
            else:
                availability[wd] = None
        emps.append(Employee(
            employee_id=str(r["employee_id"]),
            role=str(r["role"]),
            base_wage=float(r["base_wage"]),
            availability=availability
        ))
    return emps

# ---------- Targets & Horizon ----------

def _load_targets(role_targets_path: str) -> pd.DataFrame:
    # Accept parquet or csv; auto-detect by suffix
    if (role_targets_path.__str__().endswith(".parquet") if isinstance(role_targets_path,Path) else role_targets_path.endswith(".parquet")):
        df = pd.read_parquet(role_targets_path)
    else:
        df = pd.read_csv(role_targets_path, parse_dates=["timestamp_local"])
    df = df.sort_values("timestamp_local")
    return df

def _apply_min_headcount(targets: pd.DataFrame, cons: Constraints) -> pd.DataFrame:
    """Ensure that role min_headcount windows from constraints are enforced on the targets dataframe."""
    t = targets.copy()
    hour = t["timestamp_local"].dt.hour

    for role, windows in cons.role_min_headcount.items():
        if role not in ["lead", "cashier", "floor"]:
            continue
        for (hs, he, k) in windows:
            mask = (hour >= hs) & (hour < he)
            col = f"{role}_needed"
            if col not in t.columns:
                t[col] = 0
            # enforce max(target, k)
            t.loc[mask, col] = np.maximum(t.loc[mask, col].values, k).astype(int)
    return t

def _slice_first_ndays(df: pd.DataFrame, n_days: int, open_hour: int, close_hour: int) -> pd.DataFrame:
    start = df["timestamp_local"].min().normalize() + pd.Timedelta(hours=open_hour)
    end   = start + pd.Timedelta(days=n_days)  # exclusive
    return df[(df["timestamp_local"] >= start) & (df["timestamp_local"] < end)].copy()




# ---------- Candidate shifts (level-block + minimal bridging) ----------
def _level_blocks_for_role_day(demand_arr_day: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Return contiguous segments where required level is constant as (start_slot, end_slot_exclusive, level).
    demand_arr_day shape = [H], integers (headcount per hour).
    Example: [0,0,1,1,2,2,1,1,0] -> [(2,4,1), (4,6,2), (6,8,1)]
    """
    blocks: List[Tuple[int, int, int]] = []
    H = len(demand_arr_day)
    h = 0
    while h < H:
        lvl = int(demand_arr_day[h])
        if lvl <= 0:
            h += 1
            continue
        s = h
        h += 1
        while h < H and int(demand_arr_day[h]) == lvl:
            h += 1
        blocks.append((s, h, lvl))  # [s, h) at level lvl
    return blocks

def _generate_level_block_candidates_for_day(
    emp: Employee,
    day_dt: pd.Timestamp,
    open_hour: int,
    close_hour: int,
    min_len: int,
    max_len: int,
    level_blocks_for_day: List[Tuple[int, int, int]],
    demand_arr_day: np.ndarray
) -> List[CandidateShift]:
    """
    Generate candidates *inside level blocks*, with minimal bridging into adjacent
    positive-demand slots when block_len < min_len. Bridging may be split across
    forward and backward directions, but never crosses zero-demand.
    """
    # Availability ∩ store window
    wd = WEEKDAYS[day_dt.weekday()]
    avail = emp.availability.get(wd)
    if not avail:
        return []
    a_s, a_e = avail
    s_hour = max(a_s, open_hour)
    e_hour = min(a_e, close_hour)
    if s_hour >= e_hour:
        return []

    s_slot = s_hour - open_hour      # inclusive
    e_slot = e_hour - open_hour      # exclusive

    cand: List[CandidateShift] = []

    for (bs, be, lvl) in level_blocks_for_day:
        # Intersect block with employee feasible slots
        bs_eff = max(bs, s_slot)
        be_eff = min(be, e_slot)
        if bs_eff >= be_eff:
            continue

        block_len = be_eff - bs_eff


        # Case A: block long enough -> inside-block candidates
        if block_len >= min_len:
            # 1) Pure inside-block shifts
            for st_slot in range(bs_eff, be_eff - min_len + 1):
                max_L_inside = min(max_len, be_eff - st_slot)
                for L in range(min_len, max_L_inside + 1):
                    en_slot = st_slot + L
                    st_hour = open_hour + st_slot
                    en_hour = open_hour + en_slot
                    mask = list(range(st_slot, en_slot))
                    cand.append(CandidateShift(
                        sid=-1, employee_id=emp.employee_id, role=emp.role,
                        day=day_dt.normalize(), start_hour=st_hour, end_hour=en_hour,
                        length=L, hours_mask=mask
                    ))

            # 2) Adjacent positive-demand extension (at most min_len - 1 hours)
            #    This creates alternatives like 11–13 or 13–15 when the block is 12–14,
            #    provided the adjacent hour(s) have positive demand and we stay within availability.
            max_adj = min_len - 1
            if max_adj > 0:
                # Backward (pre-block) consecutive positive-demand slots available
                back_avail = 0
                cur = bs_eff - 1
                while cur >= s_slot and int(demand_arr_day[cur]) > 0:
                    back_avail += 1
                    cur -= 1
                back_avail = min(back_avail, max_adj)

                # Forward (post-block) consecutive positive-demand slots available
                fwd_avail = 0
                cur = be_eff
                while cur < e_slot and int(demand_arr_day[cur]) > 0:
                    fwd_avail += 1
                    cur += 1
                fwd_avail = min(fwd_avail, max_adj)

                # Generate backward-extended starts (e.g., 11–13 for min_len=2)
                for ext in range(1, back_avail + 1):
                    st_slot = bs_eff - ext
                    # End must be at least min_len inside [st_slot, be_eff] and within max_len
                    min_en = st_slot + min_len
                    max_en = min(st_slot + max_len, be_eff)  # don't cross block end for pure back extension
                    for en_slot in range(min_en, max_en + 1):
                        st_hour = open_hour + st_slot
                        en_hour = open_hour + en_slot
                        mask = list(range(st_slot, en_slot))
                        cand.append(CandidateShift(
                            sid=-1, employee_id=emp.employee_id, role=emp.role,
                            day=day_dt.normalize(), start_hour=st_hour, end_hour=en_hour,
                            length=en_slot - st_slot, hours_mask=mask
                        ))

                # Generate forward-extended ends (e.g., 12–15 for min_len=2)
                for ext in range(1, fwd_avail + 1):
                    # Start stays within block, end extends into adjacent positive-demand
                    for st_slot in range(bs_eff, be_eff - min_len + 1):
                        min_en = st_slot + min_len
                        max_en = min(st_slot + max_len, be_eff + ext)
                        for en_slot in range(max(min_en, be_eff + 1), max_en + 1):
                            st_hour = open_hour + st_slot
                            en_hour = open_hour + en_slot
                            mask = list(range(st_slot, en_slot))
                            cand.append(CandidateShift(
                                sid=-1, employee_id=emp.employee_id, role=emp.role,
                                day=day_dt.normalize(), start_hour=st_hour, end_hour=en_hour,
                                length=en_slot - st_slot, hours_mask=mask
                            ))
            continue

        # Case B: block shorter than min_len -> split bridging inside positive demand
        needed = min_len - block_len

        # Try forward first
        fwd = 0; cur = be_eff
        while fwd < needed and cur < e_slot and int(demand_arr_day[cur]) > 0:
            fwd += 1; cur += 1

        # Try backward for the remainder
        bwd = 0; cur = bs_eff - 1
        while (fwd + bwd) < needed and cur >= s_slot and int(demand_arr_day[cur]) > 0:
            bwd += 1; cur -= 1

        if (fwd + bwd) < needed:
            # Not enough positive-demand around to reach min_len; skip
            continue

        # Build one or both bridged candidates; both stay within positive-demand slots
        # Candidate A: start at bs_eff, extend forward 'fwd' then backward 'bwd' (if any)
        st_a = bs_eff - bwd
        en_a = be_eff + fwd
        if (st_a >= s_slot) and (en_a <= e_slot):
            L = en_a - st_a
            if min_len <= L <= max_len:
                cand.append(CandidateShift(
                    sid=-1, employee_id=emp.employee_id, role=emp.role,
                    day=day_dt.normalize(), start_hour=open_hour + st_a, end_hour=open_hour + en_a,
                    length=L, hours_mask=list(range(st_a, en_a))
                ))

        # Candidate B (optional): if availability favors backward, try pure backward-first layout
        # start earlier 'bwd' then inside block only
        if bwd > 0:
            st_b = bs_eff - bwd
            en_b = bs_eff + min_len   # enough inside + bwd to reach min_len
            if (st_b >= s_slot) and (en_b <= e_slot) and (en_b <= be_eff):
                L = en_b - st_b
                if min_len <= L <= max_len:
                    cand.append(CandidateShift(
                        sid=-1, employee_id=emp.employee_id, role=emp.role,
                        day=day_dt.normalize(), start_hour=open_hour + st_b, end_hour=open_hour + en_b,
                        length=L, hours_mask=list(range(st_b, en_b))
                    ))

    return cand

def _generate_all_candidates(
    emps: List[Employee],
    days: List[pd.Timestamp],
    cons: Constraints,
    demand_lead: np.ndarray,     # [D,H]
    demand_cashier: np.ndarray,  # [D,H]
    demand_floor: np.ndarray,    # [D,H]
    day_to_idx: Dict[pd.Timestamp, int],
    H_per_day: int
) -> List[CandidateShift]:
    """
    Level-block-contained candidate generation with minimal positive-demand bridging:
      - LEAD: one constant-level block [0, H) (baseline lead=1).
      - CASHIER/FLOOR: constant-level blocks per day from targets; allow small bridges to satisfy min_len.
      - Role-specific min/max shift hours honored; availability & open/close respected.
    """
    all_cand: List[CandidateShift] = []
    sid = 0

    blocks_by_role_day = {
        "lead":   {d: [(0, H_per_day, 1)] for d in range(len(days))},
        "cashier":{d: _level_blocks_for_role_day(demand_cashier[d, :]) for d in range(len(days))},
        "floor":  {d: _level_blocks_for_role_day(demand_floor[d, :])   for d in range(len(days))}
    }

    for day in days:
        didx = day_to_idx[day]
        for emp in emps:
            min_len = cons.role_min_hours.get(emp.role, cons.min_shift_hours)
            max_len = cons.role_max_hours.get(emp.role, cons.max_shift_hours)

            if emp.role == "lead":
                level_blocks = blocks_by_role_day["lead"][didx]
                demand_arr_day = demand_lead[didx, :]
            elif emp.role == "cashier":
                level_blocks = blocks_by_role_day["cashier"][didx]
                demand_arr_day = demand_cashier[didx, :]
                if not level_blocks:
                    continue
            else:  # floor
                level_blocks = blocks_by_role_day["floor"][didx]
                demand_arr_day = demand_floor[didx, :]
                if not level_blocks:
                    continue

            cands = _generate_level_block_candidates_for_day(
                emp, day,
                cons.open_hour, cons.close_hour,
                min_len, max_len,
                level_blocks,
                demand_arr_day
            )
            for c in cands:
                c.sid = sid
                sid += 1
                all_cand.append(c)

    return all_cand

# ---------- Build/solve CP-SAT ----------

def propose_schedule(
    role_targets_path: str,
    roster_csv: str,
    constraints_yaml: str,
    horizon_days: int = 3,
    out_dir: str = ARTIFACT_OUT_DIR,
    time_limit_s: int = 30,
    run_id: str | None = None
) -> Dict:
    """Solve staffing schedule with CP-SAT allowing explicit shortfall & overstaff tracking.

    Changes vs previous version:
      - Introduces run_id for artifact/version traceability.
      - Replaces hard staffed >= demand constraints (except lead) with balance equation:
            staffed + shortfall - overstaff == demand
        using non-negative IntVars shortfall, overstaff per role/hour.
      - Objective penalizes shortfall heavily (W_SHORT) and overstaff by role (W_OVER).
    """
    run_id = run_id or str(uuid.uuid4())
    cons = _parse_constraints(constraints_yaml)
    emps = _load_roster(roster_csv)
    tgt = _load_targets(role_targets_path)

    # Enforce role-level minimums from constraints (e.g., lead=1 at all open hours)
    tgt = _apply_min_headcount(tgt, cons)

    # Slice first N days starting from first forecast day at open_hour
    tgt3 = _slice_first_ndays(tgt, horizon_days, cons.open_hour, cons.close_hour).reset_index(drop=True)

    # Build day list and a map hour_index per day
    hours = tgt3["timestamp_local"]
    days = sorted(list({ts.normalize() for ts in hours}))
    day_to_idx = {d: i for i, d in enumerate(days)}

    # Build per-hour targets by role, indexed by (day_idx, hour_slot)
    H_per_day = cons.close_hour - cons.open_hour
    T = len(days) * H_per_day

    def mk_target(role_col: str):
        arr = np.zeros((len(days), H_per_day), dtype=int)
        for _, r in tgt3.iterrows():
            d = day_to_idx[r["timestamp_local"].normalize()]
            h = int(r["timestamp_local"].hour) - cons.open_hour
            if 0 <= h < H_per_day:
                arr[d, h] = int(r[role_col])
        return arr

    demand_lead    = mk_target("lead_needed")
    demand_cashier = mk_target("cashier_needed")
    demand_floor   = mk_target("floor_needed")

    # Candidates
    cand = _generate_all_candidates(emps, days, cons, demand_lead, demand_cashier, demand_floor, day_to_idx, H_per_day)

    # Break modeling (simple post-generation augmentation):
    # For candidates with length >= 7 (hard-coded threshold aligned with compliance break_threshold_hours),
    # create an alternate pair of split shifts with a 1-hour gap to represent a break.
    BREAK_THRESHOLD = 7  # hours
    augmented: List[CandidateShift] = []
    # Track explicit break metadata for split shifts: sid -> (break_start_hour, break_end_hour)
    break_meta: Dict[int, Tuple[int, int]] = {}
    next_sid = max([c.sid for c in cand]) + 1 if cand else 0
    for c in cand:
        if c.length >= BREAK_THRESHOLD and c.role != "lead":  # do not split lead coverage shifts
            mid = c.start_hour + c.length // 2
            # First segment: start -> mid-1
            seg1_len = mid - c.start_hour
            # Second segment: mid+1 -> end (reserve one hour gap for break)
            seg2_start = mid + 1
            seg2_len = c.end_hour - seg2_start
            # Only create if both segments respect min_shift_hours
            if seg1_len >= cons.min_shift_hours and seg2_len >= cons.min_shift_hours:
                # Build hours masks
                seg1_mask = [h for h in c.hours_mask if (c.start_hour + h) < mid]
                seg2_mask = [h for h in c.hours_mask if (c.start_hour + h) >= seg2_start]
                # Remap hours mask relative to original open hour
                # CandidateShift expects absolute hour indices? Using existing pattern: hours_mask are slots relative to day open.
                # We'll recompute masks based on day and open_hour
                day_open = cons.open_hour
                seg1_slots = [h for h in range(c.start_hour - day_open, mid - day_open)]
                seg2_slots = [h for h in range(seg2_start - day_open, c.end_hour - day_open)]
                cs1 = CandidateShift(
                    sid=next_sid,
                    employee_id=c.employee_id,
                    role=c.role,
                    day=c.day,
                    start_hour=c.start_hour,
                    end_hour=mid,
                    length=seg1_len,
                    hours_mask=seg1_slots
                )
                next_sid += 1
                cs2 = CandidateShift(
                    sid=next_sid,
                    employee_id=c.employee_id,
                    role=c.role,
                    day=c.day,
                    start_hour=seg2_start,
                    end_hour=c.end_hour,
                    length=seg2_len,
                    hours_mask=seg2_slots
                )
                next_sid += 1
                augmented.extend([cs1, cs2])
                # Record a 1-hour break between segments
                break_meta[cs1.sid] = (mid, mid+1)
                break_meta[cs2.sid] = (mid, mid+1)
            else:
                augmented.append(c)
        else:
            augmented.append(c)
    cand = augmented

    # Index helpers
    # Per (day, hour) collect candidate shift ids by role that cover the hour
    cover_by_role_day_hour = {
        "lead":   [[[] for _ in range(H_per_day)] for _ in range(len(days))],
        "cashier":[[[] for _ in range(H_per_day)] for _ in range(len(days))],
        "floor":  [[[] for _ in range(H_per_day)] for _ in range(len(days))]
    }
    # Track shifts per employee per day to enforce <= 1 shift/day (and no overlap)
    shifts_per_emp_day: Dict[Tuple[str, pd.Timestamp], List[int]] = {}

    for c in cand:
        d = day_to_idx[c.day]
        for h in c.hours_mask:
            cover_by_role_day_hour[c.role][d][h].append(c.sid)
        key = (c.employee_id, c.day)
        shifts_per_emp_day.setdefault(key, []).append(c.sid)

    # CP-SAT model
    m = cp_model.CpModel()

    # Decision: select shift
    x = {c.sid: m.NewBoolVar(f"x_s{c.sid}") for c in cand}

    # Hard: at most one shift per employee per day (implicitly ensures min rest and no overlap)
    for (emp_id, day_dt), sids in shifts_per_emp_day.items():
        m.Add(sum(x[s] for s in sids) <= 1)

    # Coverage balance per hour/role:
    # staffed - demand = over - short; with over>=0, short>=0
    # New formulation: staffed + short - over == demand (except lead kept strictly ≥ demand to guarantee coverage leadership)
    staffed = {
        "lead":   [[None for _ in range(H_per_day)] for _ in range(len(days))],
        "cashier":[[None for _ in range(H_per_day)] for _ in range(len(days))],
        "floor":  [[None for _ in range(H_per_day)] for _ in range(len(days))]
    }
    short = {
        "lead":   [[m.NewIntVar(0, 0, f"short_lead_{d}_{h}") for h in range(H_per_day)] for d in range(len(days))],  # force 0 for lead
        "cashier":[[m.NewIntVar(0, 1000, f"short_cash_{d}_{h}") for h in range(H_per_day)] for d in range(len(days))],
        "floor":  [[m.NewIntVar(0, 1000, f"short_floor_{d}_{h}") for h in range(H_per_day)] for d in range(len(days))]
    }
    over = {
        "lead":   [[m.NewIntVar(0, 1000, f"over_lead_{d}_{h}") for h in range(H_per_day)] for d in range(len(days))],
        "cashier":[[m.NewIntVar(0, 1000, f"over_cash_{d}_{h}") for h in range(H_per_day)] for d in range(len(days))],
        "floor":  [[m.NewIntVar(0, 1000, f"over_floor_{d}_{h}") for h in range(H_per_day)] for d in range(len(days))]
    }

    def staffed_expr(role: str, d: int, h: int):
        return sum(x[sid] for sid in cover_by_role_day_hour[role][d][h])

    for d in range(len(days)):
        for h in range(H_per_day):
            s_lead = staffed_expr("lead", d, h)
            staffed["lead"][d][h] = s_lead
            # Lead: keep hard lower bound, derive over only
            m.Add(s_lead >= demand_lead[d, h])
            m.Add(over["lead"][d][h] == s_lead - demand_lead[d, h])

            # Cashier
            s_cash = staffed_expr("cashier", d, h)
            staffed["cashier"][d][h] = s_cash
            # Balance equation
            m.Add(s_cash + short["cashier"][d][h] - over["cashier"][d][h] == demand_cashier[d, h])

            # Floor
            s_floor = staffed_expr("floor", d, h)
            staffed["floor"][d][h] = s_floor
            m.Add(s_floor + short["floor"][d][h] - over["floor"][d][h] == demand_floor[d, h])



    over_excess = {
        "lead":   [[m.NewIntVar(0, 1000, f"overx_lead_{d}_{h}")   for h in range(H_per_day)] for d in range(len(days))],
        "cashier":[[m.NewIntVar(0, 1000, f"overx_cash_{d}_{h}")   for h in range(H_per_day)] for d in range(len(days))],
        "floor":  [[m.NewIntVar(0, 1000, f"overx_floor_{d}_{h}")  for h in range(H_per_day)] for d in range(len(days))]
    }
    for role in ["lead","cashier","floor"]:
        cap = cons.over_caps.get(role, 0)
        for d in range(len(days)):
            for h in range(H_per_day):
                # over_excess >= over - cap; over_excess >= 0
                m.Add(over_excess[role][d][h] >= over[role][d][h] - cap)
                m.Add(over_excess[role][d][h] >= 0)



    # Cost and fairness
    # Cost per shift = wage(role_of_employee) * hours
    role_wage = cons.wages
    shift_cost = {}
    for c in cand:
        # If employee's base_wage differs from role wage, choose one; we’ll use role wage table (simple)
        wage = role_wage.get(c.role, 0.0)
        shift_cost[c.sid] = int(round(100 * wage * c.length))  # scale to cents

    total_cost_scaled = m.NewIntVar(0, int(1e9), "total_cost_scaled")
    m.Add(total_cost_scaled == sum(shift_cost[sid] * x[sid] for sid in x))

    # Fairness: small penalty on deviation from average daily hours per employee
    # For 3 days, we approximate with "selected shift length" (0 or L) per emp.
    emp_hours = {}
    total_hours_all = m.NewIntVar(0, horizon_days * len(emps) * cons.max_hours_per_day, "total_hours_all")
    m.Add(total_hours_all == sum(c.length * x[c.sid] for c in cand))
    fairness_terms = []
    for emp in emps:
        var = m.NewIntVar(0, horizon_days * cons.max_hours_per_day, f"hours_{emp.employee_id}")
        emp_hours[emp.employee_id] = var
        sids_emp = [c.sid for c in cand if c.employee_id == emp.employee_id]
        if sids_emp:
            m.Add(var == sum([cand[sid].length * x[sid] for sid in sids_emp]))
        else:
            m.Add(var == 0)
        # Deviation from target average ~ total_hours_all / N; use hinge on upper side (soft)
        # We linearize: penalty = max(0, var - avg) approximated by max(0, var - cap)
        # For simplicity set a soft cap = horizon_days * 6 (avg) -> tune as needed
        cap = horizon_days * 6
        excess = m.NewIntVar(0, horizon_days * cons.max_hours_per_day, f"excess_{emp.employee_id}")
        m.Add(excess >= var - cap)
        m.Add(excess >= 0)
        fairness_terms.append(excess)

    # Objective weights (convert to comparable scale)
    # W_OVER  = int(100 * cons.obj_w_overstaff)
    W_COST   = int(cons.obj_w_cost)        # cost already scaled by 100
    W_FAIR   = int(10 * cons.obj_w_fairness)
    W_SHORT  = int(1000 * cons.obj_w_shortfall)  # heavy penalty for unmet demand

    # Build level-1 mask for cashier
    level1_mask = [[int(demand_cashier[d, h] == 1) for h in range(H_per_day)] for d in range(len(days))]

    W_OVER      = int(100 * cons.obj_w_overstaff)    # existing (e.g., 3.0 -> 300)
    W_OVER_L1   = W_OVER * 4                         # NEW: heavier penalty on level-1 tails

    obj_terms = []
    for d in range(len(days)):
        for h in range(H_per_day):
            # Shortfall penalties (lead shortfall fixed at 0)
            obj_terms.append(W_SHORT * short["cashier"][d][h])
            obj_terms.append(W_SHORT * short["floor"][d][h])
            # Overstaff penalties
            obj_terms.append(W_OVER * over["lead"][d][h])
            obj_terms.append(W_OVER * over["floor"][d][h])
            if level1_mask[d][h]:
                obj_terms.append(W_OVER_L1 * over["cashier"][d][h])
            else:
                obj_terms.append(W_OVER * over["cashier"][d][h])


    obj_terms.append(W_COST * total_cost_scaled)
    obj_terms += [W_FAIR * t for t in fairness_terms]
    m.Minimize(sum(obj_terms))



    def log_cover_counts():
        # Debug helper originally assumed 3-day horizon; guard for shorter runs.
        if len(days) < 3:
            return
        def log_role_hour(role, day_idx, hour_slot):
            cids = cover_by_role_day_hour[role][day_idx][hour_slot]
            # Quiet debug; comment out print to avoid noisy tests
            # print(f"{role} D{day_idx} H{hour_slot} covers: {len(cids)}")
            return len(cids)
        # Sample a few points to ensure candidates exist
        _ = log_role_hour("cashier", 0, 7)
        _ = log_role_hour("cashier", 0, 8)
        _ = log_role_hour("cashier", 2, 6)

    log_cover_counts()


    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = False

    status = solver.Solve(m)

    # Build outputs
    schedule_rows = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for c in cand:
            if solver.BooleanValue(x[c.sid]):
                row = {
                    "employee_id": c.employee_id,
                    "role": c.role,
                    "day": c.day,
                    "start_hour": c.start_hour,
                    "end_hour": c.end_hour,
                    "length": c.length
                }
                # If this shift came from a split, include explicit break window
                if c.sid in break_meta:
                    b_start, b_end = break_meta[c.sid]
                    row["has_break"] = True
                    row["break_start_hour"] = b_start
                    row["break_end_hour"] = b_end
                else:
                    row["has_break"] = False
                    row["break_start_hour"] = None
                    row["break_end_hour"] = None
                schedule_rows.append(row)

    # Coverage summary
    def np_sum_int(mat):
        return int(np.sum(mat))

    staffed_counts = {"lead":0,"cashier":0,"floor":0}
    demand_counts  = {"lead":np_sum_int(demand_lead),"cashier":np_sum_int(demand_cashier),"floor":np_sum_int(demand_floor)}
    short_sum = {"lead":0,"cashier":0,"floor":0}
    over_sum  = {"lead":0,"cashier":0,"floor":0}

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for role in ["lead","cashier","floor"]:
            s = 0; o = 0
            for d in range(len(days)):
                for h in range(H_per_day):
                    s += int(solver.Value(short[role][d][h]))
                    o += int(solver.Value(over[role][d][h]))
            short_sum[role] = s
            over_sum[role]  = o

    summary = {
        "run_id": run_id,
        "status": solver.StatusName(status),
        "objective_value": float(solver.ObjectiveValue()) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "total_cost": float(solver.Value(total_cost_scaled))/100.0 if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "shortfall_by_role_hours": short_sum,
        "overstaff_by_role_hours": over_sum,
        "demand_hours_by_role": demand_counts,
        "horizon_days": horizon_days,
        "open_hour": cons.open_hour,
        "close_hour": cons.close_hour
    }

    # Write artifacts
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sch_path = out_dir / "schedule_proposal.parquet"
    sch_path_csv = out_dir / "schedule_proposal.csv"
    sum_path = out_dir / "schedule_summary.json"


    if schedule_rows:
        df_sched = pd.DataFrame(schedule_rows)
        df_sched["run_id"] = run_id
        df_sched.to_parquet(sch_path, index=False)
        df_sched.to_csv(sch_path_csv, index=False)
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return {
        "schedule_path": str(sch_path),
        "schedule_path_csv": str(sch_path_csv),
        "summary_path": str(sum_path),
        "summary": summary,
        "selected_rows": len(schedule_rows)
    }
