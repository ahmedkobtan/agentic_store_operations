# core/roster.py
from __future__ import annotations
import ast
import pandas as pd
from typing import List
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import WEEKDAYS
from walmart_ahmedkobtan_agentic_store_operations.src.models.employee import Employee

"""
from core.roster import load_roster, availability_mask_for_day
emps = load_roster("data/roster/employees.csv")
mask = availability_mask_for_day(emps[0], weekday_idx=0, open_hour=6, close_hour=21)  # Monday
print(emps[0], sum(mask), "available hours in window")
"""

def load_roster(csv_path: str) -> List[Employee]:
    df = pd.read_csv(csv_path)
    employees: List[Employee] = []
    for _, r in df.iterrows():
        # availability_json is a Python-literal dict stored as a string. Parse safely.
        avail_raw = ast.literal_eval(str(r["availability_json"]))
        # Normalize: ensure all weekdays present; if missing, mark unavailable
        availability = {}
        for wd in WEEKDAYS:
            if wd in avail_raw:
                s, e = avail_raw[wd]
                availability[wd] = (int(s), int(e))
            else:
                availability[wd] = (None, None)  # unavailable
        employees.append(Employee(
            employee_id=str(r["employee_id"]),
            role=str(r["role"]),
            base_wage=float(r["base_wage"]),
            availability=availability
        ))
    return employees

def availability_mask_for_day(emp: Employee, weekday_idx: int, open_hour: int, close_hour: int):
    """Return a list of booleans length (close_hour - open_hour) marking which hours employee can work."""
    wd = WEEKDAYS[weekday_idx]
    s, e = emp.availability.get(wd, (None, None))
    H = list(range(open_hour, close_hour))
    if s is None or e is None:
        return [False]*len(H)
    return [ (h >= s and h < e) for h in H ]
