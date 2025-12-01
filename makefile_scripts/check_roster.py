#!/usr/bin/env python
from pathlib import Path
from walmart_ahmedkobtan_agentic_store_operations.src.utils.roster_utils import (
    load_roster,
    availability_mask_for_day,
    WEEKDAYS,
)

# Determine project root (assumes script run from repo root)
ROOT_DIR = Path.cwd() / 'walmart_ahmedkobtan_agentic_store_operations'
ROSTER_FILE = ROOT_DIR / Path('data/roster/employees.csv')

emps = load_roster(ROSTER_FILE)
print(f"Loaded {len(emps)} employees from {ROSTER_FILE}")

for wd in range(7):
    total_h = sum(
        sum(availability_mask_for_day(e, weekday_idx=wd, open_hour=6, close_hour=21)) for e in emps
    )
    print(
        f"Weekday {WEEKDAYS[wd]}: total available employee-hours in window 06-21 = {total_h}"
    )
