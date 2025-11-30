import pandas as pd
from pathlib import Path
from walmart_ahmedkobtan_agentic_store_operations.src.utils.roster_utils import load_roster, availability_mask_for_day


def test_load_roster(data_path):
    emps = load_roster(data_path / Path("roster/employees.csv"))
    assert isinstance(emps, list)
    assert len(emps) > 0
    assert all(hasattr(emp, "employee_id") for emp in emps)

def test_availability_mask_for_day(data_path):
    emps = load_roster(data_path / Path("roster/employees.csv"))
    emp = emps[1]  # first employee
    mask = availability_mask_for_day(emp, weekday_idx=0, open_hour=6, close_hour=21)  # Monday
    assert isinstance(mask, list)
    assert all(isinstance(x, bool) for x in mask)
