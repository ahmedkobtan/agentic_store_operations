from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class Constraints:
    open_hour: int
    close_hour: int
    min_shift_hours: int
    max_shift_hours: int
    max_hours_per_day: int
    min_rest_hours: int
    max_hours_per_week: int | None
    wages: Dict[str, float]
    role_min_headcount: Dict[str, List[Tuple[int, int, int]]]  # role -> list of (start, end, min) in [open,close)
    role_min_hours: Dict[str, int]
    role_max_hours: Dict[str, int]
    over_caps: Dict[str, int]  # role -> max overstaff allowed
    obj_w_shortfall: float
    obj_w_overstaff: float
    obj_w_cost: float
    obj_w_fairness: float
