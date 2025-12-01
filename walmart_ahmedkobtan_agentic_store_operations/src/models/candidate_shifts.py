from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class CandidateShift:
    sid: int
    employee_id: str
    role: str
    day: pd.Timestamp  # midnight of the day
    start_hour: int    # local hour in [open, close)
    end_hour: int      # local hour in (start, close], end exclusive
    length: int
    hours_mask: List[int]  # indices of hours (0..H-1) covered within that day window
