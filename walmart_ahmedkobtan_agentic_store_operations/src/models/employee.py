from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class Employee:
    employee_id: str
    role: str
    base_wage: float
    availability: Dict[str, Tuple[int,int]]  # e.g., {'mon': (6,14), ...}
