from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

SCHEMA_VERSION = "0.1.0"
MODEL_VERSION = "forecast_0.1.0"
SOLVER_VERSION = "cpsat_0.1.0"


class ForecastMessage(BaseModel):
    store_id: str
    date_hour: datetime
    p10: float
    p50: float
    p90: float
    gen_time: datetime
    model_version: str = MODEL_VERSION
    schema_version: str = SCHEMA_VERSION


class Assignment(BaseModel):
    employee_id: str
    role: str
    start: datetime
    end: datetime


class ScheduleRequest(BaseModel):
    store_id: str
    date: datetime
    horizon_hours: int
    targets_per_hour: List[ForecastMessage]
    employee_pool_snapshot_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION


class ScheduleResponse(BaseModel):
    schedule_id: str
    store_id: str
    date: datetime
    assignments: List[Assignment]
    cost: float
    shortfall_hours: float
    overstaff_hours: float
    solver_version: str = SOLVER_VERSION
    rationale_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION


class ManagerEdit(BaseModel):
    assignment_id: Optional[str] = None
    employee_id: Optional[str] = None
    role: Optional[str] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    action: str = Field(description="add|remove|modify")


class ManagerFeedback(BaseModel):
    schedule_id: str
    edits: List[ManagerEdit]
    reason_codes: List[str]
    manager_id: str
    ts: datetime
    schema_version: str = SCHEMA_VERSION


__all__ = [
    "ForecastMessage",
    "ScheduleRequest",
    "ScheduleResponse",
    "ManagerFeedback",
    "Assignment",
    "ManagerEdit",
]
