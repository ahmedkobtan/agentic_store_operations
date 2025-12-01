from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uuid

from walmart_ahmedkobtan_agentic_store_operations.src.services.propose_schedule import run_schedule
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import (
    ARTIFACT_OUT_DIR,
    SCHEMA_VERSION,
    SOLVER_VERSION,
)


class ScheduleRequest(BaseModel):
    horizon_days: int = 3
    role_targets_path: Optional[str] = None
    roster_csv: Optional[str] = None
    constraints_yaml: Optional[str] = None


class ScheduleResponse(BaseModel):
    run_id: str
    summary_path: str
    proposal_path: str
    schema_version: str
    solver_version: str


app = FastAPI(title="Scheduler Service")


@app.post("/schedule", response_model=ScheduleResponse)
def schedule_endpoint(req: ScheduleRequest) -> ScheduleResponse:
    run_id = str(uuid.uuid4())
    # Delegate to existing programmatic entrypoint used by Streamlit UI
    res = run_schedule(
        horizon_days=req.horizon_days,
        targets=req.role_targets_path,
        roster=req.roster_csv,
        constraints=req.constraints_yaml,
        time_limit_s=30,
    )

    # Return artifact paths for downstream services/UI
    return ScheduleResponse(
        run_id=run_id,
        summary_path=str(res["summary_path"]),
        proposal_path=str(res["schedule_path_csv"]),
        schema_version=SCHEMA_VERSION,
        solver_version=SOLVER_VERSION,
    )
