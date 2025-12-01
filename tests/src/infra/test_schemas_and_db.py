from datetime import datetime, timezone
from walmart_ahmedkobtan_agentic_store_operations.src.infra.schemas import (
    ForecastMessage, ScheduleRequest, ScheduleResponse, ManagerFeedback, Assignment, ManagerEdit
)
from walmart_ahmedkobtan_agentic_store_operations.src.infra.db import init_db, get_engine, log_schedule, log_manager_feedback, log_forecasts


def test_schema_roundtrip():
    fm = ForecastMessage(store_id="S1", date_hour=datetime.now(timezone.utc), p10=1, p50=2, p90=3, gen_time=datetime.now(timezone.utc))
    req = ScheduleRequest(store_id="S1", date=datetime.now(timezone.utc), horizon_hours=24, targets_per_hour=[fm])
    asg = Assignment(employee_id="E1", role="cashier", start=datetime.now(timezone.utc), end=datetime.now(timezone.utc))
    resp = ScheduleResponse(schedule_id="sch1", store_id="S1", date=datetime.now(timezone.utc), assignments=[asg], cost=10.0, shortfall_hours=0.0, overstaff_hours=1.0)
    edit = ManagerEdit(action="modify")
    fb = ManagerFeedback(schedule_id="sch1", edits=[edit], reason_codes=["demand_spike"], manager_id="mgr1", ts=datetime.now(timezone.utc))
    assert resp.schedule_id == fb.schedule_id
    assert fm.model_version.startswith("forecast")
    assert resp.solver_version.startswith("cpsat")


def test_db_logging(tmp_path):
    engine = init_db(get_engine("sqlite:///:memory:"))
    fm = ForecastMessage(store_id="S1", date_hour=datetime.now(timezone.utc), p10=1, p50=2, p90=3, gen_time=datetime.now(timezone.utc))
    log_forecasts(engine, [fm.model_dump()])
    schedule_json = {"assignments": []}
    log_schedule(engine, schedule_id="schX", store_id="S1", date=datetime.now(timezone.utc), schedule_json=schedule_json,
                 cost=100.0, shortfall_hours=0.0, overstaff_hours=0.0, solver_version="cpsat_0.1.0",
                 model_version="forecast_0.1.0", schema_version="0.1.0")
    log_manager_feedback(engine, schedule_id="schX", edits=[], reasons=["ok"], manager_id="mgr1", schema_version="0.1.0")
    # Basic assertion: tables exist and insertions didn't raise exceptions
    # (Further queries could be added if needed)
    assert True
