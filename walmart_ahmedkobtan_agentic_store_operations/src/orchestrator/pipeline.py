from __future__ import annotations

from typing import Dict, Any
from fastapi.testclient import TestClient

from walmart_ahmedkobtan_agentic_store_operations.services.forecast_service import app as forecast_app
from walmart_ahmedkobtan_agentic_store_operations.services.scheduler_service import app as scheduler_app
from walmart_ahmedkobtan_agentic_store_operations.services.compliance_service import app as compliance_app
from walmart_ahmedkobtan_agentic_store_operations.services.adjustment_service import app as adjustment_app
from walmart_ahmedkobtan_agentic_store_operations.src.infra.trace import new_trace_id, TraceEvent, Timer
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import SCHEMA_VERSION


def run_workflow(store_id: str = "store_1", horizon_days: int = 3, apply_bias: bool = False) -> Dict[str, Any]:
    trace_id = new_trace_id()
    audit: Dict[str, Any] = {"trace_id": trace_id, "events": []}

    # Forecast
    fc_client = TestClient(forecast_app)
    with Timer() as t:
        fc_resp = fc_client.post("/forecast", json={"store_id": store_id, "horizon_days": horizon_days})
    fc_ok = fc_resp.status_code == 200
    audit["events"].append(
        TraceEvent(trace_id, "forecast", "ok" if fc_ok else "error", t.ms, SCHEMA_VERSION, {"status_code": fc_resp.status_code}).to_dict()
    )
    fc = fc_resp.json() if fc_ok else {}

    # Optional bias adjustment (produces adjusted targets artifact)
    adjusted_path = None
    if apply_bias:
        adj_client = TestClient(adjustment_app)
        with Timer() as t:
            adj_resp = adj_client.post("/apply_bias", json={"apply_bias": True})
        adj_ok = adj_resp.status_code == 200
        audit["events"].append(
            TraceEvent(trace_id, "adjustment", "ok" if adj_ok else "error", t.ms, SCHEMA_VERSION, {"status_code": adj_resp.status_code}).to_dict()
        )
        if adj_ok:
            adjusted_path = adj_resp.json().get("adjusted_path_parquet") or adj_resp.json().get("adjusted_path_csv")

    # Schedule
    sch_client = TestClient(scheduler_app)
    sch_payload = {"horizon_days": horizon_days}
    if adjusted_path:
        sch_payload["role_targets_path"] = adjusted_path
    with Timer() as t:
        sch_resp = sch_client.post("/schedule", json=sch_payload)
    sch_ok = sch_resp.status_code == 200
    audit["events"].append(
        TraceEvent(trace_id, "scheduler", "ok" if sch_ok else "error", t.ms, SCHEMA_VERSION, {"status_code": sch_resp.status_code, "adjusted": bool(adjusted_path)}).to_dict()
    )
    sch = sch_resp.json() if sch_ok else {}

    # Compliance
    comp_client = TestClient(compliance_app)
    with Timer() as t:
        comp_resp = comp_client.post("/validate", params={"proposal_path": sch.get("proposal_path", "")})
    comp_ok = comp_resp.status_code == 200
    audit["events"].append(
        TraceEvent(trace_id, "compliance", "ok" if comp_ok else "error", t.ms, SCHEMA_VERSION, {"status_code": comp_resp.status_code}).to_dict()
    )
    comp = comp_resp.json() if comp_ok else {}

    return {
        "trace_id": trace_id,
        "forecast": fc,
        "schedule": sch,
        "compliance": comp,
        "audit": audit,
    }
