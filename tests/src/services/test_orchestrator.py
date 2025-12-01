from walmart_ahmedkobtan_agentic_store_operations.src.orchestrator.pipeline import run_workflow


def test_run_workflow_end_to_end():
    res = run_workflow(store_id="store_1", horizon_days=1)
    assert "trace_id" in res
    assert "forecast" in res and res["forecast"].get("run_id")
    assert "schedule" in res and res["schedule"].get("summary_path")
    assert "compliance" in res and "pass_" in res["compliance"]
    assert "audit" in res and len(res["audit"]["events"]) == 3


def test_run_workflow_with_bias_adjustment():
    res = run_workflow(store_id="store_1", horizon_days=1, apply_bias=True)
    assert "trace_id" in res
    # Expect four events: forecast, adjustment, scheduler, compliance
    assert len(res["audit"]["events"]) == 4
    services = [e["service"] for e in res["audit"]["events"]]
    assert services == ["forecast", "adjustment", "scheduler", "compliance"]
