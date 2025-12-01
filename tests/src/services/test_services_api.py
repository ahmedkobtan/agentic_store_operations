from fastapi.testclient import TestClient
import os
from pathlib import Path

from walmart_ahmedkobtan_agentic_store_operations.services.forecast_service import app as forecast_app
from walmart_ahmedkobtan_agentic_store_operations.services.scheduler_service import app as scheduler_app
from walmart_ahmedkobtan_agentic_store_operations.services.compliance_service import app as compliance_app

from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import ARTIFACT_OUT_DIR


def test_forecast_service_runs():
    client = TestClient(forecast_app)
    resp = client.post("/forecast", json={"store_id": "store_1", "horizon_days": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    assert data["store_id"] == "store_1"
    assert len(data["points"]) > 0
    assert data["schema_version"] == "1.0"
    assert data["model_version"]


def test_scheduler_service_runs(tmp_path):
    client = TestClient(scheduler_app)
    # Use defaults; service should read existing artifacts/targets
    resp = client.post("/schedule", json={"horizon_days": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    assert Path(data["proposal_path"]).exists()
    assert Path(data["summary_path"]).exists()
    assert data["schema_version"] == "1.0"
    assert data["solver_version"]


def test_compliance_service_basic():
    # Validate last proposal stored under artifacts
    proposal_csv = Path(ARTIFACT_OUT_DIR) / "schedule_proposal.csv"
    client = TestClient(compliance_app)
    resp = client.post("/validate", params={"proposal_path": str(proposal_csv)})
    assert resp.status_code == 200
    data = resp.json()
    assert "pass_" in data
    assert "violations" in data
    assert data["schema_version"] == "1.0"
