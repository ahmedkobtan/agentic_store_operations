from fastapi.testclient import TestClient
from pathlib import Path

from walmart_ahmedkobtan_agentic_store_operations.services.adjustment_service import app as adjustment_app
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import ARTIFACT_OUT_DIR


def test_adjustment_service_applies_or_skips_bias():
    client = TestClient(adjustment_app)
    resp = client.post("/apply_bias", json={"apply_bias": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"]
    assert Path(data["adjusted_path_parquet"]).exists()
    assert Path(data["adjusted_path_csv"]).exists()
    # Points should not be empty
    assert len(data["points"]) > 0
    assert data["schema_version"] == "1.0"
    assert data["model_version"]
