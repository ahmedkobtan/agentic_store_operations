from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import uuid
import pandas as pd

from walmart_ahmedkobtan_agentic_store_operations.src.infra.db import get_engine, init_db, get_bias_profile
from walmart_ahmedkobtan_agentic_store_operations.src.algorithms.learning import apply_biases_to_targets
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import (
    ARTIFACT_OUT_DIR,
    SCHEMA_VERSION,
    MODEL_VERSION,
)


class AdjustmentRequest(BaseModel):
    role_targets_path: Optional[str] = None  # existing targets (parquet or csv)
    apply_bias: bool = True


class AdjustedTarget(BaseModel):
    timestamp_local: str
    lead_needed: int
    cashier_needed: int
    floor_needed: int


class AdjustmentResponse(BaseModel):
    run_id: str
    adjusted_path_parquet: str
    adjusted_path_csv: str
    applied_bias: bool
    points: List[AdjustedTarget]
    schema_version: str
    model_version: str


app = FastAPI(title="Adjustment Service")


def _load_targets(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["timestamp_local"]) if "timestamp_local" in pd.read_csv(path, nrows=1).columns else pd.read_csv(path)


@app.post("/apply_bias", response_model=AdjustmentResponse)
def apply_bias_endpoint(req: AdjustmentRequest) -> AdjustmentResponse:
    run_id = str(uuid.uuid4())
    # Resolve targets path (fallback to artifact next7d role targets)
    base_path = Path(req.role_targets_path) if req.role_targets_path else Path(ARTIFACT_OUT_DIR) / "role_targets_next7d.parquet"
    if not base_path.exists():
        # Attempt CSV fallback
        alt = base_path.with_suffix(".csv")
        if alt.exists():
            base_path = alt
        else:
            raise FileNotFoundError(f"Role targets not found at {base_path}")

    df = _load_targets(base_path)
    if "timestamp_local" not in df.columns:
        raise ValueError("Targets file missing 'timestamp_local' column")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp_local"]):
        df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])  # ensure datetime

    applied = False
    if req.apply_bias:
        eng = init_db()
        bias_profile = get_bias_profile(eng)
        if bias_profile:  # only apply if something learned
            df = apply_biases_to_targets(df, bias_profile)
            applied = True

    out_dir = Path(ARTIFACT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / f"role_targets_adjusted_{run_id}.parquet"
    out_csv = out_dir / f"role_targets_adjusted_{run_id}.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    points = []
    for _, r in df.iterrows():
        points.append(
            AdjustedTarget(
                timestamp_local=str(r["timestamp_local"]),
                lead_needed=int(r.get("lead_needed", 0)),
                cashier_needed=int(r.get("cashier_needed", 0)),
                floor_needed=int(r.get("floor_needed", 0)),
            )
        )

    return AdjustmentResponse(
        run_id=run_id,
        adjusted_path_parquet=str(out_parquet),
        adjusted_path_csv=str(out_csv),
        applied_bias=applied,
        points=points,
        schema_version=SCHEMA_VERSION,
        model_version=MODEL_VERSION,
    )
