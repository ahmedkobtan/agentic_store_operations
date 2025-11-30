from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from pathlib import Path
import uuid

from walmart_ahmedkobtan_agentic_store_operations.src.algorithms.forecast_how_next7d import (
    forecast_next7d_how,
)
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import (
    ARTIFACT_OUT_DIR,
    SCHEMA_VERSION,
    MODEL_VERSION,
)


class ForecastRequest(BaseModel):
    store_id: str = "store_1"
    horizon_days: int = 7
    seed: Optional[int] = None


class ForecastPoint(BaseModel):
    timestamp_local: str
    p10: float
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    run_id: str
    store_id: str
    points: List[ForecastPoint]
    schema_version: str
    model_version: str


app = FastAPI(title="Forecast Service")


@app.post("/forecast", response_model=ForecastResponse)
def forecast_endpoint(req: ForecastRequest) -> ForecastResponse:
    run_id = str(uuid.uuid4())
    # Use existing helper to generate hour-of-week forecast for next N days
    df = forecast_next7d_how(horizon_days=req.horizon_days, seed=req.seed)

    # Persist artifact for interoperability
    out_path = Path(ARTIFACT_OUT_DIR) / "df_forecast_next7d_how.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    points = []
    for _, r in df.iterrows():
        p50 = r.get("yhat_p50")
        if p50 is None and "yhat" in r.index:
            p50 = r["yhat"]
        p10 = r.get("yhat_p10")
        if p10 is None:
            p10 = float(p50) * 0.9
        p90 = r.get("yhat_p90")
        if p90 is None:
            p90 = float(p50) * 1.1
        points.append(
            ForecastPoint(
                timestamp_local=str(r["timestamp_local"]),
                p10=float(p10),
                p50=float(p50),
                p90=float(p90),
            )
        )
    return ForecastResponse(
        run_id=run_id,
        store_id=req.store_id,
        points=points,
        schema_version=SCHEMA_VERSION,
        model_version=MODEL_VERSION,
    )
