from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional, List
from pathlib import Path
import json
import os

from sqlalchemy import (
    create_engine, Integer, String, Float, DateTime, Text, ForeignKey, select
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, relationship

from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import ARTIFACT_OUT_DIR


DB_PATH = os.getenv("STORE_AGENT_DB", str(ARTIFACT_OUT_DIR / "store_agent.db"))


class Base(DeclarativeBase):
    pass


class ForecastRow(Base):
    __tablename__ = "forecasts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    store_id: Mapped[str] = mapped_column(String(32))
    ts: Mapped[datetime] = mapped_column(DateTime)
    p10: Mapped[float] = mapped_column(Float)
    p50: Mapped[float] = mapped_column(Float)
    p90: Mapped[float] = mapped_column(Float)
    gen_at: Mapped[datetime] = mapped_column(DateTime)
    model_version: Mapped[str] = mapped_column(String(32))
    schema_version: Mapped[str] = mapped_column(String(16))


class ScheduleRow(Base):
    __tablename__ = "schedules"
    schedule_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    store_id: Mapped[str] = mapped_column(String(32))
    date: Mapped[datetime] = mapped_column(DateTime)
    json_blob: Mapped[str] = mapped_column(Text)
    cost: Mapped[float] = mapped_column(Float)
    shortfall_hours: Mapped[float] = mapped_column(Float)
    overstaff_hours: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    model_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    solver_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    schema_version: Mapped[str] = mapped_column(String(16))
    feedback: Mapped[List["ManagerFeedbackRow"]] = relationship(back_populates="schedule", cascade="all, delete-orphan")


class ManagerFeedbackRow(Base):
    __tablename__ = "manager_feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schedule_id: Mapped[str] = mapped_column(ForeignKey("schedules.schedule_id"))
    edits_json: Mapped[str] = mapped_column(Text)
    reasons: Mapped[str] = mapped_column(Text)
    manager_id: Mapped[str] = mapped_column(String(32))
    ts: Mapped[datetime] = mapped_column(DateTime)
    schema_version: Mapped[str] = mapped_column(String(16))
    schedule: Mapped[ScheduleRow] = relationship(back_populates="feedback")


class BiasRow(Base):
    __tablename__ = "bias_hourly"
    hour: Mapped[int] = mapped_column(Integer, primary_key=True)
    bias: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


def get_engine(path: str | None = None):
    db_file = path or DB_PATH
    if db_file.startswith("sqlite://"):
        return create_engine(db_file, future=True)
    return create_engine(f"sqlite:///{db_file}", future=True)


def init_db(engine=None):
    eng = engine or get_engine()
    Base.metadata.create_all(eng)
    return eng


def log_forecasts(engine, rows: List[dict]):
    with Session(engine) as ses:
        for r in rows:
            # Map schema keys -> DB columns
            mapped = {
                "store_id": r.get("store_id"),
                "ts": r.get("date_hour") or r.get("ts"),
                "p10": float(r.get("p10")),
                "p50": float(r.get("p50")),
                "p90": float(r.get("p90")),
                "gen_at": r.get("gen_time") or r.get("gen_at"),
                "model_version": r.get("model_version"),
                "schema_version": r.get("schema_version"),
            }
            ses.add(ForecastRow(**mapped))
        ses.commit()


def log_schedule(engine, schedule_id: str, store_id: str, date: datetime, schedule_json: dict,
                 cost: float, shortfall_hours: float, overstaff_hours: float, solver_version: str,
                 model_version: str | None, schema_version: str):
    with Session(engine) as ses:
        ses.add(ScheduleRow(
            schedule_id=schedule_id,
            store_id=store_id,
            date=date,
            json_blob=json.dumps(schedule_json),
            cost=cost,
            shortfall_hours=shortfall_hours,
            overstaff_hours=overstaff_hours,
            created_at=datetime.now(timezone.utc),
            model_version=model_version,
            solver_version=solver_version,
            schema_version=schema_version
        ))
        ses.commit()


def log_manager_feedback(engine, schedule_id: str, edits: List[dict], reasons: List[str], manager_id: str,
                         schema_version: str):
    with Session(engine) as ses:
        ses.add(ManagerFeedbackRow(
            schedule_id=schedule_id,
            edits_json=json.dumps(edits),
            reasons=json.dumps(reasons),
            manager_id=manager_id,
            ts=datetime.now(timezone.utc),
            schema_version=schema_version
        ))
        ses.commit()


def get_bias_profile(engine) -> dict:
    with Session(engine) as ses:
        rows = ses.execute(select(BiasRow)).scalars().all()
        return {r.hour: r.bias for r in rows}


def upsert_bias(engine, hour: int, bias: float):
    with Session(engine) as ses:
        row = ses.get(BiasRow, hour)
        now = datetime.now(timezone.utc)
        if row:
            row.bias = bias
            row.updated_at = now
        else:
            ses.add(BiasRow(hour=hour, bias=bias, updated_at=now))
        ses.commit()
