from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict


def new_trace_id() -> str:
    return str(uuid.uuid4())


@dataclass
class TraceEvent:
    trace_id: str
    service: str
    status: str
    latency_ms: int
    payload_version: str
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.t1 = time.perf_counter_ns()
        return False

    @property
    def ms(self) -> int:
        return int((self.t1 - self.t0) / 1_000_000)
