from __future__ import annotations
from typing import Dict
import pandas as pd

from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import ALPHA, MAX_BIAS

def update_biases_from_feedback(existing: Dict[int, float], edit_deltas: Dict[int, float]) -> Dict[int, float]:
    """Given existing bias profile and aggregated hour->delta edits, return updated profile.
    edit_deltas[hour] = (final_staffed - proposed_staffed) for that hour aggregated across feedbacks.
    Positive delta means manager increased staffing -> bias up; negative means bias down.
    """
    updated = existing.copy()
    for h, delta in edit_deltas.items():
        prev = updated.get(h, 0.0)
        new_val = (1 - ALPHA) * prev + ALPHA * delta
        # Cap for safety
        if new_val > MAX_BIAS:
            new_val = MAX_BIAS
        if new_val < -MAX_BIAS:
            new_val = -MAX_BIAS
        updated[h] = new_val
    return updated


def apply_biases_to_targets(targets: pd.DataFrame, bias_profile: Dict[int, float], roles=("cashier","floor")) -> pd.DataFrame:
    t = targets.copy()
    hour = t["timestamp_local"].dt.hour
    for role in roles:
        col = f"{role}_needed"
        if col not in t.columns:
            continue
        adjustments = hour.map(lambda h: bias_profile.get(int(h), 0.0))
        # Apply and floor to zero
        t[col] = (t[col] + adjustments).clip(lower=0).round().astype(int)
    return t


__all__ = ["update_biases_from_feedback", "apply_biases_to_targets"]
