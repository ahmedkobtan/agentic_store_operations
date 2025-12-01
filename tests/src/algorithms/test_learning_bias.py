from walmart_ahmedkobtan_agentic_store_operations.src.algorithms.learning import update_biases_from_feedback, apply_biases_to_targets
import pandas as pd
from datetime import datetime, timedelta


def _make_targets():
    base_dt = datetime(2025, 1, 1, 6, 0, 0)
    hours = [base_dt + timedelta(hours=i) for i in range(10)]
    df = pd.DataFrame({
        "timestamp_local": hours,
        "cashier_needed": [1]*10,
        "floor_needed": [2]*10,
        "lead_needed": [1]*10,
    })
    return df


def test_update_biases_positive():
    existing = {h:0.0 for h in range(24)}
    deltas = {14: 1.0, 15: 2.0}
    updated = update_biases_from_feedback(existing, deltas)
    assert updated[14] > 0 and updated[15] > updated[14]


def test_apply_biases():
    targets = _make_targets()
    bias_profile = { (ts.hour): 1.0 for ts in targets["timestamp_local"][:3] }
    biased = apply_biases_to_targets(targets, bias_profile)
    # First three hours cashier_needed should be bumped from 1 to 2
    assert biased.loc[:2, "cashier_needed"].tolist() == [2,2,2]
