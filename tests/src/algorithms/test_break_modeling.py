import pandas as pd
from pathlib import Path

from walmart_ahmedkobtan_agentic_store_operations.src.services.propose_schedule import run_schedule
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import ARTIFACT_OUT_DIR, CONSTRAINTS_YAML_PATH, ROSTER_FILE


def test_long_shifts_split_for_break():
    # Run a short horizon schedule solve
    res = run_schedule(horizon_days=1)
    proposal_csv = Path(ARTIFACT_OUT_DIR) / "schedule_proposal.csv"
    assert proposal_csv.exists()
    df = pd.read_csv(proposal_csv)
    # Ensure long non-lead shifts are split (lead may remain long for continuous coverage)
    long_non_lead = df[(df['length'] >= 7) & (df['role'] != 'lead')]
    assert len(long_non_lead) == 0, long_non_lead
    # If any original demand would have produced long shifts, we expect split segments (length between min_shift and threshold)
    # Not strictly asserting counts due to variability; presence of schedule rows suffices
    assert len(df) > 0
