import streamlit as st
import pandas as pd
import uuid
from datetime import datetime
from pathlib import Path

from walmart_ahmedkobtan_agentic_store_operations.src.services.propose_schedule import run_schedule
from walmart_ahmedkobtan_agentic_store_operations.src.services.evaluate_schedule import evaluate as eval_schedule
from walmart_ahmedkobtan_agentic_store_operations.src.algorithms.learning import apply_biases_to_targets
from walmart_ahmedkobtan_agentic_store_operations.src.infra.db import init_db, get_engine, log_manager_feedback, get_bias_profile, upsert_bias
from walmart_ahmedkobtan_agentic_store_operations.src.infra.schemas import ManagerFeedback, ManagerEdit, SCHEMA_VERSION
from fastapi.testclient import TestClient
from walmart_ahmedkobtan_agentic_store_operations.services.compliance_service import app as compliance_app
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import ARTIFACT_OUT_DIR, MAX_BIAS

st.set_page_config(page_title="Store Staffing Agent", layout="wide")

engine = init_db(get_engine())

st.title("Store Staffing Agent MVP")

col_left, col_right = st.columns([2,1])

with col_left:
    st.header("Forecast & Targets")
    forecast_path = ARTIFACT_OUT_DIR / "df_forecast_next7d_how.csv"
    targets_path_csv = ARTIFACT_OUT_DIR / "role_targets_next7d.csv"
    if forecast_path.exists():
        df_f = pd.read_csv(forecast_path, parse_dates=["timestamp_local"]) if "timestamp_local" in pd.read_csv(forecast_path, nrows=1).columns else pd.read_csv(forecast_path)
        st.subheader("Raw Forecast (head)")
        st.dataframe(df_f.head())
    if targets_path_csv.exists():
        df_t = pd.read_csv(targets_path_csv, parse_dates=["timestamp_local"])
        st.subheader("Role Targets (head)")
        st.dataframe(df_t.head())
    apply_bias = st.checkbox("Apply learned biases", value=False)
    if apply_bias and targets_path_csv.exists():
        bias_profile = get_bias_profile(engine)
        df_t_b = apply_biases_to_targets(df_t, bias_profile)
        st.dataframe(df_t_b.head())

with col_right:
    st.header("Actions")
    horizon_days = st.slider("Horizon Days", min_value=1, max_value=3, value=3)
    if st.button("Propose Schedule"):
        res = run_schedule(horizon_days=horizon_days)
        st.success(f"Schedule proposed run_id={res['summary']['run_id']}")
        # Auto-run compliance and show panel
        proposal_csv = ARTIFACT_OUT_DIR / "schedule_proposal.csv"
        comp_client = TestClient(compliance_app)
        comp_resp = comp_client.post(
            "/validate",
            params={
                "proposal_path": str(proposal_csv),
                "config_path": str(ARTIFACT_OUT_DIR.parent / "configs" / "compliance_rules.yaml")
            }
        )
        if comp_resp.status_code == 200:
            comp = comp_resp.json()
            st.subheader("Compliance Results")
            st.write(f"Pass: {comp['pass_']} | Schema v{comp['schema_version']}")
            if comp['violations']:
                vdf = pd.DataFrame(comp['violations'])
                st.dataframe(vdf)
            st.caption(f"Checked rules: {', '.join(comp['checked_rules'])}")
    if st.button("Evaluate Current Schedule"):
        eval_schedule()
        # Show latest compliance again (if schedule exists)
        proposal_csv = ARTIFACT_OUT_DIR / "schedule_proposal.csv"
        if proposal_csv.exists():
            comp_client = TestClient(compliance_app)
            comp_resp = comp_client.post("/validate", params={"proposal_path": str(proposal_csv)})
            if comp_resp.status_code == 200:
                comp = comp_resp.json()
                st.subheader("Compliance (Latest)")
                st.write(f"Pass: {comp['pass_']}")
                if comp['violations']:
                    st.dataframe(pd.DataFrame(comp['violations']))
    st.markdown("---")
    st.subheader("Feedback")
    schedule_id = st.text_input("Schedule ID (run_id)")
    hour_to_delta = st.text_input("Hour->Delta JSON (e.g. {\"14\":1,\"15\":-1})", value="{}")
    reasons = st.text_input("Reason codes comma separated", value="demand_spike")
    manager_id = st.text_input("Manager ID", value="mgr001")
    if st.button("Submit Feedback") and schedule_id:
        try:
            import json
            deltas = json.loads(hour_to_delta)
            edits = []
            for h, d in deltas.items():
                edits.append(ManagerEdit(action="modify"))
                # Store bias update directly
                existing = get_bias_profile(engine)
                prev = existing.get(int(h), 0.0)
                upsert_bias(engine, int(h), max(-MAX_BIAS, min(MAX_BIAS, prev + float(d))))
            fb = ManagerFeedback(
                schedule_id=schedule_id,
                edits=edits,
                reason_codes=[r.strip() for r in reasons.split(",")],
                manager_id=manager_id,
                ts=datetime.utcnow()
            )
            log_manager_feedback(engine, fb.schedule_id, [e.model_dump() for e in fb.edits], fb.reason_codes, fb.manager_id, SCHEMA_VERSION)
            st.success("Feedback logged & biases updated.")
        except Exception as e:
            st.error(f"Failed to parse/submit feedback: {e}")

st.markdown("---")
st.caption("Artifacts directory: " + str(ARTIFACT_OUT_DIR))
st.caption("Compliance config: data/configs/compliance_rules.yaml")
