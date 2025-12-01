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
st.caption(f"Schema v{SCHEMA_VERSION}")

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
        # Visualize proposed schedule with explicit breaks
        proposal_csv = ARTIFACT_OUT_DIR / "schedule_proposal.csv"
        if proposal_csv.exists():
            df_sched = pd.read_csv(proposal_csv, parse_dates=["day"]) if "day" in pd.read_csv(proposal_csv, nrows=1).columns else pd.read_csv(proposal_csv)
            st.subheader("Schedule (head)")
            st.dataframe(df_sched.head())
            # Simple Gantt-style view per employee with break highlight
            try:
                import plotly.express as px
                df_plot = df_sched.copy()
                df_plot["start_ts"] = pd.to_datetime(df_plot["day"]) + pd.to_timedelta(df_plot["start_hour"], unit="h")
                df_plot["end_ts"] = pd.to_datetime(df_plot["day"]) + pd.to_timedelta(df_plot["end_hour"], unit="h")
                fig = px.timeline(df_plot, x_start="start_ts", x_end="end_ts", y="employee_id", color="role", hover_data=["has_break","break_start_hour","break_end_hour"])
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
                # Overlay breaks as red bars
                if "has_break" in df_plot.columns:
                    df_b = df_plot[df_plot["has_break"] == True].dropna(subset=["break_start_hour","break_end_hour"])
                    if not df_b.empty:
                        df_b["break_start_ts"] = pd.to_datetime(df_b["day"]) + pd.to_timedelta(df_b["break_start_hour"], unit="h")
                        df_b["break_end_ts"] = pd.to_datetime(df_b["day"]) + pd.to_timedelta(df_b["break_end_hour"], unit="h")
                        fig_b = px.timeline(df_b, x_start="break_start_ts", x_end="break_end_ts", y="employee_id", color_discrete_sequence=["#ff4d4d"], hover_name="role")
                        fig_b.update_traces(opacity=0.6, showlegend=False)
                        st.plotly_chart(fig_b, use_container_width=True)
            except Exception as e:
                st.info(f"Install plotly for Gantt visualization. Error: {e}")
        # Auto-run compliance and show panel
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
            st.write(f"Pass: {comp['pass_']} | Schema v{comp['schema_version']} | Rules v{comp.get('rules_version')}")
            if comp['violations']:
                vdf = pd.DataFrame(comp['violations'])
                st.dataframe(vdf)
            st.caption(f"Checked rules: {', '.join(comp['checked_rules'])}")
        # Metrics summary
        summary_path = ARTIFACT_OUT_DIR / "schedule_summary.json"
        if summary_path.exists():
            import json as _json
            summ = _json.loads(summary_path.read_text())
            st.subheader("Schedule Metrics")
            # Compute totals if not present
            total_shortfall = sum(summ.get("shortfall_by_role_hours", {}).values()) if "total_shortfall_hours" not in summ else summ.get("total_shortfall_hours")
            total_overstaff = sum(summ.get("overstaff_by_role_hours", {}).values()) if "total_overstaff_hours" not in summ else summ.get("total_overstaff_hours")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Cost", f"${summ.get('total_cost',0):.2f}")
            col_b.metric("Shortfall Hrs", f"{total_shortfall}")
            col_c.metric("Overstaff Hrs", f"{total_overstaff}")
            # Role breakdown table
            rb = pd.DataFrame({
                "role": ["lead","cashier","floor"],
                "shortfall_hours": [summ.get("shortfall_by_role_hours", {}).get("lead",0), summ.get("shortfall_by_role_hours", {}).get("cashier",0), summ.get("shortfall_by_role_hours", {}).get("floor",0)],
                "overstaff_hours": [summ.get("overstaff_by_role_hours", {}).get("lead",0), summ.get("overstaff_by_role_hours", {}).get("cashier",0), summ.get("overstaff_by_role_hours", {}).get("floor",0)]
            })
            st.dataframe(rb)
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
        summary_path = ARTIFACT_OUT_DIR / "schedule_summary.json"
        if summary_path.exists():
            import json as _json
            summ = _json.loads(summary_path.read_text())
            st.subheader("Current Metrics")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Cost", f"${summ.get('total_cost',0):.2f}")
            col_b.metric("Shortfall Hrs", f"{summ.get('total_shortfall_hours',0)}")
            col_c.metric("Overstaff Hrs", f"{summ.get('total_overstaff_hours',0)}")
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
