import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.dashboard import feedback
from src.dashboard.detection import DEMO_PATH, risk_levels, score_upload, validate_access_log

PROCESSED_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "reports" / "figures"

st.set_page_config(page_title="Behavioral Anomaly Detection", page_icon="🛡️", layout="wide")

st.markdown("""<style>
    .block-container {max-width: 1320px; padding-top: 2rem;}
    h1, h2, h3 {letter-spacing: -.035em;}
    [data-testid="stMetric"] {border: 1px solid rgba(128, 146, 166, .24);
        border-radius: 12px; padding: 1rem; background: rgba(98, 130, 158, .07);}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(128, 146, 166, .18);}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    scored = pd.read_csv(DEMO_PATH, parse_dates=["timestamp"], low_memory=False)
    with open(DEMO_PATH.parent / "metrics.json") as f:
        metrics = json.load(f)
    return scored, metrics


@st.cache_resource(show_spinner="Loading the trained detection models…")
def cached_bundle():
    from src.models.persistence import load_bundle
    return load_bundle()


@st.cache_data
def load_demo_summary():
    path = PROCESSED_DIR / "demo_coldstart_drift.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def top_level_metrics(scored, metrics, decisions):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions scored", f"{len(scored):,}")
    c2.metric("PR-AUC (binary anomaly)", f"{metrics['pr_auc_binary_anomaly']:.3f}")
    c3.metric(
        f"Precision @ top {metrics['alert_budget_fraction']:.0%}",
        f"{metrics['precision_at_alert_budget']:.3f}",
    )
    c4.metric("Analyst decisions logged", f"{len(decisions):,}")


def alert_detail(alert, scored):
    st.markdown(f"### {alert['entity_id']} · {alert['predicted_type'].replace('_', ' ').title()}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk score", f"{alert['risk_score']:.3f}")
    c2.metric("Effective (after feedback)", f"{alert['effective_risk']:.3f}",
              delta=f"{alert['risk_adjustment']:+.2f}" if alert["risk_adjustment"] else None)
    c3.metric("Type confidence", f"{alert['confidence']:.3f}")
    c4.metric("Prior sessions", f"{int(alert['entity_history_length']):,}")
    st.caption(f"Event {alert['session_id']} · {alert['risk_level']} risk band · "
               "Bands are based on score percentiles within this dataset, not threat probabilities.")

    explanation = alert.get("explanation")
    if isinstance(explanation, str) and explanation.strip():
        st.info(explanation)
    else:
        st.info("No explanation was generated for this session (below the attribution cutoff).")

    st.caption(
        f"{alert['timestamp']:%Y-%m-%d %H:%M:%S} | resource `{alert['resource_accessed']}` | "
        f"source IP `{alert['source_ip']}` | auth "
        f"{'succeeded' if alert['auth_success'] else 'failed'}"
    )

    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Recent session history for this entity**")
        history = scored[scored["entity_id"] == alert["entity_id"]].sort_values("timestamp")
        window = history[history["timestamp"] <= alert["timestamp"]].tail(30)
        if len(window) > 1:
            st.line_chart(window.set_index("timestamp")["risk_score"], height=180)
        st.dataframe(
            window[["timestamp", "resource_accessed", "predicted_type", "risk_score"]]
            .sort_values("timestamp", ascending=False),
            width="stretch", height=200, hide_index=True,
        )

    with right:
        st.markdown("**Score breakdown**")
        st.dataframe(
            pd.DataFrame({
                "component": ["baseline profiler", "tabular classifier", "sequence model"],
                "score": [
                    round(alert["baseline_risk"], 3),
                    round(alert["tabular_anomaly"], 3),
                    round(alert["sequence_anomaly"], 3),
                ],
            }),
            width="stretch", hide_index=True,
        )

        with st.expander("Raw SHAP contributions"):
            detail = alert.get("shap_detail")
            if isinstance(detail, str) and detail.strip():
                st.dataframe(pd.DataFrame(json.loads(detail)), width="stretch",
                             hide_index=True)
            else:
                st.caption("No attribution stored for this session.")

    st.markdown("**Triage**")
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Confirm threat", type="primary", width="stretch"):
        feedback.record_decision(alert["session_id"], alert["entity_id"], feedback.CONFIRM)
        st.rerun()
    if b2.button("Dismiss as false positive", width="stretch"):
        feedback.record_decision(alert["session_id"], alert["entity_id"], feedback.DISMISS)
        st.rerun()


def alert_queue_tab(scored, decisions):
    c1, c2, c3 = st.columns(3)
    entity_types = c1.multiselect(
        "Entity type", sorted(scored["entity_type"].unique()),
        default=list(scored["entity_type"].unique()),
    )
    pred_types = c2.multiselect(
        "Predicted type", sorted(scored["predicted_type"].unique()),
        default=[t for t in sorted(scored["predicted_type"].unique()) if t != "normal"],
    )
    budget_only = c3.checkbox("Top 1% risk budget only", value=True)

    filtered = scored[scored["entity_type"].isin(entity_types)]
    if pred_types:
        filtered = filtered[filtered["predicted_type"].isin(pred_types)]
    if budget_only:
        filtered = filtered[filtered["risk_score"] >= scored["risk_score"].quantile(0.99)]

    ip_filter = st.text_input("Filter by source IP", placeholder="Type part of an IP address")
    if ip_filter:
        filtered = filtered[filtered["source_ip"].astype(str).str.contains(ip_filter, case=False, regex=False)]

    decided = set(decisions["session_id"]) if not decisions.empty else set()
    if decided and st.checkbox("Hide sessions I've already triaged", value=True):
        filtered = filtered[~filtered["session_id"].isin(decided)]

    filtered = filtered.sort_values("effective_risk", ascending=False).head(300)
    st.caption(f"{len(filtered)} alerts shown, ranked by risk after analyst feedback")

    if filtered.empty:
        st.success("Nothing left in the queue for these filters.")
        return

    selection = st.dataframe(
        filtered[[
            "entity_id", "source_ip", "timestamp", "predicted_type", "risk_level",
            "effective_risk", "confidence", "resource_accessed", "explanation",
        ]],
        width="stretch", height=340, hide_index=True,
        on_select="rerun", selection_mode="single-row",
    )

    rows = selection["selection"]["rows"]
    st.divider()
    if not rows:
        st.caption("Select a row above, then open Investigate for the full event view.")
        return
    st.session_state["selected_session"] = filtered.iloc[rows[0]]["session_id"]
    st.info("Event selected. Open **Investigate** in the sidebar for the evidence and session history.")


def system_behavior_tab(demo):
    st.markdown(
        "How the system handles entities with no history, and behaviour that "
        "legitimately changes over time. Both charts are regenerated by "
        "`python -m src.demo.coldstart_drift`."
    )

    cold_start_png = FIGURE_DIR / "cold_start_progression.png"
    drift_png = FIGURE_DIR / "drift_vs_attack.png"

    if not cold_start_png.exists():
        st.warning("Run `python -m src.demo.coldstart_drift` to generate these charts.")
        return

    st.markdown("#### Cold start")
    st.image(str(cold_start_png), width="stretch")
    if demo:
        checkpoints = pd.DataFrame(demo["cold_start"]["checkpoints"])
        st.dataframe(
            checkpoints.rename(columns={
                "session": "session #",
                "mean_blended_risk": "reported risk",
                "mean_raw_risk": "raw model risk",
                "std_across_devices": "sd across devices",
                "blend_weight_on_own_history": "weight on own history",
            }).drop(columns=["n_devices"]),
            width="stretch", hide_index=True,
        )
        first = demo["cold_start"]["checkpoints"][0]
        st.caption(
            f"A brand-new device scores {first['mean_raw_risk']:.2f} from the models alone - "
            f"it is novel on every feature. Blending against the population baseline reports "
            f"{first['mean_blended_risk']:.2f} instead, which is what keeps new devices out of "
            "the alert queue."
        )

    st.markdown("#### Concept drift")
    st.image(str(drift_png), width="stretch")
    if demo:
        mech = demo["drift_mechanism"]
        pop = demo["drift_vs_attack"]["population"]
        st.caption(
            f"Left: after a permanent change in behaviour, the trailing "
            f"{mech['rolling_window']} baseline settles back to |z| "
            f"{mech['rolling_z_once_settled']:.2f}, while a baseline that is never refreshed "
            f"stays at {mech['frozen_baseline_z_once_settled']:.1f} forever. "
            f"Right: this only covers timing and duration. Resource-footprint growth is not "
            f"forgiven - median risk rose {pop['median_risk_before']:.2f} to "
            f"{pop['median_risk_during_shift']:.2f} across "
            f"{pop['n_drift_entities_considered']} drifting entities, though none of them "
            "crossed the top-1% alert threshold."
        )


def feedback_tab(decisions, scored):
    st.markdown(
        "Dismissals push an entity down the queue and confirmations pull it up, by "
        f"{feedback.PERCENTILE_STEP:.2f} percentile ranks per decision, capped at "
        f"**{feedback.PERCENTILE_CAP:.2f} percentile ranks** however many decisions "
        "are recorded. Because the top-1% queue spans exactly one percentile point, "
        f"that cap means an entity can be moved through at most "
        f"{feedback.PERCENTILE_CAP * 100:.0f}% of the queue. This only reorders the "
        "queue - it does not retrain the models or change any stored risk score."
    )
    st.caption(
        "The earlier version capped the raw score offset at "
        f"+/-{feedback.MAX_STEPS * feedback.DISMISS_STEP:.2f} instead. That bounded "
        "far less than it looked like: scores cluster tightly just above the cutoff, "
        "so 0.10 of score was wider than the cutoff margin for 96% of queued entities."
    )

    if decisions.empty:
        st.info("No decisions recorded yet. Confirm or dismiss an alert to start.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Decisions**")
        st.dataframe(
            decisions.sort_values("decided_at", ascending=False),
            width="stretch", height=280, hide_index=True,
        )
    with c2:
        st.markdown("**Current per-entity movement**")
        steps = feedback.entity_net_steps(decisions)
        moved = (
            scored[scored["entity_id"].isin(steps)]
            .groupby("entity_id")["pct_movement"].max()
            .to_dict()
        )
        st.dataframe(
            pd.DataFrame([
                {
                    "entity_id": eid,
                    "net decisions": net,
                    "percentile movement": round(moved.get(eid, 0.0), 3),
                    "at cap": abs(moved.get(eid, 0.0)) >= feedback.PERCENTILE_CAP - 1e-9,
                }
                for eid, net in steps.items()
            ]).sort_values("percentile movement"),
            width="stretch", height=280, hide_index=True,
        )


def overview(scored):
    st.title("Behavioral Anomaly Detection")
    st.subheader("ML-powered behavioral analytics for detecting anomalous access activity.")
    st.write("Explore synthetic access sessions, spot unusual patterns, and inspect the model evidence behind each alert.")
    st.info("Start with **Detect Anomalies → Load Demo Dataset**. The demo is a fixed 8,000-event excerpt from the project's scored synthetic dataset.")
    predicted_anomalies = scored["predicted_type"].ne("normal")
    counts = [len(scored), int((~predicted_anomalies).sum()), int(predicted_anomalies.sum()),
              scored["entity_id"].nunique(), scored["source_ip"].nunique()]
    for column, label, value in zip(st.columns(5),
                                    ["Total events", "Predicted normal", "Predicted anomalies", "Unique entities", "Unique IPs"], counts):
        column.metric(label, f"{value:,}")
    st.metric("Predicted anomaly rate", f"{predicted_anomalies.mean():.1%}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Anomaly activity over time")
        activity = (scored.assign(day=scored["timestamp"].dt.date, anomaly=predicted_anomalies)
                    .groupby("day")["anomaly"].sum())
        st.area_chart(activity, color="#3fa9a2", height=280)
    with right:
        st.markdown("#### Predicted anomaly types")
        types = scored.loc[predicted_anomalies, "predicted_type"].value_counts()
        st.bar_chart(types, horizontal=True, color="#79a8db", height=280)
    st.markdown("#### How it works")
    st.caption("Access logs → Behavioral features → Isolation Forest + Random Forest + GRU → Risk ranking → Analyst investigation")


def detect_page(scored):
    st.title("Detect Anomalies")
    st.write("Run the saved demo instantly, or score your own access-log CSV with the project's trained ensemble.")
    source = st.radio("Data source", ["Bundled demo", "Upload CSV"], horizontal=True)
    if source == "Bundled demo":
        st.caption("Fixed synthetic excerpt; scores and SHAP explanations were produced by the original full pipeline. The button loads those saved results without retraining.")
        if st.button("Load Demo Dataset · Run Detection", type="primary"):
            st.session_state["active_scored"] = scored
            st.session_state["source_name"] = "Bundled demo · saved ensemble results"
            st.success("Demo results loaded. Review suspicious events below.")
    else:
        st.caption("Upload up to 10,000 rows in the project's access-log format. Use the sample below as a schema example; scores are calculated from the uploaded events.")
        @st.cache_data
        def sample_csv():
            return (DEMO_PATH.parent / "access_log.csv").read_bytes()
        st.download_button("Download sample access log", sample_csv(), "sample_access_log.csv", "text/csv")
        uploaded = st.file_uploader("Access-log CSV", type="csv")
        if uploaded is not None and st.button("Run Detection", type="primary"):
            try:
                incoming = validate_access_log(pd.read_csv(uploaded, low_memory=False))
                with st.spinner("Engineering behavioral features and scoring events…"):
                    st.session_state["active_scored"] = score_upload(incoming, cached_bundle())
                st.session_state["source_name"] = "Uploaded CSV · live ensemble inference"
                st.success("Detection complete.")
            except (ValueError, FileNotFoundError, KeyError) as exc:
                st.error(str(exc))

    active = st.session_state.get("active_scored")
    if active is None:
        st.info("Choose a source and run detection to see the alert queue.")
        return
    active = active.copy()
    active["risk_level"] = risk_levels(active["risk_score"])
    decisions = feedback.load_decisions()
    active = feedback.apply_feedback(active, decisions)
    st.divider()
    st.subheader("Detection results")
    anomalous = active["predicted_type"].ne("normal")
    for column, label, value in zip(st.columns(4),
                                    ["Events", "Predicted anomalies", "Anomaly rate", "Model"],
                                    [f"{len(active):,}", f"{anomalous.sum():,}", f"{anomalous.mean():.1%}", "3-model ensemble"]):
        column.metric(label, value)
    st.caption(st.session_state.get("source_name", "") + " · Predictions are model outputs, not verified incidents.")
    alert_queue_tab(active, decisions)


def investigate_page():
    st.title("Investigate Event")
    scored = st.session_state.get("active_scored")
    if scored is None:
        st.info("Run detection first, then select a suspicious event here.")
        return
    scored = scored.copy()
    scored["risk_level"] = risk_levels(scored["risk_score"])
    scored = feedback.apply_feedback(scored, feedback.load_decisions())
    alerts = scored.sort_values("risk_score", ascending=False).head(300)
    selected = st.session_state.get("selected_session")
    ids = alerts["session_id"].tolist()
    index = ids.index(selected) if selected in ids else 0
    choice = st.selectbox(
        "Suspicious event", ids, index=index,
        format_func=lambda sid: (lambda row: f"{row['entity_id']} · {row['predicted_type'].replace('_', ' ')} · {row['risk_score']:.3f} · {row['timestamp']:%b %d %H:%M}")(
            alerts.loc[alerts["session_id"] == sid].iloc[0]),
    )
    st.session_state["selected_session"] = choice
    alert_detail(alerts.loc[alerts["session_id"] == choice].iloc[0], scored)


def insights_page(metrics, scored):
    st.title("Model Insights")
    st.write("Three existing models contribute to each risk score. Their weights are 25% baseline, 35% tabular, and 40% sequence before cold-start blending.")
    for heading, description in [
        ("Isolation Forest", "Identifies unusual behavior by measuring how readily sessions separate from the population for each entity type."),
        ("Random Forest", "Classifies attack patterns from engineered session features. Its tree model provides the SHAP attributions shown in investigations."),
        ("GRU sequence model", "Reads each entity's recent sequence of sessions to add temporal context to anomaly and type predictions."),
    ]:
        with st.container(border=True):
            st.markdown(f"**{heading}**")
            st.caption(description)
    st.subheader("Held-out evaluation")
    st.caption("These figures come from the project's saved time-split evaluation of the full synthetic dataset. They are not recalculated for the demo excerpt or uploads.")
    for column, label, value in zip(st.columns(3),
                                    ["Binary anomaly PR-AUC", "Precision at top 1%", "Test sessions"],
                                    [f"{metrics['pr_auc_binary_anomaly']:.3f}",
                                     f"{metrics['precision_at_alert_budget']:.3f}",
                                     f"{metrics['n_test_sessions']:,}"]):
        column.metric(label, value)
    report = metrics["classification_report"]
    rows = [{"type": key, "precision": value["precision"], "recall": value["recall"],
             "F1": value["f1-score"], "support": int(value["support"])}
            for key, value in report.items() if isinstance(value, dict) and key not in ("macro avg", "weighted avg")]
    st.dataframe(pd.DataFrame(rows).style.format({"precision": "{:.3f}", "recall": "{:.3f}", "F1": "{:.3f}"}),
                 hide_index=True, width="stretch")
    st.caption("SHAP contributions for an individual alert are available in Investigate. They explain the Random Forest's predicted type, not the full blended risk score.")
    if scored is not None:
        st.subheader("Current result distribution")
        st.bar_chart(scored["predicted_type"].value_counts(), horizontal=True)
    system_behavior_tab(load_demo_summary())


def methodology_page():
    st.title("About / Methodology")
    st.markdown("""### Problem
Access logs contain rare attacks among routine activity. The project prioritizes suspicious sessions for an analyst and explains the evidence available for review.

### Data and features
The bundled demonstration is a fixed sample of synthetic users, service accounts, and edge-device access sessions. The generator injects labeled attack patterns. Feature engineering includes location changes, session timing, authentication failures, device changes, resource breadth, and rolling behavioral baselines.

### Models and evaluation
Entity-type Isolation Forests, a Random Forest classifier, and a GRU sequence model feed a weighted risk score. The saved metrics use a time-based holdout from the full generated dataset; no live ground truth is assumed for uploaded logs.

### Explainability
SHAP attributes the Random Forest's predicted class to tabular features for high-risk sessions. The sentence shown to an analyst is deterministic and only names indicators present in those features.

### Limitations
This is a synthetic-data research demo, not a production intrusion detector. Risk scores are relative rankings, and the severity bands are dataset percentiles. Attack-type precision varies substantially; feedback reorders a local queue but does not retrain the models. Uploaded data is scored with features built from the uploaded batch, so incomplete history can change results. Streamlit Community Cloud's local feedback database is ephemeral.
""")
    st.code("Access log → Feature engineering → Isolation Forest + Random Forest + GRU → Risk ranking → SHAP → Analyst review", language=None)


def main():
    scored, metrics = load_data()
    st.sidebar.markdown("## Security analytics")
    st.sidebar.caption("Behavioral Anomaly Detection")
    page = st.sidebar.radio("Navigate", ["Overview", "Detect Anomalies", "Investigate", "Model Insights", "About / Methodology", "Analyst Feedback"])
    if page == "Overview":
        overview(scored)
    elif page == "Detect Anomalies":
        detect_page(scored)
    elif page == "Investigate":
        investigate_page()
    elif page == "Model Insights":
        insights_page(metrics, st.session_state.get("active_scored"))
    elif page == "About / Methodology":
        methodology_page()
    else:
        st.title("Analyst Feedback")
        active = st.session_state.get("active_scored")
        if active is None:
            st.info("Run detection to review feedback.")
        else:
            decisions = feedback.load_decisions()
            feedback_tab(decisions, feedback.apply_feedback(active, decisions))


if __name__ == "__main__":
    main()
