import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.dashboard import feedback

PROCESSED_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "reports" / "figures"

st.set_page_config(page_title="Behavioral Anomaly Detection", layout="wide")


@st.cache_data
def load_data():
    scored = pd.read_csv(
        PROCESSED_DIR / "scored_sessions.csv", parse_dates=["timestamp"], low_memory=False
    )
    with open(PROCESSED_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return scored, metrics


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
    st.markdown(f"### {alert['entity_id']} - {alert['predicted_type']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk score", f"{alert['risk_score']:.3f}")
    c2.metric("Effective (after feedback)", f"{alert['effective_risk']:.3f}",
              delta=f"{alert['risk_adjustment']:+.2f}" if alert["risk_adjustment"] else None)
    c3.metric("Type confidence", f"{alert['confidence']:.3f}")
    c4.metric("Prior sessions", f"{int(alert['entity_history_length']):,}")

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
            use_container_width=True, height=200, hide_index=True,
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
            use_container_width=True, hide_index=True,
        )

        with st.expander("Raw SHAP contributions"):
            detail = alert.get("shap_detail")
            if isinstance(detail, str) and detail.strip():
                st.dataframe(pd.DataFrame(json.loads(detail)), use_container_width=True,
                             hide_index=True)
            else:
                st.caption("No attribution stored for this session.")

    st.markdown("**Triage**")
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Confirm threat", type="primary", use_container_width=True):
        feedback.record_decision(alert["session_id"], alert["entity_id"], feedback.CONFIRM)
        st.rerun()
    if b2.button("Dismiss as false positive", use_container_width=True):
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
            "entity_id", "entity_type", "timestamp", "predicted_type",
            "effective_risk", "risk_adjustment", "confidence", "resource_accessed", "explanation",
        ]],
        use_container_width=True, height=340, hide_index=True,
        on_select="rerun", selection_mode="single-row",
    )

    rows = selection["selection"]["rows"]
    st.divider()
    if not rows:
        st.caption("Select a row above to open the alert.")
        return
    alert_detail(filtered.iloc[rows[0]], scored)


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
    st.image(str(cold_start_png), use_container_width=True)
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
            use_container_width=True, hide_index=True,
        )
        first = demo["cold_start"]["checkpoints"][0]
        st.caption(
            f"A brand-new device scores {first['mean_raw_risk']:.2f} from the models alone - "
            f"it is novel on every feature. Blending against the population baseline reports "
            f"{first['mean_blended_risk']:.2f} instead, which is what keeps new devices out of "
            "the alert queue."
        )

    st.markdown("#### Concept drift")
    st.image(str(drift_png), use_container_width=True)
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
            use_container_width=True, height=280, hide_index=True,
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
            use_container_width=True, height=280, hide_index=True,
        )


def main():
    st.title("Behavioral Anomaly Detection - Analyst Dashboard")

    scored, metrics = load_data()
    decisions = feedback.load_decisions()
    scored = feedback.apply_feedback(scored, decisions)

    top_level_metrics(scored, metrics, decisions)
    st.divider()

    queue_tab, behavior_tab, review_tab = st.tabs(
        ["Alert queue", "System behaviour", "Analyst feedback"]
    )
    with queue_tab:
        alert_queue_tab(scored, decisions)
    with behavior_tab:
        system_behavior_tab(load_demo_summary())
    with review_tab:
        feedback_tab(decisions, scored)


if __name__ == "__main__":
    main()
