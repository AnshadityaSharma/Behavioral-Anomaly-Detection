import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

st.set_page_config(page_title="Behavioral Anomaly Detection", layout="wide")


@st.cache_data
def load_data():
    scored = pd.read_csv(PROCESSED_DIR / "scored_sessions.csv", parse_dates=["timestamp"])
    with open(PROCESSED_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return scored, metrics


def top_level_metrics(scored, metrics):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions scored", f"{len(scored):,}")
    c2.metric("PR-AUC (binary anomaly)", f"{metrics['pr_auc_binary_anomaly']:.3f}")
    c3.metric(
        f"Precision @ top {metrics['alert_budget_fraction']:.0%} alert budget",
        f"{metrics['precision_at_alert_budget']:.3f}",
    )
    c4.metric(
        "False positive rate @ budget",
        f"{metrics['false_positive_rate_at_alert_budget']:.3f}",
    )


def alert_queue(scored):
    st.subheader("Alert queue")

    col1, col2, col3 = st.columns(3)
    entity_types = col1.multiselect(
        "Entity type", sorted(scored["entity_type"].unique()), default=list(scored["entity_type"].unique())
    )
    pred_types = col2.multiselect(
        "Predicted type", sorted(scored["predicted_type"].unique()),
        default=[t for t in scored["predicted_type"].unique() if t != "normal"],
    )
    budget_only = col3.checkbox("Top 1% risk budget only", value=True)

    filtered = scored[scored["entity_type"].isin(entity_types)]
    if pred_types:
        filtered = filtered[filtered["predicted_type"].isin(pred_types)]
    if budget_only:
        cutoff = scored["risk_score"].quantile(0.99)
        filtered = filtered[filtered["risk_score"] >= cutoff]

    display_cols = [
        "entity_id", "entity_type", "timestamp", "predicted_type", "risk_score",
        "confidence", "resource_accessed", "source_ip", "reasons", "label",
    ]
    st.dataframe(
        filtered[display_cols].sort_values("risk_score", ascending=False),
        use_container_width=True,
        height=420,
    )
    return filtered


def entity_history(scored):
    st.subheader("Entity history")
    entity_ids = sorted(scored["entity_id"].unique())
    default_idx = 0
    top_entity = scored.sort_values("risk_score", ascending=False).iloc[0]["entity_id"]
    if top_entity in entity_ids:
        default_idx = entity_ids.index(top_entity)

    selected = st.selectbox("Entity", entity_ids, index=default_idx)
    history = scored[scored["entity_id"] == selected].sort_values("timestamp")

    st.line_chart(history.set_index("timestamp")["risk_score"])

    st.caption(f"{len(history)} sessions on record for {selected}")
    st.dataframe(
        history[[
            "timestamp", "resource_accessed", "source_ip", "predicted_type",
            "risk_score", "reasons", "label",
        ]],
        use_container_width=True,
        height=300,
    )


def main():
    st.title("Behavioral Anomaly Detection - Analyst Dashboard")
    st.caption(
        "Ranked alerts, risk scores, and contributing factors from the "
        "baseline profiler, sequence model, and tabular classifier ensemble."
    )

    scored, metrics = load_data()
    top_level_metrics(scored, metrics)
    st.divider()
    alert_queue(scored)
    st.divider()
    entity_history(scored)


if __name__ == "__main__":
    main()
