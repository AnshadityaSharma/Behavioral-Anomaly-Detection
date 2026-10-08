import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report

from src.explain.attribution import build_explainer, explain_sessions
from src.features.build_features import build_features
from src.generator.config import ALL_LABELS, COLUMN_ORDER, SIM_DAYS
from src.models.baseline import BaselineProfiler
from src.models.classifier import TabularClassifier
from src.models.cold_start import blend_cold_start_scores
from src.models.persistence import save_bundle
from src.models.sequence_model import SequenceAnomalyModel, build_sequences

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

TRAIN_FRACTION = 0.7
RISK_WEIGHTS = {"baseline": 0.25, "tabular": 0.35, "sequence": 0.40}
ALERT_BUDGET = 0.01
REASON_COVERAGE = 0.05


def load_or_generate_raw(seed):
    access_path = RAW_DIR / "access_log.csv"
    labels_path = RAW_DIR / "labels.csv"
    entities_path = RAW_DIR / "entities.csv"

    if access_path.exists() and labels_path.exists() and entities_path.exists():
        access_log = pd.read_csv(access_path, parse_dates=["timestamp"])
        labels = pd.read_csv(labels_path)
        entities = pd.read_csv(entities_path)
        return access_log, labels, entities

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    from src.generator.generate_dataset import build_dataset

    df, entities = build_dataset(seed)
    labels = df[["session_id", "label"]].copy()
    access_log = df[COLUMN_ORDER].copy()

    access_log.to_csv(access_path, index=False)
    labels.to_csv(labels_path, index=False)
    entities.to_csv(entities_path, index=False)
    return access_log, labels, entities


def time_split_mask(timestamps, train_fraction=TRAIN_FRACTION):
    cutoff_day = int(SIM_DAYS * train_fraction)
    cutoff_ts = timestamps.min() + pd.Timedelta(days=cutoff_day)
    return timestamps < cutoff_ts


def train_models(train_df):
    baseline = BaselineProfiler().fit(train_df)
    tabular = TabularClassifier().fit(train_df)

    x_seq, y_seq, seq_session_ids = build_sequences(train_df)
    seq_model = SequenceAnomalyModel(epochs=8).fit(x_seq, y_seq)

    # the cold-start prior is fixed on training data so that scoring a new
    # entity later doesn't depend on whatever else happens to be in the batch.
    train_components = _component_scores(train_df, baseline, tabular, seq_model)
    type_prior_map = (
        pd.Series(train_components["raw_risk"], index=train_df["entity_type"])
        .groupby(level=0).mean().to_dict()
    )

    return {
        "baseline": baseline,
        "tabular": tabular,
        "sequence": seq_model,
        "type_prior_map": type_prior_map,
    }


def _component_scores(features, baseline, tabular, seq_model):
    baseline_risk = baseline.score(features)

    tabular_proba = tabular.predict_proba(features)
    tabular_classes = tabular.classes_
    tabular_anomaly = 1 - tabular_proba[:, tabular_classes.index("normal")]

    x_seq, _, seq_session_ids = build_sequences(features)
    seq_order = pd.Series(range(len(seq_session_ids)), index=seq_session_ids)
    x_seq = x_seq[seq_order.loc[features["session_id"]].values]
    seq_proba = seq_model.predict_proba(x_seq)
    seq_anomaly = 1 - seq_proba[:, ALL_LABELS.index("normal")]

    tabular_proba_aligned = np.zeros((len(features), len(ALL_LABELS)))
    for j, cls in enumerate(tabular_classes):
        tabular_proba_aligned[:, ALL_LABELS.index(cls)] = tabular_proba[:, j]
    combined_proba = (tabular_proba_aligned + seq_proba) / 2

    raw_risk = (
        RISK_WEIGHTS["baseline"] * baseline_risk
        + RISK_WEIGHTS["tabular"] * tabular_anomaly
        + RISK_WEIGHTS["sequence"] * seq_anomaly
    )

    return {
        "baseline_risk": baseline_risk,
        "tabular_anomaly": tabular_anomaly,
        "sequence_anomaly": seq_anomaly,
        "combined_proba": combined_proba,
        "raw_risk": raw_risk,
    }


def score_frame(features, models):
    components = _component_scores(
        features, models["baseline"], models["tabular"], models["sequence"]
    )
    combined_proba = components["combined_proba"]
    predicted_idx = combined_proba.argmax(axis=1)

    risk_score = blend_cold_start_scores(
        features, components["raw_risk"], models["type_prior_map"]
    )

    return {
        "risk_score": risk_score,
        "raw_risk": components["raw_risk"],
        "predicted_type": [ALL_LABELS[i] for i in predicted_idx],
        "confidence": combined_proba.max(axis=1),
        "baseline_risk": components["baseline_risk"],
        "tabular_anomaly": components["tabular_anomaly"],
        "sequence_anomaly": components["sequence_anomaly"],
    }


def run(seed=42):
    access_log, labels, entities = load_or_generate_raw(seed)
    features = build_features(access_log)
    features = features.merge(labels, on="session_id", how="left")

    train_mask = time_split_mask(features["timestamp"])
    train_df = features[train_mask].reset_index(drop=True)

    models = train_models(train_df)
    scores = score_frame(features, models)
    risk_score = scores["risk_score"]
    predicted_type = scores["predicted_type"]

    save_bundle(models)

    # SHAP attribution is only worth the cost for sessions an analyst would
    # actually see, not the entire (mostly normal) session log.
    reason_cutoff = np.quantile(risk_score, 1 - REASON_COVERAGE)
    reason_mask = risk_score >= reason_cutoff
    explainer = build_explainer(models["tabular"])
    explained = explain_sessions(
        explainer,
        features[reason_mask].reset_index(drop=True),
        np.array(predicted_type)[reason_mask],
        models["tabular"].classes_,
    )

    scored = features[[
        "session_id", "entity_id", "entity_type", "timestamp", "resource_accessed",
        "source_ip", "auth_success", "entity_history_length", "is_cold_start", "label",
    ]].copy()
    scored["risk_score"] = risk_score
    scored["predicted_type"] = predicted_type
    scored["confidence"] = scores["confidence"]
    scored["baseline_risk"] = scores["baseline_risk"]
    scored["tabular_anomaly"] = scores["tabular_anomaly"]
    scored["sequence_anomaly"] = scores["sequence_anomaly"]

    scored["explanation"] = ""
    scored["reasons"] = ""
    scored["shap_detail"] = ""
    scored.loc[reason_mask, "explanation"] = explained["explanation"].values
    scored.loc[reason_mask, "reasons"] = explained["reasons"].values
    scored.loc[reason_mask, "shap_detail"] = explained["shap_detail"].values

    scored = scored.sort_values("risk_score", ascending=False).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(PROCESSED_DIR / "scored_sessions.csv", index=False)

    metrics = evaluate(features["label"].values, predicted_type, risk_score, train_mask.values)
    with open(PROCESSED_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return scored, metrics


def evaluate(true_labels, predicted_type, risk_score, train_mask):
    test_mask = ~train_mask
    y_true_test = true_labels[test_mask]
    y_pred_test = np.array(predicted_type)[test_mask]

    report = classification_report(y_true_test, y_pred_test, output_dict=True, zero_division=0)

    is_anomaly = (true_labels != "normal").astype(int)
    pr_auc = average_precision_score(is_anomaly[test_mask], risk_score[test_mask])

    n_alerts = max(1, int(len(y_true_test) * ALERT_BUDGET))
    order = np.argsort(-risk_score[test_mask])[:n_alerts]
    top_alert_true = is_anomaly[test_mask][order]
    precision_at_budget = top_alert_true.mean()
    false_positive_rate_at_budget = 1 - precision_at_budget

    return {
        "classification_report": report,
        "pr_auc_binary_anomaly": pr_auc,
        "alert_budget_fraction": ALERT_BUDGET,
        "precision_at_alert_budget": float(precision_at_budget),
        "false_positive_rate_at_alert_budget": float(false_positive_rate_at_budget),
        "n_test_sessions": int(test_mask.sum()),
        "n_test_anomalies": int(is_anomaly[test_mask].sum()),
    }


if __name__ == "__main__":
    scored, metrics = run()
    print(f"scored {len(scored)} sessions")
    print(f"PR-AUC (binary anomaly): {metrics['pr_auc_binary_anomaly']:.3f}")
    print(f"precision @ top {metrics['alert_budget_fraction']:.0%} alert budget: "
          f"{metrics['precision_at_alert_budget']:.3f}")
