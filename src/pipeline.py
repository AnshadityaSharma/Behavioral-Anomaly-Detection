import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report

from src.explain.attribution import build_explainer, compute_reasons
from src.features.build_features import FEATURE_COLUMNS, build_features
from src.generator.config import ALL_LABELS, SIM_DAYS
from src.generator.generate_dataset import build_dataset
from src.models.baseline import BaselineProfiler
from src.models.classifier import TabularClassifier
from src.models.cold_start import blend_cold_start_scores
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
    df, entities = build_dataset(seed)
    labels = df[["session_id", "label"]].copy()
    from src.generator.generate_dataset import COLUMN_ORDER
    access_log = df[COLUMN_ORDER].copy()

    access_log.to_csv(access_path, index=False)
    labels.to_csv(labels_path, index=False)
    entities.to_csv(entities_path, index=False)
    return access_log, labels, entities


def time_split_mask(timestamps, train_fraction=TRAIN_FRACTION):
    cutoff_day = int(SIM_DAYS * train_fraction)
    cutoff_ts = timestamps.min() + pd.Timedelta(days=cutoff_day)
    return timestamps < cutoff_ts


def run(seed=42):
    access_log, labels, entities = load_or_generate_raw(seed)
    features = build_features(access_log)
    features = features.merge(labels, on="session_id", how="left")

    train_mask = time_split_mask(features["timestamp"])
    train_df = features[train_mask].reset_index(drop=True)

    baseline = BaselineProfiler().fit(train_df)
    baseline_risk = baseline.score(features)

    tabular = TabularClassifier().fit(train_df)
    tabular_proba = tabular.predict_proba(features)
    tabular_classes = tabular.classes_
    normal_idx = tabular_classes.index("normal")
    tabular_anomaly = 1 - tabular_proba[:, normal_idx]

    x_seq, y_seq, seq_session_ids = build_sequences(features)
    seq_order = pd.Series(range(len(seq_session_ids)), index=seq_session_ids)
    features_order = seq_order.loc[features["session_id"]].values
    x_seq = x_seq[features_order]
    y_seq = y_seq[features_order]

    seq_train_mask = train_mask.values
    seq_model = SequenceAnomalyModel(epochs=8).fit(x_seq[seq_train_mask], y_seq[seq_train_mask])
    seq_proba = seq_model.predict_proba(x_seq)
    normal_label_idx = ALL_LABELS.index("normal")
    seq_anomaly = 1 - seq_proba[:, normal_label_idx]

    tabular_proba_aligned = np.zeros((len(features), len(ALL_LABELS)))
    for j, cls in enumerate(tabular_classes):
        tabular_proba_aligned[:, ALL_LABELS.index(cls)] = tabular_proba[:, j]
    combined_proba = (tabular_proba_aligned + seq_proba) / 2
    predicted_idx = combined_proba.argmax(axis=1)
    predicted_type = [ALL_LABELS[i] for i in predicted_idx]
    confidence = combined_proba.max(axis=1)

    raw_risk = (
        RISK_WEIGHTS["baseline"] * baseline_risk
        + RISK_WEIGHTS["tabular"] * tabular_anomaly
        + RISK_WEIGHTS["sequence"] * seq_anomaly
    )
    type_prior_map = pd.Series(raw_risk, index=features["entity_type"]).groupby(level=0).mean().to_dict()
    risk_score = blend_cold_start_scores(features, raw_risk, type_prior_map)

    # SHAP attribution is only worth the cost for sessions an analyst would
    # actually see, not the entire (mostly normal) session log.
    reasons = np.full(len(features), "", dtype=object)
    reason_cutoff = np.quantile(risk_score, 1 - REASON_COVERAGE)
    reason_mask = risk_score >= reason_cutoff
    explainer = build_explainer(tabular)
    reasons[reason_mask] = compute_reasons(
        explainer,
        features[reason_mask].reset_index(drop=True),
        np.array(predicted_type)[reason_mask],
        tabular_classes,
    )

    scored = features[[
        "session_id", "entity_id", "entity_type", "timestamp", "resource_accessed",
        "source_ip", "auth_success", "entity_history_length", "is_cold_start", "label",
    ]].copy()
    scored["risk_score"] = risk_score
    scored["predicted_type"] = predicted_type
    scored["confidence"] = confidence
    scored["reasons"] = reasons
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
