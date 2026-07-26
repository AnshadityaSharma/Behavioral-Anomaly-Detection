import numpy as np
import shap

from src.features.build_features import FEATURE_COLUMNS
from src.models.cold_start import cold_start_note

FEATURE_LABELS = {
    "geo_velocity_kmh": lambda v: f"implausible travel speed ({v:.0f} km/h since last session)",
    "geo_distance_from_home_km": lambda v: f"access from {v:.0f} km outside usual location",
    "is_new_resource": lambda v: "first-ever access to this resource" if v else None,
    "is_device_mismatch": lambda v: "device fingerprint does not match this entity's known device",
    "ip_distinct_entities_10min": lambda v: f"source IP shared with {v:.0f} other entities in the last 10 min",
    "ip_failure_rate_10min": lambda v: f"{v:.0%} auth failure rate from this source IP recently",
    "ip_session_count_10min": lambda v: f"{v:.0f} sessions from this source IP in the last 10 min",
    "entity_failed_count_10min": lambda v: f"{v:.0f} failed logins from this entity in the last 10 min",
    "entity_resource_breadth_24h": lambda v: f"{v:.0f} distinct resources touched in 24h (unusually broad)",
    "hour_of_day_z": lambda v: f"login time far from usual pattern (z={v:.1f})",
    "session_duration_z": lambda v: f"session duration far from usual pattern (z={v:.1f})",
    "has_privileged_command": lambda v: "privileged command executed during session",
    "command_sequence_length": lambda v: f"unusually long command sequence ({v:.0f} actions)",
    "auth_failed": lambda v: "authentication failed" if v else None,
    "hours_since_last_session": lambda v: f"{v:.0f} hours since last activity",
}


def build_explainer(tabular_classifier):
    return shap.TreeExplainer(tabular_classifier.model)


def _class_index(classes, label):
    return classes.index(label) if label in classes else 0


def compute_reasons(explainer, features_df, predicted_labels, classes, top_k=3):
    x = features_df[FEATURE_COLUMNS].values
    explanation = explainer(x)
    values = explanation.values

    reasons = []
    for i in range(len(features_df)):
        label = predicted_labels[i]
        if values.ndim == 3:
            cls_idx = _class_index(classes, label)
            row_shap = values[i, :, cls_idx]
        else:
            row_shap = values[i]

        order = np.argsort(-np.abs(row_shap))[:top_k]
        row_reasons = []
        for j in order:
            feat = FEATURE_COLUMNS[j]
            value = features_df.iloc[i][feat]
            if row_shap[j] <= 0:
                continue
            template = FEATURE_LABELS.get(feat)
            text = template(value) if template else f"{feat}={value:.2f}"
            if text:
                row_reasons.append(text)

        note = cold_start_note(features_df.iloc[i]["entity_history_length"])
        if note:
            row_reasons.append(note)

        reasons.append("; ".join(row_reasons) if row_reasons else "no single dominant factor")

    return reasons
