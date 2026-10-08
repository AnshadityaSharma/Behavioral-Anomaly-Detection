"""Input validation and inference for the Streamlit analyst demo."""

from pathlib import Path

import pandas as pd

from src.generator.generate_dataset import COLUMN_ORDER

ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = ROOT / "data" / "demo" / "scored_sessions.csv"
MAX_UPLOAD_ROWS = 10_000


def validate_access_log(frame):
    missing = sorted(set(COLUMN_ORDER) - set(frame.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("The CSV contains no events.")
    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError(f"Upload at most {MAX_UPLOAD_ROWS:,} events per run.")

    clean = frame[COLUMN_ORDER].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce")
    if clean["timestamp"].isna().any():
        raise ValueError("Every timestamp must be a valid date and time.")
    if clean["session_id"].isna().any() or clean["session_id"].duplicated().any():
        raise ValueError("session_id values must be present and unique.")
    for column in ("geo_lat", "geo_lon", "session_duration"):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        if clean[column].isna().any():
            raise ValueError(f"{column} must contain numbers in every row.")
    if not clean["auth_success"].isin([True, False, "True", "False", "true", "false", 1, 0, "1", "0"]).all():
        raise ValueError("auth_success must contain true or false values.")
    clean["auth_success"] = clean["auth_success"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    for column in ("entity_id", "entity_type", "source_ip", "resource_accessed", "device_os", "device_mac"):
        if clean[column].isna().any() or clean[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must be present in every row.")
    return clean


def risk_levels(scores):
    """Within-batch percentile bands; descriptive, not calibrated probabilities."""
    ranks = scores.rank(pct=True, method="average")
    return pd.cut(
        ranks, bins=[0, 0.90, 0.98, 0.995, 1],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"], include_lowest=True,
    ).astype(str)


def score_upload(access_log, bundle):
    """Run the project's existing feature builder, ensemble and SHAP explainer."""
    import numpy as np

    from src.explain.attribution import build_explainer, explain_sessions
    from src.features.build_features import build_features
    from src.pipeline import score_frame

    features = build_features(validate_access_log(access_log))
    scores = score_frame(features, bundle)
    scored = features[[
        "session_id", "entity_id", "entity_type", "timestamp", "resource_accessed",
        "source_ip", "auth_success", "entity_history_length", "is_cold_start",
    ]].copy()
    for column, values in scores.items():
        if column != "raw_risk":
            scored[column] = values
    scored["explanation"] = ""
    scored["reasons"] = ""
    scored["shap_detail"] = ""

    # Match the pipeline's attribution policy, with a small public-demo cap.
    n_explain = min(100, max(1, int(len(scored) * 0.05)))
    indices = np.argsort(-scored["risk_score"].to_numpy())[:n_explain]
    explained = explain_sessions(
        build_explainer(bundle["tabular"]),
        features.iloc[indices].reset_index(drop=True),
        scored.iloc[indices]["predicted_type"].to_numpy(),
        bundle["tabular"].classes_,
    )
    for column in ("explanation", "reasons", "shap_detail"):
        scored.loc[indices, column] = explained[column].to_numpy()
    return scored.sort_values("risk_score", ascending=False).reset_index(drop=True)
