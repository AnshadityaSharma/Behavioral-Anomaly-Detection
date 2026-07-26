import numpy as np

from src.features.build_features import COLD_START_THRESHOLD


def blend_cold_start_scores(features_df, risk_scores, type_prior_scores, threshold=COLD_START_THRESHOLD):
    """New entities have too little history for their own profile or the
    sequence model's context window to mean much, so their raw score is
    unreliable in both directions. Blend it toward the entity_type's average
    risk instead of trusting it outright, weighted by how much history exists."""
    history = features_df["entity_history_length"].values
    weight = np.clip(history / threshold, 0, 1)
    prior = features_df["entity_type"].map(type_prior_scores).fillna(np.mean(risk_scores)).values
    return weight * risk_scores + (1 - weight) * prior


def cold_start_note(history_length, threshold=COLD_START_THRESHOLD):
    if history_length >= threshold:
        return None
    return f"limited history ({int(history_length)} prior sessions) - score blended with population baseline"
