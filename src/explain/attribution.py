import json

import numpy as np
import pandas as pd
import shap

from src.explain.narrative import compose_sentence
from src.features.build_features import FEATURE_COLUMNS
from src.models.cold_start import cold_start_note


def build_explainer(tabular_classifier):
    return shap.TreeExplainer(tabular_classifier.model)


def _class_index(classes, label):
    return classes.index(label) if label in classes else 0


def top_contributions(explainer, features_df, predicted_labels, classes, top_k=3):
    """Per row, the top_k features pushing the score toward the predicted class."""
    values = explainer(features_df[FEATURE_COLUMNS].values).values

    per_row = []
    for i in range(len(features_df)):
        if values.ndim == 3:
            row_shap = values[i, :, _class_index(classes, predicted_labels[i])]
        else:
            row_shap = values[i]

        order = np.argsort(-row_shap)[:top_k]
        contributions = [
            (FEATURE_COLUMNS[j], features_df.iloc[i][FEATURE_COLUMNS[j]], float(row_shap[j]))
            for j in order
            if row_shap[j] > 0
        ]
        per_row.append(contributions)
    return per_row


def explain_sessions(explainer, features_df, predicted_labels, classes, top_k=3):
    """Readable sentence per alert, plus the raw SHAP values behind it."""
    per_row = top_contributions(explainer, features_df, predicted_labels, classes, top_k)

    explanations, reasons, details = [], [], []
    for i, contributions in enumerate(per_row):
        row = features_df.iloc[i]
        note = cold_start_note(row["entity_history_length"])
        pairs = [(feat, value) for feat, value, _ in contributions]

        explanations.append(compose_sentence(predicted_labels[i], pairs, row, note))
        reasons.append("; ".join(feat for feat, _, _ in contributions))
        details.append(json.dumps([
            {"feature": feat, "value": float(value), "shap": shap_value}
            for feat, value, shap_value in contributions
        ]))

    return pd.DataFrame({
        "explanation": explanations,
        "reasons": reasons,
        "shap_detail": details,
    })
