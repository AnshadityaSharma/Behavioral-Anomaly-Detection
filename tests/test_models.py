import numpy as np

from src.features.build_features import build_features
from src.generator.config import ALL_LABELS
from src.generator.generate_dataset import COLUMN_ORDER, build_dataset
from src.models.baseline import BaselineProfiler
from src.models.classifier import TabularClassifier
from src.models.cold_start import blend_cold_start_scores
from src.models.sequence_model import SequenceAnomalyModel, build_sequences


def _features_with_labels(small_config):
    df, _ = build_dataset(seed=4)
    access_log = df[COLUMN_ORDER].copy()
    features = build_features(access_log)
    return features.merge(df[["session_id", "label"]], on="session_id")


def test_baseline_profiler_scores_in_unit_range(small_config):
    features = _features_with_labels(small_config)
    profiler = BaselineProfiler().fit(features)
    scores = profiler.score(features)
    assert len(scores) == len(features)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0 + 1e-9


def test_cold_start_blend_shrinks_toward_prior(small_config):
    features = _features_with_labels(small_config)
    raw = np.random.RandomState(0).rand(len(features))
    prior = {t: 0.5 for t in features["entity_type"].unique()}
    blended = blend_cold_start_scores(features, raw, prior)
    cold_mask = features["is_cold_start"] == 1
    assert (blended[cold_mask.values] != raw[cold_mask.values]).any()


def test_tabular_classifier_predicts_known_labels(small_config):
    features = _features_with_labels(small_config)
    clf = TabularClassifier().fit(features)
    preds = clf.predict(features)
    assert set(preds) <= set(ALL_LABELS)
    assert len(preds) == len(features)


def test_sequence_model_output_shape_and_normalization(small_config):
    features = _features_with_labels(small_config)
    x, y, session_ids = build_sequences(features, window=5)
    assert x.shape[0] == len(features)
    assert x.shape[1] == 5
    assert len(session_ids) == len(features)

    model = SequenceAnomalyModel(window=5, epochs=1).fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (len(features), len(ALL_LABELS))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
