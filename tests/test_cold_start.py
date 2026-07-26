import numpy as np
import pandas as pd

from src.features.build_features import COLD_START_THRESHOLD
from src.models.cold_start import blend_cold_start_scores, cold_start_note


def _frame(history_lengths, entity_type="edge_device"):
    return pd.DataFrame({
        "entity_history_length": history_lengths,
        "entity_type": [entity_type] * len(history_lengths),
    })


def test_first_session_takes_the_population_prior():
    frame = _frame([0])
    blended = blend_cold_start_scores(frame, np.array([0.9]), {"edge_device": 0.2})
    assert blended[0] == 0.2


def test_settled_entity_keeps_its_own_score():
    frame = _frame([COLD_START_THRESHOLD, COLD_START_THRESHOLD + 50])
    raw = np.array([0.9, 0.9])
    blended = blend_cold_start_scores(frame, raw, {"edge_device": 0.2})
    np.testing.assert_allclose(blended, raw)


def test_blend_moves_monotonically_toward_own_score():
    history = np.arange(0, COLD_START_THRESHOLD + 1)
    frame = _frame(history)
    raw = np.full(len(history), 0.9)
    blended = blend_cold_start_scores(frame, raw, {"edge_device": 0.1})

    assert np.all(np.diff(blended) > 0)
    assert blended[0] == 0.1
    assert blended[-1] == 0.9


def test_blend_suppresses_a_spurious_new_entity_spike():
    """A brand-new device looks anomalous on every novelty feature; the blend
    is what stops that from reaching the alert queue."""
    frame = _frame([0, 1])
    blended = blend_cold_start_scores(frame, np.array([0.95, 0.95]), {"edge_device": 0.15})
    assert (blended < 0.5).all()


def test_unknown_entity_type_falls_back_to_mean():
    frame = _frame([0], entity_type="new_kind_of_thing")
    raw = np.array([0.8, 0.4])[:1]
    blended = blend_cold_start_scores(frame, raw, {"edge_device": 0.2})
    assert np.isfinite(blended).all()


def test_cold_start_note_only_below_threshold():
    assert cold_start_note(0) is not None
    assert cold_start_note(COLD_START_THRESHOLD - 1) is not None
    assert cold_start_note(COLD_START_THRESHOLD) is None
