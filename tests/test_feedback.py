import pandas as pd

from src.dashboard import feedback


def _db(tmp_path):
    return tmp_path / "feedback.db"


def test_no_decisions_means_no_adjustment(tmp_path):
    decisions = feedback.load_decisions(_db(tmp_path))
    assert decisions.empty
    assert feedback.entity_adjustments(decisions) == {}


def test_decisions_round_trip(tmp_path):
    db = _db(tmp_path)
    feedback.record_decision("s1", "user_1", feedback.DISMISS, db)
    feedback.record_decision("s2", "user_1", feedback.CONFIRM, db)

    decisions = feedback.load_decisions(db)
    assert len(decisions) == 2
    assert set(decisions["session_id"]) == {"s1", "s2"}


def test_revisiting_a_session_overwrites_not_duplicates(tmp_path):
    db = _db(tmp_path)
    feedback.record_decision("s1", "user_1", feedback.DISMISS, db)
    feedback.record_decision("s1", "user_1", feedback.CONFIRM, db)

    decisions = feedback.load_decisions(db)
    assert len(decisions) == 1
    assert decisions.iloc[0]["decision"] == feedback.CONFIRM


def test_dismissals_lower_and_confirmations_raise(tmp_path):
    db = _db(tmp_path)
    for i in range(3):
        feedback.record_decision(f"d{i}", "noisy_entity", feedback.DISMISS, db)
    feedback.record_decision("c0", "real_threat", feedback.CONFIRM, db)

    adjustments = feedback.entity_adjustments(feedback.load_decisions(db))
    assert adjustments["noisy_entity"] == -3 * feedback.DISMISS_STEP
    assert adjustments["real_threat"] == feedback.CONFIRM_STEP


def test_adjustment_is_capped(tmp_path):
    db = _db(tmp_path)
    for i in range(feedback.MAX_STEPS + 10):
        feedback.record_decision(f"d{i}", "very_noisy", feedback.DISMISS, db)

    adjustments = feedback.entity_adjustments(feedback.load_decisions(db))
    assert adjustments["very_noisy"] == -feedback.MAX_STEPS * feedback.DISMISS_STEP


def test_mixed_decisions_net_out(tmp_path):
    db = _db(tmp_path)
    feedback.record_decision("d0", "mixed", feedback.DISMISS, db)
    feedback.record_decision("d1", "mixed", feedback.DISMISS, db)
    feedback.record_decision("c0", "mixed", feedback.CONFIRM, db)

    adjustments = feedback.entity_adjustments(feedback.load_decisions(db))
    assert adjustments["mixed"] == feedback.CONFIRM_STEP - 2 * feedback.DISMISS_STEP


def test_apply_adjustments_reranks_and_stays_in_range():
    scored = pd.DataFrame({
        "session_id": ["s1", "s2", "s3"],
        "entity_id": ["noisy", "clean", "noisy"],
        "risk_score": [0.90, 0.88, 0.02],
    })
    adjusted = feedback.apply_adjustments(scored, {"noisy": -0.10})

    assert adjusted.loc[0, "effective_risk"] == 0.80
    assert adjusted.loc[1, "effective_risk"] == 0.88
    # the clean entity now outranks the dismissed one
    assert adjusted.sort_values("effective_risk", ascending=False).iloc[0]["entity_id"] == "clean"
    # and nothing escapes [0, 1]
    assert adjusted["effective_risk"].between(0, 1).all()


def test_unknown_decision_rejected(tmp_path):
    try:
        feedback.record_decision("s1", "e1", "maybe", _db(tmp_path))
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown decision")
