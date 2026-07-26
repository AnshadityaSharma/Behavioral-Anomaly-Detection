from pathlib import Path

import numpy as np
import pandas as pd

from src.dashboard import feedback


def _db(tmp_path):
    return tmp_path / "feedback.db"


def _queue(n=10_000):
    """A score distribution shaped like the real one: the top 1% packed into a
    narrow score band just above the cutoff, which is exactly the clustering
    that made the fixed score cap ineffective.

    Alerts run low->high, split into named blocks by queue position:
      ``marginal``  bottom of the queue, sitting on the cutoff
      ``midblock``  middle of the queue - score margin < 0.10 but percentile
                    margin > the percentile cap, so the two schemes disagree
      ``topblock``  top of the queue, far enough above the cutoff that even the
                    old score cap could not remove it (the dev_0025 case)
    """
    n_alerts = n // 100
    rng = np.random.default_rng(0)
    scores = np.concatenate([
        rng.uniform(0.0, 0.885, n - n_alerts),
        np.linspace(0.886, 1.0, n_alerts),      # queue: 0.114 wide, like the real one
    ])

    entity = [f"bulk{i % 50}" for i in range(n - n_alerts)]
    for i in range(n_alerts):
        frac = i / (n_alerts - 1)
        if frac < 0.10:
            entity.append("marginal")
        elif 0.45 <= frac < 0.55:
            entity.append("midblock")
        elif frac >= 0.90:
            entity.append("topblock")
        else:
            entity.append(f"filler{i}")

    return pd.DataFrame({
        "session_id": [f"s{i}" for i in range(n)],
        "entity_id": entity,
        "risk_score": scores,
    })


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


# --- percentile-rank capping -------------------------------------------------


def test_percentile_ranks_span_the_distribution():
    ranks = feedback.percentile_ranks([0.1, 0.5, 0.9, 0.3])
    assert ranks.min() == 0.0
    assert ranks.max() == 100.0
    # order is preserved
    assert list(np.argsort(ranks)) == list(np.argsort([0.1, 0.5, 0.9, 0.3]))


def test_zero_movement_round_trips_exactly():
    """Rank -> score has to be an exact identity, or every untouched entity
    drifts slightly the moment anyone triages anything."""
    scores = np.linspace(0.0, 1.0, 500) ** 2
    ranks = feedback.percentile_ranks(scores)
    np.testing.assert_allclose(np.quantile(scores, ranks / 100.0), scores)


def test_net_steps_are_uncapped():
    """The bound belongs on queue movement, not on how often an analyst clicks."""
    decisions = pd.DataFrame({
        "session_id": [f"s{i}" for i in range(20)],
        "entity_id": ["noisy"] * 20,
        "decision": [feedback.DISMISS] * 20,
        "decided_at": ["2026-01-01"] * 20,
    })
    assert feedback.entity_net_steps(decisions) == {"noisy": -20}


def _still_queued(frame, entity, cutoff):
    rows = frame[(frame["entity_id"] == entity) & (frame["risk_score"] >= cutoff)]
    return int((rows["effective_risk"] >= cutoff).sum())


def test_percentile_movement_is_capped_regardless_of_decision_count():
    scored = _queue()
    modest = feedback.apply_percentile_adjustments(scored, {"midblock": -5})
    extreme = feedback.apply_percentile_adjustments(scored, {"midblock": -500})

    for frame in (modest, extreme):
        moved = frame[frame["entity_id"] == "midblock"]["pct_movement"]
        assert moved.abs().max() <= feedback.PERCENTILE_CAP + 1e-9

    # 5 decisions already reaches the cap, so 500 buys nothing extra
    np.testing.assert_allclose(
        modest[modest["entity_id"] == "midblock"]["effective_risk"].values,
        extreme[extreme["entity_id"] == "midblock"]["effective_risk"].values,
    )


def test_percentile_cap_protects_a_mid_queue_entity_the_score_cap_does_not():
    """The case the change exists for: a mid-queue entity whose score margin is
    under 0.10 gets cleared out entirely by the old cap, and stays visible under
    the percentile cap because it is nowhere near the bottom of the queue."""
    scored = _queue()
    cutoff = scored["risk_score"].quantile(0.99)
    n_alerts = int(((scored["entity_id"] == "midblock")
                    & (scored["risk_score"] >= cutoff)).sum())
    assert n_alerts > 0

    old = feedback.apply_adjustments(scored, {"midblock": -5 * feedback.DISMISS_STEP})
    new = feedback.apply_percentile_adjustments(scored, {"midblock": -5})

    assert _still_queued(old, "midblock", cutoff) == 0
    assert _still_queued(new, "midblock", cutoff) == n_alerts


def test_top_of_queue_entity_is_bounded_under_both_schemes():
    """Honest counterpart: an entity far enough above the cutoff was already
    safe under the old cap, so the change buys nothing for it."""
    scored = _queue()
    cutoff = scored["risk_score"].quantile(0.99)
    n_alerts = int(((scored["entity_id"] == "topblock")
                    & (scored["risk_score"] >= cutoff)).sum())

    old = feedback.apply_adjustments(scored, {"topblock": -5 * feedback.DISMISS_STEP})
    new = feedback.apply_percentile_adjustments(scored, {"topblock": -5})

    assert _still_queued(old, "topblock", cutoff) == n_alerts
    assert _still_queued(new, "topblock", cutoff) == n_alerts


def test_percentile_cap_still_lets_marginal_alerts_be_dismissed_away():
    """The flip side: an entity sitting on the cutoff is still removable, so
    legitimate triage of borderline noise keeps working."""
    scored = _queue()
    cutoff = scored["risk_score"].quantile(0.99)
    new = feedback.apply_percentile_adjustments(scored, {"marginal": -5})

    rows = new[(new["entity_id"] == "marginal") & (new["risk_score"] >= cutoff)]
    assert len(rows) > 0
    assert (rows["effective_risk"] < cutoff).any()


def test_percentile_confirmations_raise_and_dismissals_lower():
    scored = _queue()
    adjusted = feedback.apply_percentile_adjustments(
        scored, {"midblock": -5, "marginal": 5}
    )

    assert (adjusted[adjusted["entity_id"] == "midblock"]["pct_movement"] <= 0).all()
    assert (adjusted[adjusted["entity_id"] == "marginal"]["pct_movement"] >= 0).all()
    # untouched entities do not move at all
    assert (adjusted[adjusted["entity_id"] == "bulk0"]["pct_movement"] == 0).all()


def test_percentile_edges_clip_at_the_distribution_boundary():
    """Entities already at the top or bottom cannot be pushed off the scale."""
    scored = _queue()
    top_entity = scored.loc[scored["risk_score"].idxmax(), "entity_id"]
    bottom_entity = scored.loc[scored["risk_score"].idxmin(), "entity_id"]

    adjusted = feedback.apply_percentile_adjustments(
        scored, {top_entity: 500, bottom_entity: -500}
    )
    assert adjusted["effective_pct_rank"].between(0, 100).all()
    assert adjusted["effective_risk"].between(0, 1).all()


def test_apply_feedback_dispatches_and_rejects_unknown_mode():
    scored = _queue()
    decisions = pd.DataFrame({
        "session_id": ["s0"], "entity_id": ["e0"],
        "decision": [feedback.DISMISS], "decided_at": ["2026-01-01"],
    })

    for mode in (feedback.CAP_MODE_SCORE, feedback.CAP_MODE_PERCENTILE):
        out = feedback.apply_feedback(scored, decisions, mode=mode)
        assert {"risk_adjustment", "effective_risk"} <= set(out.columns)

    try:
        feedback.apply_feedback(scored, decisions, mode="sideways")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown cap mode")


def test_apply_feedback_with_no_decisions_leaves_the_queue_alone():
    scored = _queue()
    empty = feedback.load_decisions(Path("does-not-exist.db"))
    out = feedback.apply_feedback(scored, empty)
    np.testing.assert_allclose(out["effective_risk"].values,
                               out["risk_score"].values)
