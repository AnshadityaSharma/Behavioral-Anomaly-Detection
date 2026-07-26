"""Analyst triage decisions and the per-entity adjustment they produce.

The adjustment is deliberately dumb: a fixed step per decision, capped, applied
only to alert ranking. It is not model retraining and does not change any
stored risk score - it just pushes entities an analyst keeps dismissing further
down the queue, and pulls confirmed ones up.

Two capping modes exist:

- ``score`` (the original) caps the raw score offset at +/-0.10. This turned out
  to bound almost nothing: risk scores cluster tightly just above the alert
  cutoff, so 0.10 of score is wider than the cutoff margin for 96% of queued
  entities and a handful of dismissals can clear nearly any entity out of the
  queue entirely.
- ``percentile`` (the default) caps how far an entity can move in *percentile
  rank* instead. Because the top-1% queue spans exactly one percentile point, a
  cap of N points means precisely "an entity can be moved through N% of the
  queue" - so the cap value states what it actually enforces, independent of how
  tightly scores happen to be clustered.

Both are kept so the report can compare them on the same data.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "analyst_feedback.db"

DISMISS_STEP = 0.02
CONFIRM_STEP = 0.02
MAX_STEPS = 5

# Percentile-rank capping. PERCENTILE_STEP * MAX_STEPS == PERCENTILE_CAP, so a
# decision still "counts" the same way and five of them still max out the
# adjustment - only the ceiling is expressed in queue movement rather than score.
PERCENTILE_STEP = 0.05
PERCENTILE_CAP = 0.25

CAP_MODE_SCORE = "score"
CAP_MODE_PERCENTILE = "percentile"
DEFAULT_CAP_MODE = CAP_MODE_PERCENTILE

CONFIRM = "confirm"
DISMISS = "dismiss"


def connect(db_path=DB_PATH):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            session_id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            decided_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def record_decision(session_id, entity_id, decision, db_path=DB_PATH):
    if decision not in (CONFIRM, DISMISS):
        raise ValueError(f"unknown decision: {decision}")

    conn = connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO decisions (session_id, entity_id, decision, decided_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                decision = excluded.decision,
                decided_at = excluded.decided_at
            """,
            (session_id, entity_id, decision, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def load_decisions(db_path=DB_PATH):
    if not Path(db_path).exists():
        return pd.DataFrame(columns=["session_id", "entity_id", "decision", "decided_at"])
    conn = connect(db_path)
    try:
        return pd.read_sql_query("SELECT * FROM decisions", conn)
    finally:
        conn.close()


def entity_adjustments(decisions):
    """Net score offset per entity, negative meaning "rank this lower"."""
    if decisions.empty:
        return {}

    counts = decisions.groupby(["entity_id", "decision"]).size().unstack(fill_value=0)

    adjustments = {}
    for entity_id, row in counts.iterrows():
        offset = (
            CONFIRM_STEP * min(int(row.get(CONFIRM, 0)), MAX_STEPS)
            - DISMISS_STEP * min(int(row.get(DISMISS, 0)), MAX_STEPS)
        )
        if offset:
            adjustments[entity_id] = offset
    return adjustments


def apply_adjustments(scored, adjustments):
    """Legacy score-space capping. Kept for comparison against percentile mode."""
    result = scored.copy()
    result["risk_adjustment"] = result["entity_id"].map(adjustments).fillna(0.0)
    result["effective_risk"] = (result["risk_score"] + result["risk_adjustment"]).clip(0, 1)
    return result


def entity_net_steps(decisions):
    """Net decision count per entity, positive meaning "rank this higher".

    Deliberately uncapped - in percentile mode the bound is applied to queue
    movement, not to how many times an analyst is allowed to click.
    """
    if decisions.empty:
        return {}

    counts = decisions.groupby(["entity_id", "decision"]).size().unstack(fill_value=0)

    steps = {}
    for entity_id, row in counts.iterrows():
        net = int(row.get(CONFIRM, 0)) - int(row.get(DISMISS, 0))
        if net:
            steps[entity_id] = net
    return steps


def percentile_ranks(scores):
    """Percentile rank (0-100) of each score within its own distribution.

    Normalised by ``n - 1`` rather than ``n`` so that feeding a rank straight
    back through ``np.quantile`` returns the original score exactly - i.e. zero
    movement is genuinely a no-op rather than a small resampling nudge.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.zeros(1)
    ranks = np.empty(n)
    ranks[np.argsort(scores, kind="mergesort")] = np.arange(n)
    return ranks / (n - 1) * 100.0


def apply_percentile_adjustments(scored, net_steps, cap=PERCENTILE_CAP,
                                 step=PERCENTILE_STEP):
    """Move each entity's alerts by at most ``cap`` percentile ranks.

    The requested movement is ``net_steps * step`` percentile points, clipped to
    +/-cap however many decisions were recorded, then mapped back through the
    empirical score distribution so the queue can still be ranked on a score.
    """
    result = scored.copy()
    scores = result["risk_score"].to_numpy(dtype=float)

    pct = percentile_ranks(scores)
    requested = result["entity_id"].map(net_steps).fillna(0.0).to_numpy(dtype=float) * step
    movement = np.clip(requested, -cap, cap)
    target_pct = np.clip(pct + movement, 0.0, 100.0)

    effective = np.quantile(scores, target_pct / 100.0)

    result["pct_rank"] = pct
    result["pct_movement"] = target_pct - pct
    result["effective_pct_rank"] = target_pct
    result["effective_risk"] = np.clip(effective, 0.0, 1.0)
    result["risk_adjustment"] = result["effective_risk"] - result["risk_score"]
    return result


def apply_feedback(scored, decisions, mode=DEFAULT_CAP_MODE, cap=None):
    """Apply analyst decisions to the queue under the chosen capping mode.

    Both modes return ``risk_adjustment`` and ``effective_risk`` so callers do
    not need to care which one is active.
    """
    if mode == CAP_MODE_SCORE:
        return apply_adjustments(scored, entity_adjustments(decisions))
    if mode == CAP_MODE_PERCENTILE:
        return apply_percentile_adjustments(
            scored, entity_net_steps(decisions),
            cap=PERCENTILE_CAP if cap is None else cap,
        )
    raise ValueError(f"unknown cap mode: {mode}")
