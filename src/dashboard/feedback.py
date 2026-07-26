"""Analyst triage decisions and the per-entity score offset they produce.

The offset is deliberately dumb: a fixed step per dismissal, capped, applied
only to alert ranking. It is not model retraining and does not change any
stored risk score - it just pushes entities an analyst keeps dismissing further
down the queue, and pulls confirmed ones up.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "analyst_feedback.db"

DISMISS_STEP = 0.02
CONFIRM_STEP = 0.02
MAX_STEPS = 5

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
    result = scored.copy()
    result["risk_adjustment"] = result["entity_id"].map(adjustments).fillna(0.0)
    result["effective_risk"] = (result["risk_score"] + result["risk_adjustment"]).clip(0, 1)
    return result
