# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""FEP-0012 Phase 6: the decision → outcome reward junction.

``log_decision`` (``chain.py``) records every decision with a correlation
spine (``session_id``/``turn_id``/``decision_id``) to the JSONL log. This
module stamps the **outcome** side: once a session/task resolves (pass/fail),
:func:`record_session_outcome` writes one ``decision_outcome`` row per
decision logged under that session, carrying the reward.

The junction is what lets the trainer do **reward-supervised** (not
imitation) learning — :mod:`victor.ml.outcome_training` JOINs the JSONL
decisions to these outcome rows by ``decision_id``.

Design notes
------------
- Decisions live in the JSONL log (``log_decision``'s fast-append path); the
  SQL ``decision_log`` table is not yet written. This reader reads the JSONL
  directly (same source as ``victor.evaluation.manifest``) and writes outcomes
  to the durable SQL ``decision_outcome`` table — so outcomes accumulate and
  survive JSONL rotation even though the full decision text stays in JSONL.
- v1 credit is **uniform** (``credit_method="session_uniform"``): every
  decision in the session gets ``attributed_reward``. The schema reserves
  ``credit_method``/``segment_rewards`` for GAE/recency-weighted credit later.
- Recording is **idempotent**: it DELETEs existing rows for the session first,
  so re-recording (or a resumed run) replaces rather than duplicates.
- Best-effort: like ``log_decision``, it never raises — a DB problem must not
  break the benchmark/session.
"""

from __future__ import annotations

from victor.core.json_utils import json_loads
from json import JSONDecodeError
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# JSONL decision log written by victor.agent.decisions.chain.log_decision.
_DECISIONS_LOG = Path.home() / ".victor" / "logs" / "decisions.jsonl"


def _decisions_for_session(session_id: str) -> list[dict[str, Any]]:
    """Read decisions logged under ``session_id`` from the JSONL log.

    Returns the raw records (each has ``decision_id``/``turn_id``/``type``/…).
    Robust to a missing log file and malformed lines.
    """
    if not session_id or not _DECISIONS_LOG.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with _DECISIONS_LOG.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json_loads(line)
                except JSONDecodeError:
                    continue
                if rec.get("session_id") == session_id:
                    out.append(rec)
    except OSError as exc:
        logger.warning("record_session_outcome: could not read decisions log: %s", exc)
    return out


def record_session_outcome(
    session_id: str,
    *,
    success: bool,
    quality_score: float,
    attributed_reward: Optional[float] = None,
    credit_method: str = "session_uniform",
) -> int:
    """Stamp ``decision_outcome`` rows for every decision logged under a session.

    For each decision in the session (read from the JSONL log), writes one
    outcome row joining ``decision_id`` → ``success``/``quality_score``/
    ``attributed_reward``. v1 credit is uniform: every decision gets the same
    reward (the session/task outcome). Idempotent — existing rows for the
    session are replaced.

    Args:
        session_id: The correlation-spine session_id decisions were logged under.
        success: Did the session/task succeed (e.g. all tests passed)?
        quality_score: Outcome quality in ``[0, 1]`` (e.g. test pass rate).
        attributed_reward: Per-decision credit. Defaults to ``quality_score``.
        credit_method: How credit was assigned (default ``session_uniform``).

    Returns:
        Number of outcome rows written (0 if no decisions exist for the session).
    """
    if not session_id:
        return 0
    try:
        from victor.core.database import get_database
        from victor.core.schema import Tables

        decisions = _decisions_for_session(session_id)
        if not decisions:
            logger.debug("record_session_outcome: no decisions for session %s", session_id)
            return 0

        reward = quality_score if attributed_reward is None else attributed_reward
        success_int = 1 if success else 0
        db = get_database()

        # Idempotent: replace any prior outcome rows for this session.
        db.execute(
            f"DELETE FROM {Tables.DECISION_OUTCOME} WHERE session_id = ?",
            (session_id,),
        )

        rows = [
            (
                rec.get("decision_id", ""),
                session_id,
                rec.get("turn_id", ""),
                success_int,
                float(quality_score),
                float(reward),
                credit_method,
            )
            for rec in decisions
            if rec.get("decision_id")
        ]
        if rows:
            db.executemany(
                f"""
                INSERT INTO {Tables.DECISION_OUTCOME} (
                    decision_id, session_id, turn_id, success,
                    quality_score, attributed_reward, credit_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        logger.info(
            "record_session_outcome: wrote %d outcome rows for session %s "
            "(success=%s, reward=%.3f, method=%s)",
            len(rows),
            session_id,
            success,
            reward,
            credit_method,
        )

        # FEP-0012 Phase 6: reward-weight-update the per-project classifier
        # delta from this session's decisions (local-only, gated by the
        # local_learning_enabled setting, best-effort — never raises). The
        # decisions list was already read above; pass it through to avoid a
        # second JSONL pass.
        try:
            from victor.agent.decisions.local_delta import update_delta_from_session

            update_delta_from_session(session_id, reward=reward, decisions=decisions)
        except Exception as exc:  # defensive — the delta must never break this path
            logger.debug("local_classifier_delta update skipped: %s", exc)

        return len(rows)
    except Exception as exc:  # never break the benchmark/session
        logger.warning("record_session_outcome failed for %s: %s", session_id, exc)
        return 0
