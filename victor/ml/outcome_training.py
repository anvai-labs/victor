# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Reward-supervised training from the ``decision_outcome`` junction.

The **production** training source for the edge classifier (vs. the offline
:mod:`victor.ml.mining` path that reads a benchmark manifest). This module
JOINs the JSONL decision log (the decision *text*) to the durable SQL
``decision_outcome`` table (the decision *reward*) by ``decision_id`` and
projects the result into reward-labeled ``(text, label)`` samples for
:func:`victor.ml.trainer.train_model`.

This is the swap the trainer's docstring anticipates: supervision switches
from caller-supplied (imitation) labels to ``decision_outcome.attributed_reward``
(reward) once that junction is populated by
:func:`victor.agent.decisions.outcome.record_session_outcome`.

Label bucketing
---------------
``attributed_reward`` (the test pass-rate-style credit) → categorical label:
``pass`` (==1.0) / ``partial`` (0<r<1) / ``fail`` (0). A head trains only when
a decision type has ≥2 distinct buckets, so the all-fail case trains nothing
until the pass rate improves.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

from victor.ml.mining import MINED_DECISION_TYPES, _decision_text

logger = logging.getLogger(__name__)

_DECISIONS_LOG = Path.home() / ".victor" / "logs" / "decisions.jsonl"


def _reward_label(reward: float) -> str:
    """Bucket an attributed reward into a categorical label."""
    if reward >= 1.0:
        return "pass"
    if reward > 0.0:
        return "partial"
    return "fail"


def _load_outcome_rewards() -> dict[str, float]:
    """Read ``{decision_id: attributed_reward}`` from the SQL junction."""
    rewards: dict[str, float] = {}
    try:
        from victor.core.database import get_database
        from victor.core.schema import Tables

        db = get_database()
        # Table name is the hardcoded Tables.DECISION_OUTCOME constant, not
        # user input (identifiers cannot be bound as SQL parameters).
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query, python.lang.security.audit.formatted-sql-query.formatted-sql-query
        cursor = db.execute(  # nosemgrep
            f"SELECT decision_id, attributed_reward FROM {Tables.DECISION_OUTCOME}"
        )
        for decision_id, reward in cursor.fetchall():
            if decision_id:
                rewards[decision_id] = float(reward) if reward is not None else 0.0
    except Exception as exc:
        logger.warning("load_outcome_samples: could not read decision_outcome: %s", exc)
    return rewards


def load_outcome_samples(
    decision_types: Optional[Iterable[str]] = None,
) -> dict[str, list[tuple[str, str]]]:
    """JOIN JSONL decisions ⋈ SQL outcomes → ``{decision_type: [(text, label)]}``.

    Args:
        decision_types: Restrict to these types. Defaults to
            :data:`victor.ml.mining.MINED_DECISION_TYPES`.

    Returns:
        Per-decision-type ``(text, label)`` samples where ``label`` is the
        reward bucket. Decisions without an outcome row are skipped (no reward
        signal yet).
    """
    types = set(decision_types) if decision_types else set(MINED_DECISION_TYPES)
    rewards = _load_outcome_rewards()
    if not rewards or not _DECISIONS_LOG.exists():
        return {}

    per_type: dict[str, list[tuple[str, str]]] = {}
    try:
        with _DECISIONS_LOG.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                dtype = rec.get("type") or rec.get("decision_type")
                if dtype not in types:
                    continue
                decision_id = rec.get("decision_id")
                if not decision_id or decision_id not in rewards:
                    continue  # no outcome stamped yet
                per_type.setdefault(dtype, []).append(
                    (_decision_text(rec), _reward_label(rewards[decision_id]))
                )
    except OSError as exc:
        logger.warning("load_outcome_samples: could not read decisions log: %s", exc)
    return per_type


def train_from_outcomes(
    output_model: Optional[str | Path] = None,
    *,
    decision_types: Optional[Iterable[str]] = None,
    model_version: str = "outcomes-1",
    threshold: float = 0.6,
) -> Optional[Any]:
    """Train a classifier artifact from the decision_outcome junction.

    Mines the JOIN of decisions ⋈ outcomes and trains a head per decision type
    that has ≥2 distinct reward buckets. Returns the trained
    :class:`~victor.ml.model.EdgeClassifierModel`, or ``None`` if nothing could
    train (no outcomes populated, or insufficient label diversity).
    """
    per_type = load_outcome_samples(decision_types)
    if not per_type:
        logger.warning(
            "No reward-labeled samples (is decision_outcome populated? " "run a benchmark first)"
        )
        return None

    try:
        from victor.ml.trainer import train_head
        from victor.ml.model import EdgeClassifierModel
    except ImportError as exc:
        logger.error("Training needs the [ml] extra (scikit-learn/scipy): %s", exc)
        return None

    heads = {}
    for dtype, samples in per_type.items():
        distinct = {label for _, label in samples}
        if len(distinct) < 2:
            logger.info(
                "Skipping head %s: only %d distinct label(s) %s across %d samples",
                dtype,
                len(distinct),
                sorted(distinct),
                len(samples),
            )
            continue
        try:
            heads[dtype] = train_head(dtype, samples, threshold=threshold)
            logger.info(
                "Trained head %s on %d reward-supervised samples (labels=%s)",
                dtype,
                len(samples),
                sorted(distinct),
            )
        except ValueError as exc:
            logger.info("Skipping head %s: %s", dtype, exc)

    if not heads:
        logger.warning(
            "No heads trained (need ≥1 decision type with ≥2 distinct reward "
            "buckets — run more tasks until some pass)"
        )
        return None

    model = EdgeClassifierModel(heads=heads, model_version=model_version)
    if output_model is not None:
        output_model = Path(output_model)
        output_model.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output_model))
        logger.info("Saved reward-supervised classifier to %s", output_model)
    return model
