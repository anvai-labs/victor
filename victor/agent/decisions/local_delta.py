# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""FEP-0012 Phase 6: per-project online RL personalization delta.

The shipped edge classifier is a *universal* baseline trained on aggregate
decision/outcome data. This module learns a small per-project **reward-weighted
overlay** on top of it, stored in the project DB's ``local_classifier_delta``
table and blended into the head's logits at predict time
(``score = bias + W·x + α·(delta·x)``, α from the model).

Consistency with the offline trainer
-------------------------------------
The head labels ARE reward buckets (e.g. ``task_completion`` →
``fail/partial/pass``; see :mod:`victor.ml.outcome_training`), so the online
delta is a continuation of the same reward-supervised objective: for each
decision in a resolved session, the observed reward bucket ``b`` is the target,
and we take a softmax-cross-entropy step toward it:

    ΔW[h, k] = lr · x_h · (δ_{k,b} − p_k)

where ``p = softmax(bias + W·x)`` comes from the **universal** model. Using the
universal (not blended) prediction makes the gradient proportional to where the
universal model is *wrong* — exactly the project-specific error the delta exists
to correct — and keeps the updater free of a delta self-dependency.

Properties
----------
- **Per-label & per-hash**: one row per ``(decision_type, feature_hash, label)``
  so a multi-class head carries a per-label nudge.
- **Bounded & L2-decayed**: after each update, all weights are multiplied by
  ``decay`` (default 0.995) and trimmed to ``top_k`` rows per
  ``(decision_type, label)`` by ``|weight|`` — so the universal model
  re-asserts and the overlay never grows unbounded.
- **Local-only, never uploaded**: lives in ``./.victor/project.db``; nothing
  leaves the machine.
- **Best-effort**: a project-DB problem (e.g. "database is locked") is logged
  and skipped — it must never break a benchmark/session.

The update is triggered from :func:`victor.agent.decisions.outcome.record_session_outcome`
on each resolved session/task. The read side is wired in
:class:`victor.agent.services.local_classifier_service.LocalClassifierDecisionService`
via :func:`load_delta`.
"""

from __future__ import annotations

import logging
import sqlite3
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from victor.ml.features import FEATURE_SPEC_VERSION, extract_features
from victor.ml.mining import _decision_text
from victor.ml.outcome_training import _reward_label

logger = logging.getLogger(__name__)

# Reuse the same JSONL reader as the outcome junction so a session's decisions
# are read once (record_session_outcome passes the already-read list in).
from victor.agent.decisions.outcome import _decisions_for_session  # noqa: E402

# A predict callable: (decision_type, features) -> softmax probabilities over
# the head's labels (shape ``[len(labels)]``), or ``None`` to skip that
# decision (no head / un-featurizable). The default implementation scores
# features with the universal artifact; tests inject a stub.
PredictFn = Callable[[str, Dict[int, float]], Optional[np.ndarray]]

# Cached universal model + derived (predict_fn, head_labels). ``_model_cache``
# is either an EdgeClassifierModel, ``None`` (not yet tried), or ``False``
# (tried and unavailable — don't retry every call).
_model_cache: Any = None
_predict_cache: Optional[Tuple[PredictFn, Dict[str, List[str]]]] = None

# Idempotency: sessions whose delta has already been applied this process. Guards
# against by-design double-records — e.g. the benchmark records each task in the
# per-task on_progress callback AND again in the post-run safety-net loop (the
# outcome rows are idempotent via DELETE-then-INSERT, but the SGD update is not).
# Process-local + capped; a cross-restart duplicate stays bounded by L2 decay +
# top-K. ``dict`` preserves insertion order (3.7+), so it doubles as an ordered
# set with FIFO eviction.
_PROCESSED_SESSIONS: Dict[str, None] = {}
_PROCESSED_SESSIONS_CAP = 4096


def _claim_delta_session(session_id: str) -> bool:
    """Atomically claim a session for delta processing.

    Returns ``True`` iff this session was NOT already claimed this process
    (i.e. this is the first application and the caller should proceed). A repeat
    call for the same session returns ``False`` so the SGD update isn't applied
    twice for one session's outcome.
    """
    if session_id in _PROCESSED_SESSIONS:
        return False
    _PROCESSED_SESSIONS[session_id] = None
    if len(_PROCESSED_SESSIONS) > _PROCESSED_SESSIONS_CAP:
        # Drop the oldest quarter (FIFO) to bound memory in long-running servers.
        for stale in list(_PROCESSED_SESSIONS)[: _PROCESSED_SESSIONS_CAP // 4]:
            _PROCESSED_SESSIONS.pop(stale, None)
    return True


def _artifact_path() -> Optional[Path]:
    """Resolve the shipped-classifier artifact path (env override wins)."""
    env = os.environ.get("VICTOR_EDGE_CLASSIFIER_PATH")
    if env:
        return Path(env)
    try:
        import victor

        return Path(victor.__file__).resolve().parent / "models" / "edge_classifier_v1.npz"
    except Exception:
        return None


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (mirrors ``victor.ml.model._softmax``)."""
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _get_default_predict() -> Tuple[Optional[PredictFn], Dict[str, List[str]]]:
    """Load the universal artifact once and return (predict_fn, head_labels).

    Returns ``(None, {})`` if no healthy artifact is available — callers skip
    the update in that case (degrade gracefully, same as no-delta today).
    """
    global _model_cache, _predict_cache
    if _predict_cache is not None:
        return _predict_cache
    path = _artifact_path()
    if path is None or not path.exists():
        _predict_cache = (None, {})
        return _predict_cache
    try:
        from victor.ml.model import EdgeClassifierModel

        _model_cache = EdgeClassifierModel.load(str(path))
    except Exception as exc:
        logger.debug("delta updater: artifact load failed (%s): %s", path, exc)
        _predict_cache = (None, {})
        return _predict_cache

    model = _model_cache
    head_labels = {name: list(head.labels) for name, head in model.heads.items()}

    def _predict(dtype: str, features: Dict[int, float]) -> Optional[np.ndarray]:
        head = model.heads.get(dtype)
        if head is None or not features:
            return None
        scores = head.bias.astype(float).copy()
        weights = head.weights
        for h, val in features.items():
            row = weights.get(int(h))
            if row is not None:
                scores += row * val
        return _softmax(scores)

    _predict_cache = (_predict, head_labels)
    return _predict_cache


def _reset_predict_cache_for_tests() -> None:
    """Clear the cached artifact/predictor (test isolation)."""
    global _model_cache, _predict_cache
    _model_cache = None
    _predict_cache = None


def _compute_label_updates(
    decisions: List[Dict[str, Any]],
    reward: float,
    head_labels: Dict[str, List[str]],
    predict_fn: PredictFn,
    lr: float,
) -> Tuple[Dict[Tuple[str, int, str], float], Dict[Tuple[str, int], int]]:
    """Pure: compute per-(decision_type, hash, label) weight deltas for a session.

    Args:
        decisions: Raw JSONL decision records (each has ``type``/``input``/…).
        reward: The session/task reward in ``[0, 1]``.
        head_labels: ``{decision_type: [label, ...]}`` for heads to update.
        predict_fn: Universal-model softmax over labels (injectable for tests).
        lr: Step size.

    Returns:
        ``(acc, touched)`` where ``acc`` maps ``(dtype, hash, label) -> Δweight``
        and ``touched`` maps ``(dtype, hash) -> #decisions`` (for ``samples``).
    """
    bucket = _reward_label(reward)
    acc: Dict[Tuple[str, int, str], float] = {}
    touched: Dict[Tuple[str, int], int] = {}
    for rec in decisions:
        dtype = rec.get("type") or rec.get("decision_type")
        if not dtype or dtype not in head_labels:
            continue
        labels = head_labels[dtype]
        if bucket not in labels:
            continue
        b_idx = labels.index(bucket)
        text = _decision_text(rec)
        features = extract_features(text)
        if not features:
            continue
        probs = predict_fn(dtype, features)
        if probs is None:
            continue
        k = len(labels)
        if len(probs) != k:
            continue
        for h, xh in features.items():
            key = (dtype, int(h))
            touched[key] = touched.get(key, 0) + 1
            for j in range(k):
                grad = (1.0 if j == b_idx else 0.0) - float(probs[j])
                delta_w = lr * xh * grad
                rkey = (dtype, int(h), labels[j])
                acc[rkey] = acc.get(rkey, 0.0) + delta_w
    return acc, touched


def _persist_delta(
    acc: Dict[Tuple[str, int, str], float],
    reward: float,
    touched: Dict[Tuple[str, int], int],
    top_k: int,
    decay: float,
) -> int:
    """Upsert accumulated weights, then L2-decay + top-K trim. Best-effort."""
    try:
        from victor.core.database import get_project_database
        from victor.core.schema import Tables

        db = get_project_database()
        spec = FEATURE_SPEC_VERSION
        tbl = Tables.LOCAL_CLASSIFIER_DELTA
        # Table/identifier is a hardcoded schema constant, not user input.
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
        upsert_sql = f"""
            INSERT INTO {tbl} (
                decision_type, feature_hash, label, weight,
                samples, sum_reward, feature_spec_version, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(decision_type, feature_hash, label, feature_spec_version)
            DO UPDATE SET
                weight = weight + excluded.weight,
                samples = samples + excluded.samples,
                sum_reward = sum_reward + excluded.sum_reward,
                updated_at = datetime('now')
        """
        rows = [
            (
                dtype,
                h,
                label,
                float(w),
                touched.get((dtype, h), 1),
                float(reward) * touched.get((dtype, h), 1),
                spec,
            )
            for (dtype, h, label), w in acc.items()
        ]
        # ProjectDatabaseManager exposes execute() but not executemany(); the
        # row count is bounded by per-session feature cardinality, so a
        # parameterized loop is fine.
        for row in rows:
            db.execute(upsert_sql, row)  # nosemgrep
        # L2 decay across the whole overlay so the universal model re-asserts.
        db.execute(f"UPDATE {tbl} SET weight = weight * ?", (float(decay),))  # nosemgrep
        # Top-K bound per (decision_type, label, spec) by |weight|.
        db.execute(  # nosemgrep
            f"""
            DELETE FROM {tbl} WHERE rowid IN (
                SELECT rowid FROM (
                    SELECT rowid,
                           ROW_NUMBER() OVER (
                               PARTITION BY decision_type, label, feature_spec_version
                               ORDER BY ABS(weight) DESC, rowid
                           ) AS rn
                    FROM {tbl}
                ) WHERE rn > ?
            )
            """,
            (int(top_k),),
        )
        return len(rows)
    except sqlite3.OperationalError as exc:
        # "database is locked" under concurrent writers — skip, never raise.
        logger.debug("delta persist skipped (db locked?): %s", exc)
        return 0
    except Exception as exc:  # never break the benchmark/session
        logger.warning("delta persist failed: %s", exc)
        return 0


def update_delta_from_session(
    session_id: str,
    *,
    reward: float,
    decisions: Optional[List[Dict[str, Any]]] = None,
    predict_fn: Optional[PredictFn] = None,
    head_labels: Optional[Dict[str, List[str]]] = None,
    lr: Optional[float] = None,
    top_k: Optional[int] = None,
    decay: Optional[float] = None,
) -> int:
    """Reward-weight-update the project delta from one session's decisions.

    Called on session/task resolution (from ``record_session_outcome``). Reads
    the session's decisions (or accepts a pre-read ``decisions`` list to avoid a
    second JSONL pass), restricts to decision types that have an artifact head,
    and writes per-label cross-entropy updates to ``local_classifier_delta``.

    The setting gate (``DecisionServiceSettings.local_learning_enabled``) is
    consulted on the production path. Injecting ``predict_fn`` selects **test
    mode**: the setting gate is bypassed and the universal artifact is not
    loaded (the caller controls the prediction).

    Args:
        session_id: Correlation-spine session_id decisions were logged under.
        reward: Session/task reward in ``[0, 1]`` (e.g. test pass rate).
        decisions: Pre-read decision records (else read from the JSONL log).
        predict_fn: Injectable universal-softmax (test mode).
        head_labels: ``{decision_type: [label, ...]}`` (required with predict_fn).
        lr, top_k, decay: Optional overrides (default from settings).

    Returns:
        Number of (dtype, hash, label) rows upserted (0 = nothing written).
    """
    test_mode = predict_fn is not None

    if not test_mode:
        settings: Any = None
        try:
            from victor.config.decision_settings import DecisionServiceSettings

            settings = DecisionServiceSettings()
        except Exception:
            settings = None
        if settings is not None and not settings.local_learning_enabled:
            return 0
        lr = lr if lr is not None else getattr(settings, "local_learning_lr", 0.1)
        top_k = top_k if top_k is not None else getattr(settings, "local_learning_top_k", 2000)
        decay = decay if decay is not None else getattr(settings, "local_learning_decay", 0.995)
    else:
        lr = 0.1 if lr is None else lr
        top_k = 2000 if top_k is None else top_k
        decay = 0.995 if decay is None else decay

    if decisions is None:
        decisions = _decisions_for_session(session_id)
    if not decisions:
        return 0

    if predict_fn is None or head_labels is None:
        default_fn, default_labels = _get_default_predict()
        if predict_fn is None:
            predict_fn = default_fn
        if head_labels is None:
            head_labels = default_labels
    if predict_fn is None or not head_labels:
        logger.debug("delta update: no usable artifact heads for session %s", session_id)
        return 0

    # Idempotency (production path only): skip sessions already applied this
    # process so a double-recorded outcome (e.g. benchmark on_progress + the
    # post-run safety-net loop) doesn't double-count the SGD update. Test mode
    # (injected predict_fn) bypasses this so tests can replay a session.
    if not test_mode and not _claim_delta_session(session_id):
        logger.debug("delta update: session %s already applied; skipping", session_id)
        return 0

    acc, touched = _compute_label_updates(decisions, reward, head_labels, predict_fn, lr)
    if not acc:
        return 0
    return _persist_delta(acc, reward, touched, top_k, decay)


def load_delta(decision_type: str, labels: List[str]) -> Dict[int, np.ndarray]:
    """Load the project delta for one decision type as per-label vectors.

    Args:
        decision_type: The head key (e.g. ``"task_completion"``).
        labels: The head's label ordering — returned vectors are indexed to
            match this order (missing label → 0.0). Must match the head the
            caller will blend into.

    Returns:
        ``{feature_hash: np.ndarray([w_per_label in `labels` order])}``.
        Empty dict on any error or when no rows exist (→ no blend, graceful).
    """
    try:
        from victor.core.database import get_project_database
        from victor.core.schema import Tables

        db = get_project_database()
        spec = FEATURE_SPEC_VERSION
        # Hardcoded schema constants, not user input.
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
        cursor = db.execute(  # nosemgrep
            f"SELECT feature_hash, label, weight FROM {Tables.LOCAL_CLASSIFIER_DELTA} "
            f"WHERE decision_type = ? AND feature_spec_version = ?",
            (decision_type, spec),
        )
        rows = cursor.fetchall()
    except Exception as exc:
        logger.debug("load_delta failed for %s: %s", decision_type, exc)
        return {}

    idx = {lab: i for i, lab in enumerate(labels)}
    out: Dict[int, np.ndarray] = {}
    for h, label, weight in rows:
        if label not in idx:
            continue
        h = int(h)
        if h not in out:
            out[h] = np.zeros(len(labels), dtype=float)
        out[h][idx[label]] += float(weight)
    return out


def clear_delta_for_tests() -> None:
    """Delete all delta rows + reset the claimed-sessions guard (test isolation)."""
    try:
        from victor.core.database import get_project_database
        from victor.core.schema import Tables

        db = get_project_database()
        db.execute(f"DELETE FROM {Tables.LOCAL_CLASSIFIER_DELTA}")  # nosemgrep
    except Exception:
        pass
    _PROCESSED_SESSIONS.clear()
