# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""FEP-0012 Phase 6: per-project online RL delta (victor.agent.decisions.local_delta).

Covers the pure cross-entropy update (sign + magnitude scaling), per-label
vector assembly + label-order alignment, the feature_spec_version guard,
top-K bounding, L2 decay, and best-effort/setting-gated behavior. The update
path is exercised with an injected ``predict_fn`` + ``head_labels`` so tests are
deterministic and independent of the shipped artifact.
"""

from __future__ import annotations

import numpy as np
import pytest

from victor.agent.decisions import local_delta as ld
from victor.core.schema import Tables

LABELS = ["fail", "partial", "pass"]
HEAD_LABELS = {"task_completion": LABELS}


@pytest.fixture(autouse=True)
def _clean_delta_table():
    """Start each test with an empty delta table in the isolated project DB."""
    ld.clear_delta_for_tests()
    yield
    ld.clear_delta_for_tests()


def _delta_rows(db):
    return db.execute(
        f"SELECT decision_type, feature_hash, label, weight, samples "
        f"FROM {Tables.LOCAL_CLASSIFIER_DELTA}"
    ).fetchall()


# --------------------------------------------------------------------------
# Pure update math
# --------------------------------------------------------------------------


def _decisions(n=1):
    return [{"type": "task_completion", "input": "anything", "session_id": "s"} for _ in range(n)]


def test_compute_update_sign_toward_observed_bucket(monkeypatch):
    """Observed pass -> pass weight positive, fail/partial negative."""
    monkeypatch.setattr(ld, "extract_features", lambda txt: {10: 1.0})

    def pfn(dtype, features):  # model leans fail
        return np.array([0.6, 0.3, 0.1])

    acc, _touched = ld._compute_label_updates(_decisions(), 1.0, HEAD_LABELS, pfn, lr=0.5)
    assert acc[("task_completion", 10, "pass")] > 0
    assert acc[("task_completion", 10, "fail")] < 0
    assert acc[("task_completion", 10, "partial")] < 0


def test_compute_update_magnitude_scales_with_model_error(monkeypatch):
    """Gradient is largest where the universal model is confidently wrong."""
    monkeypatch.setattr(ld, "extract_features", lambda txt: {10: 1.0})

    # Model confidently predicts pass (p_pass=0.95) but we observe FAIL.
    pfn_wrong = lambda dtype, features: np.array([0.05, 0.0, 0.95])  # noqa: E731
    acc_wrong, _ = ld._compute_label_updates(_decisions(), 0.0, HEAD_LABELS, pfn_wrong, lr=1.0)

    # Model confidently predicts fail (p_fail=0.95) and we observe FAIL.
    pfn_right = lambda dtype, features: np.array([0.95, 0.0, 0.05])  # noqa: E731
    acc_right, _ = ld._compute_label_updates(_decisions(), 0.0, HEAD_LABELS, pfn_right, lr=1.0)

    # (δ_fail − p_fail): wrong case 1−0.05=0.95 (large); right case 1−0.95=0.05 (small).
    assert acc_wrong[("task_completion", 10, "fail")] > acc_right[("task_completion", 10, "fail")]


def test_compute_skips_unknown_decision_types(monkeypatch):
    monkeypatch.setattr(ld, "extract_features", lambda txt: {10: 1.0})
    pfn = lambda dtype, features: np.array([0.4, 0.3, 0.3])  # noqa: E731
    decisions = [{"type": "tool_selection", "input": "x"}]  # no head for this
    acc, touched = ld._compute_label_updates(decisions, 1.0, HEAD_LABELS, pfn, lr=0.5)
    assert acc == {} and touched == {}


# --------------------------------------------------------------------------
# Persistence: load order, spec filter, top-K, decay
# --------------------------------------------------------------------------


def _pfn(dtype, features):
    return np.array([0.6, 0.3, 0.1])


def test_update_then_load_assembles_vectors_in_label_order(monkeypatch):
    monkeypatch.setattr(ld, "extract_features", lambda txt: {42: 1.0})
    from victor.core.database import get_project_database

    n = ld.update_delta_from_session(
        "s1",
        reward=1.0,
        decisions=_decisions(),
        predict_fn=_pfn,
        head_labels=HEAD_LABELS,
    )
    assert n > 0

    delta = ld.load_delta("task_completion", LABELS)
    assert 42 in delta
    vec = delta[42]
    # Order matches LABELS = [fail, partial, pass]; observed pass -> pass index positive.
    assert vec[LABELS.index("pass")] > 0
    assert vec[LABELS.index("fail")] < 0


def test_load_missing_label_is_zero(monkeypatch):
    from victor.core.database import get_project_database

    db = get_project_database()
    # Only a "fail" row exists for hash 5.
    db.execute(
        f"INSERT INTO {Tables.LOCAL_CLASSIFIER_DELTA}"
        " (decision_type, feature_hash, label, weight, feature_spec_version)"
        " VALUES (?,?,?,?,?)",
        ("task_completion", 5, "fail", 0.7, "1"),
    )
    delta = ld.load_delta("task_completion", LABELS)
    vec = delta[5]
    assert vec[LABELS.index("fail")] == pytest.approx(0.7)
    assert vec[LABELS.index("pass")] == 0.0
    assert vec[LABELS.index("partial")] == 0.0


def test_load_filters_stale_feature_spec_version(monkeypatch):
    from victor.core.database import get_project_database

    db = get_project_database()
    db.execute(
        f"INSERT INTO {Tables.LOCAL_CLASSIFIER_DELTA}"
        " (decision_type, feature_hash, label, weight, feature_spec_version)"
        " VALUES (?,?,?,?,?)",
        ("task_completion", 9, "pass", 0.5, "1"),  # current spec
    )
    db.execute(
        f"INSERT INTO {Tables.LOCAL_CLASSIFIER_DELTA}"
        " (decision_type, feature_hash, label, weight, feature_spec_version)"
        " VALUES (?,?,?,?,?)",
        ("task_completion", 9, "fail", -0.5, "0"),  # stale spec -> ignored
    )
    delta = ld.load_delta("task_completion", LABELS)
    # Only the spec="1" row counts; fail (spec 0) is dropped.
    assert delta[9][LABELS.index("pass")] == pytest.approx(0.5)
    assert delta[9][LABELS.index("fail")] == 0.0


def test_top_k_trims_per_label(monkeypatch):
    """Beyond top_k rows per (decision_type, label), the smallest |weight| go."""
    from victor.core.database import get_project_database

    db = get_project_database()
    # Insert 10 rows for "pass" with increasing |weight|; top_k=3 keeps the 3 largest.
    for i in range(10):
        db.execute(
            f"INSERT INTO {Tables.LOCAL_CLASSIFIER_DELTA}"
            " (decision_type, feature_hash, label, weight, feature_spec_version)"
            " VALUES (?,?,?,?,?)",
            ("task_completion", i, "pass", float(i + 1), "1"),
        )
    # Run an update that touches none of these hashes but triggers a trim.
    monkeypatch.setattr(ld, "extract_features", lambda txt: {999: 1.0})
    ld.update_delta_from_session(
        "s",
        reward=1.0,
        decisions=_decisions(),
        predict_fn=_pfn,
        head_labels=HEAD_LABELS,
        top_k=3,
    )
    rows = db.execute(
        f"SELECT feature_hash FROM {Tables.LOCAL_CLASSIFIER_DELTA} "
        f"WHERE decision_type='task_completion' AND label='pass' ORDER BY weight DESC"
    ).fetchall()
    kept = {r[0] for r in rows}
    # Largest |weight| are hashes 9,8,7 (weights 10,9,8). (The trim runs after a
    # decay+upsert; assert the three largest hashes survived and the smallest did not.)
    assert len(rows) <= 3 + 1  # +1 tolerance for the freshly upserted hash 999 if it lands here
    assert 9 in kept and 0 not in kept


def test_l2_decay_shrinks_weights(monkeypatch):
    from victor.core.database import get_project_database

    db = get_project_database()
    db.execute(
        f"INSERT INTO {Tables.LOCAL_CLASSIFIER_DELTA}"
        " (decision_type, feature_hash, label, weight, feature_spec_version)"
        " VALUES (?,?,?,?,?)",
        ("task_completion", 1, "pass", 1.0, "1"),
    )
    monkeypatch.setattr(ld, "extract_features", lambda txt: {2: 1.0})
    ld.update_delta_from_session(
        "s",
        reward=1.0,
        decisions=_decisions(),
        predict_fn=_pfn,
        head_labels=HEAD_LABELS,
        decay=0.5,
    )
    # The pre-existing hash-1/pass row decayed by 0.5 (no gradient touched it).
    w = db.execute(
        f"SELECT weight FROM {Tables.LOCAL_CLASSIFIER_DELTA} "
        f"WHERE feature_hash=1 AND label='pass'"
    ).fetchone()
    assert w[0] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Best-effort + setting gate
# --------------------------------------------------------------------------


def test_update_is_best_effort_on_db_error(monkeypatch):
    """A raising get_project_database must not propagate."""
    monkeypatch.setattr(ld, "extract_features", lambda txt: {1: 1.0})
    # get_project_database is imported lazily inside _persist_delta; patch the
    # source so the lazy import picks up the raising stub.
    import victor.core.database as dbmod

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(dbmod, "get_project_database", _boom)
    n = ld.update_delta_from_session(
        "s",
        reward=1.0,
        decisions=_decisions(),
        predict_fn=_pfn,
        head_labels=HEAD_LABELS,
    )
    assert n == 0  # swallowed, not raised


def test_update_disabled_returns_zero(monkeypatch):
    """Production path honors local_learning_enabled=False."""

    class _DisabledSettings:
        local_learning_enabled = False
        local_learning_lr = 0.1
        local_learning_top_k = 2000
        local_learning_decay = 0.995

    monkeypatch.setattr(ld, "extract_features", lambda txt: {1: 1.0})
    monkeypatch.setattr(
        "victor.config.decision_settings.DecisionServiceSettings", _DisabledSettings
    )
    n = ld.update_delta_from_session(  # no predict_fn -> production path
        "s", reward=1.0, decisions=_decisions()
    )
    assert n == 0


def test_update_empty_session_returns_zero():
    n = ld.update_delta_from_session(
        "empty-session",
        reward=1.0,
        decisions=[],
        predict_fn=_pfn,
        head_labels=HEAD_LABELS,
    )
    assert n == 0


# --------------------------------------------------------------------------
# Idempotency (production path: double-recorded outcome must not double-count)
# --------------------------------------------------------------------------


def test_update_idempotent_on_repeated_session(monkeypatch):
    """A second call for the same session is a no-op (no double SGD count).

    Reproduces the benchmark's by-design double-record (per-task on_progress +
    the post-run safety-net loop both call record_session_outcome per task). The
    production path (no injected predict_fn) claims the session on first apply.
    """
    monkeypatch.setattr(ld, "extract_features", lambda txt: {7: 1.0})
    # Deterministic artifact-independent predict (production path resolves it
    # via _get_default_predict).
    monkeypatch.setattr(
        ld,
        "_get_default_predict",
        lambda: ((lambda dtype, feats: np.array([0.6, 0.3, 0.1])), dict(HEAD_LABELS)),
    )

    n1 = ld.update_delta_from_session("dup-session", reward=1.0, decisions=_decisions())
    n2 = ld.update_delta_from_session("dup-session", reward=1.0, decisions=_decisions())
    assert n1 > 0
    assert n2 == 0  # already claimed -> skipped

    # Delta weights equal ONE application, not two.
    delta = ld.load_delta("task_completion", LABELS)
    vec = delta[7]
    # Observed pass, model p_pass=0.1 -> grad_pass = 1-0.1 = 0.9; lr default 0.1;
    # plus 0.995 decay. Two applications would roughly double the magnitude.
    assert vec[LABELS.index("pass")] == pytest.approx(0.1 * 1.0 * 0.9 * 0.995)


def test_update_distinct_sessions_both_applied(monkeypatch):
    """Distinct sessions are each applied exactly once."""
    monkeypatch.setattr(ld, "extract_features", lambda txt: {7: 1.0})
    monkeypatch.setattr(
        ld,
        "_get_default_predict",
        lambda: ((lambda dtype, feats: np.array([0.6, 0.3, 0.1])), dict(HEAD_LABELS)),
    )
    n1 = ld.update_delta_from_session("sess-A", reward=1.0, decisions=_decisions())
    n2 = ld.update_delta_from_session("sess-B", reward=1.0, decisions=_decisions())
    assert n1 > 0 and n2 > 0
