# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""FEP-0012 Phase 6: decision → outcome reward junction.

Tests the writer (``record_session_outcome``) and the reward-supervised
trainer reader (``load_outcome_samples``) against an isolated temp DB.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from victor.agent.decisions import outcome as outcome_mod
from victor.agent.decisions.outcome import record_session_outcome
from victor.core.database import get_database, reset_database
from victor.core.schema import Schema, Tables
from victor.ml import outcome_training as ot_mod


@pytest.fixture
def isolated_db_and_log(tmp_path, monkeypatch):
    """Point get_database + both decisions-log paths at temp artifacts."""
    db_path = tmp_path / "test_outcome.db"
    db = get_database(db_path)  # sets the singleton
    # A custom db_path doesn't trigger the global-table migration, so create
    # the decision tables explicitly (the production global DB has them).
    db.execute(Schema.DECISION_LOG)
    db.execute(Schema.DECISION_OUTCOME)
    decisions_log = tmp_path / "decisions.jsonl"
    # Both modules read the JSONL via their own module-level constant.
    monkeypatch.setattr(outcome_mod, "_DECISIONS_LOG", decisions_log)
    monkeypatch.setattr(ot_mod, "_DECISIONS_LOG", decisions_log)
    yield db_path, decisions_log
    reset_database()


def _write_decisions(log: Path, mapping: dict[str, list[dict]]) -> None:
    with log.open("w") as fh:
        for sid, recs in mapping.items():
            for r in recs:
                r = dict(r)
                r["session_id"] = sid
                fh.write(json.dumps(r) + "\n")


def _outcome_rows(session_id: str) -> list[tuple]:
    db = get_database()
    cur = db.execute(
        f"SELECT decision_id, success, quality_score, attributed_reward, "
        f"credit_method FROM {Tables.DECISION_OUTCOME} WHERE session_id=?",
        (session_id,),
    )
    return cur.fetchall()


class TestRecordSessionOutcome:
    def test_writes_one_row_per_decision(self, isolated_db_and_log):
        _, log = isolated_db_and_log
        _write_decisions(
            log,
            {
                "S1": [
                    {
                        "decision_id": "d1",
                        "type": "task_completion",
                        "input": {"m": "a"},
                    },
                    {
                        "decision_id": "d2",
                        "type": "task_completion",
                        "input": {"m": "b"},
                    },
                    {
                        "decision_id": "d3",
                        "type": "stage_detection",
                        "input": {"m": "c"},
                    },
                ]
            },
        )
        n = record_session_outcome("S1", success=True, quality_score=1.0)
        assert n == 3
        rows = _outcome_rows("S1")
        assert len(rows) == 3
        assert {r[0] for r in rows} == {"d1", "d2", "d3"}

    def test_row_contents_uniform_credit(self, isolated_db_and_log):
        _, log = isolated_db_and_log
        _write_decisions(
            log, {"S1": [{"decision_id": "d1", "type": "task_completion", "input": {}}]}
        )
        record_session_outcome("S1", success=True, quality_score=0.75)
        rows = _outcome_rows("S1")
        assert len(rows) == 1
        _decision_id, success, quality, reward, method = rows[0]
        assert success == 1
        assert quality == pytest.approx(0.75)
        assert reward == pytest.approx(0.75)  # uniform credit defaults to quality
        assert method == "session_uniform"

    def test_attributed_reward_override(self, isolated_db_and_log):
        _, log = isolated_db_and_log
        _write_decisions(
            log, {"S1": [{"decision_id": "d1", "type": "task_completion", "input": {}}]}
        )
        record_session_outcome("S1", success=False, quality_score=0.0, attributed_reward=0.3)
        _, _, _, reward, _ = _outcome_rows("S1")[0]
        assert reward == pytest.approx(0.3)

    def test_idempotent_rerecord_replaces(self, isolated_db_and_log):
        _, log = isolated_db_and_log
        _write_decisions(
            log, {"S1": [{"decision_id": "d1", "type": "task_completion", "input": {}}]}
        )
        record_session_outcome("S1", success=True, quality_score=1.0)
        record_session_outcome("S1", success=False, quality_score=0.0)  # re-record
        rows = _outcome_rows("S1")
        assert len(rows) == 1  # replaced, not duplicated
        assert rows[0][1] == 0  # success now False

    def test_empty_session_id_noop(self, isolated_db_and_log):
        assert record_session_outcome("", success=True, quality_score=1.0) == 0

    def test_no_decisions_for_session(self, isolated_db_and_log):
        # session_id with no logged decisions writes nothing
        n = record_session_outcome("NOPE", success=True, quality_score=1.0)
        assert n == 0

    def test_db_error_never_raises(self, isolated_db_and_log, monkeypatch):
        _, log = isolated_db_and_log
        _write_decisions(
            log, {"S1": [{"decision_id": "d1", "type": "task_completion", "input": {}}]}
        )

        # record_session_outcome imports get_database locally; patch the source.
        import victor.core.database as db_mod

        def boom(*a, **kw):
            raise RuntimeError("db down")

        monkeypatch.setattr(db_mod, "get_database", boom)
        # Must not raise — returns 0.
        assert record_session_outcome("S1", success=True, quality_score=1.0) == 0


class TestLoadOutcomeSamples:
    def test_joins_and_buckets_labels(self, isolated_db_and_log):
        _, log = isolated_db_and_log
        _write_decisions(
            log,
            {
                "SP": [
                    {
                        "decision_id": "p1",
                        "type": "task_completion",
                        "input": {"m": "pass1"},
                    },
                    {
                        "decision_id": "p2",
                        "type": "task_completion",
                        "input": {"m": "pass2"},
                    },
                ],
                "SF": [
                    {
                        "decision_id": "f1",
                        "type": "task_completion",
                        "input": {"m": "fail1"},
                    },
                    {
                        "decision_id": "f2",
                        "type": "stage_detection",
                        "input": {"m": "fail2"},
                    },
                ],
            },
        )
        record_session_outcome("SP", success=True, quality_score=1.0)
        record_session_outcome("SF", success=False, quality_score=0.0)

        samples = ot_mod.load_outcome_samples(["task_completion", "stage_detection"])
        assert "task_completion" in samples
        tc_labels = {lbl for _, lbl in samples["task_completion"]}
        assert tc_labels == {"pass", "fail"}
        # stage_detection decision f2 got the fail reward too
        assert "stage_detection" in samples
        assert {lbl for _, lbl in samples["stage_detection"]} == {"fail"}

    def test_partial_reward_bucket(self, isolated_db_and_log):
        _, log = isolated_db_and_log
        _write_decisions(
            log, {"S1": [{"decision_id": "d1", "type": "task_completion", "input": {}}]}
        )
        record_session_outcome("S1", success=False, quality_score=0.5)
        samples = ot_mod.load_outcome_samples(["task_completion"])
        assert samples["task_completion"][0][1] == "partial"

    def test_skips_decisions_without_outcome(self, isolated_db_and_log):
        _, log = isolated_db_and_log
        # A decision logged but never given an outcome is excluded from training.
        _write_decisions(
            log, {"S1": [{"decision_id": "d1", "type": "task_completion", "input": {}}]}
        )
        assert ot_mod.load_outcome_samples(["task_completion"]) == {}

    def test_empty_when_no_outcomes(self, isolated_db_and_log):
        assert ot_mod.load_outcome_samples(["task_completion"]) == {}

    def test_reward_label_bucketing(self):
        assert ot_mod._reward_label(1.0) == "pass"
        assert ot_mod._reward_label(0.5) == "partial"
        assert ot_mod._reward_label(0.0) == "fail"
        assert ot_mod._reward_label(0.99) == "partial"


class TestDeltaWriteFromOutcome:
    """FEP-0012 Phase 6 / acceptance #3: a resolved session grows the project delta."""

    def test_outcome_recording_writes_project_delta(self, isolated_db_and_log, monkeypatch):
        import numpy as np

        from victor.agent.decisions import local_delta as ld
        from victor.core.database import get_project_database

        _, log = isolated_db_and_log
        _write_decisions(
            log,
            {
                "S1": [
                    {
                        "decision_id": "d1",
                        "type": "task_completion",
                        "input": {"m": "all done"},
                    }
                ]
            },
        )
        # Deterministic, artifact-independent predict path.
        labels = ["fail", "partial", "pass"]
        monkeypatch.setattr(
            ld,
            "_get_default_predict",
            lambda: (
                (lambda dtype, feats: np.array([0.6, 0.3, 0.1])),
                {"task_completion": labels},
            ),
        )
        ld.clear_delta_for_tests()

        n = record_session_outcome("S1", success=True, quality_score=1.0)
        assert n == 1  # one outcome row

        # The project delta grew (FEP #3) — local-only, in the project DB.
        rows = (
            get_project_database()
            .execute(f"SELECT decision_type, label FROM {Tables.LOCAL_CLASSIFIER_DELTA}")
            .fetchall()
        )
        assert rows, "expected local_classifier_delta rows after a resolved session"
        assert any(r[0] == "task_completion" for r in rows)
        # Observed pass -> a positive pass nudge exists.
        delta = ld.load_delta("task_completion", labels)
        assert any(vec[labels.index("pass")] > 0 for vec in delta.values())

    def test_outcome_recording_no_delta_when_disabled(self, isolated_db_and_log, monkeypatch):
        from victor.agent.decisions import local_delta as ld
        from victor.core.database import get_project_database

        _, log = isolated_db_and_log
        _write_decisions(
            log,
            {
                "S1": [
                    {
                        "decision_id": "d1",
                        "type": "task_completion",
                        "input": {"m": "x"},
                    }
                ]
            },
        )

        class _Disabled:
            local_learning_enabled = False
            local_learning_lr = 0.1
            local_learning_top_k = 2000
            local_learning_decay = 0.995

        monkeypatch.setattr("victor.config.decision_settings.DecisionServiceSettings", _Disabled)
        ld.clear_delta_for_tests()

        record_session_outcome("S1", success=True, quality_score=1.0)
        rows = (
            get_project_database()
            .execute(f"SELECT COUNT(*) FROM {Tables.LOCAL_CLASSIFIER_DELTA}")
            .fetchone()
        )
        assert rows[0] == 0
