# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for trace-signal grounding (real outcomes + JSONL tool_result fix)."""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from victor.core.schema import Tables
from victor.framework.rl.learners.prompt_optimizer import (
    ExecutionTrace,
    PromptOptimizerLearner,
)


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


def _write_usage(tmp_path, events):
    """Write a usage.jsonl of events into tmp_path and return its path."""
    log = tmp_path / "usage.jsonl"
    with log.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return log


def _patch_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "victor.config.settings.get_project_paths",
        lambda: SimpleNamespace(global_logs_dir=tmp_path),
    )


def test_v1_counts_tool_calls_from_tool_result(tmp_path, monkeypatch, db):
    """usage.jsonl emits tool_result (not tool_call); calls must still be counted
    so sessions aren't all dropped by the <2-calls filter."""
    _patch_logs(monkeypatch, tmp_path)
    _write_usage(
        tmp_path,
        [
            {
                "session_id": "s1",
                "event_type": "tool_result",
                "data": {"tool_name": "shell", "success": True},
            }
            for _ in range(3)
        ],
    )
    learner = PromptOptimizerLearner(name="t", db_connection=db)
    traces = learner._collect_traces(limit=10)
    assert len(traces) == 1
    assert traces[0].tool_calls == 3
    assert traces[0].session_id == "s1"


def test_v1_no_double_count_when_both_events_present(tmp_path, monkeypatch, db):
    """If an emitter ever logs BOTH tool_call and tool_result per invocation,
    count once (on tool_result) — not twice."""
    _patch_logs(monkeypatch, tmp_path)
    events = []
    for _ in range(2):
        events.append({"session_id": "s1", "event_type": "tool_call", "data": {}})
        events.append({"session_id": "s1", "event_type": "tool_result", "data": {"success": True}})
    _write_usage(tmp_path, events)
    learner = PromptOptimizerLearner(name="t", db_connection=db)
    traces = learner._collect_traces(limit=10)
    assert len(traces) == 1
    assert traces[0].tool_calls == 2  # counted once per tool_result, not twice


def test_v2_counts_tool_calls_from_tool_result(tmp_path, monkeypatch, db):
    """The v2 (Pareto) JSONL path also counts tool_result events and builds
    details even when no paired tool_call is logged."""
    _patch_logs(monkeypatch, tmp_path)
    _write_usage(
        tmp_path,
        [
            {
                "session_id": "s1",
                "event_type": "tool_result",
                "data": {"tool_name": "shell", "success": True, "result_summary": "ok"},
            }
            for _ in range(3)
        ],
    )
    learner = PromptOptimizerLearner(name="t", db_connection=db, use_pareto=True)
    traces = learner._collect_traces_v2(limit=10)
    assert len(traces) == 1
    assert traces[0].tool_calls == 3
    assert len(traces[0].tool_call_details) == 3
    assert traces[0].tool_call_details[0].tool_name == "shell"


def _seed_rl_outcome(db):
    """Create a minimal rl_outcome table (coordinator-owned) for join tests."""
    db.execute(
        f"CREATE TABLE IF NOT EXISTS {Tables.RL_OUTCOME} ("
        "learner_id TEXT, provider TEXT, model TEXT, task_type TEXT, "
        "success INTEGER, quality_score REAL, session_id TEXT)"
    )


def test_apply_real_outcomes_overrides_synthetic_score_and_task_type(db):
    """Real linked outcomes override the failure-rate proxy + 'default' task_type."""
    _seed_rl_outcome(db)
    db.execute(
        f"INSERT INTO {Tables.RL_OUTCOME} "
        "(learner_id, provider, model, task_type, success, quality_score, session_id) "
        "VALUES ('quality_weights','zai','glm','coding',1,0.9,'sX')"
    )
    db.execute(
        f"INSERT INTO {Tables.RL_OUTCOME} "
        "(learner_id, provider, model, task_type, success, quality_score, session_id) "
        "VALUES ('prompt_optimizer','zai','glm','coding',1,0.7,'sX')"
    )
    db.commit()
    learner = PromptOptimizerLearner(name="t", db_connection=db)
    trace = ExecutionTrace(
        session_id="sX",
        task_type="action",
        provider="zai",
        model="glm",
        tool_calls=5,
        tool_failures={"x": 3},
        success=False,
        completion_score=0.1,
        tokens_used=0,
    )
    learner._apply_real_outcomes([trace])
    assert trace.completion_score == pytest.approx(0.8)  # mean(0.9, 0.7)
    assert trace.success is True  # 0.8 >= 0.5
    assert trace.task_type == "coding"  # from rl_outcome, not synthetic 'action'


def test_apply_real_outcomes_falls_back_when_no_outcome(db):
    """No linked outcome → synthetic score/task_type are left untouched."""
    _seed_rl_outcome(db)
    learner = PromptOptimizerLearner(name="t", db_connection=db)
    trace = ExecutionTrace(
        session_id="sNone",
        task_type="default",
        provider="zai",
        model="glm",
        tool_calls=5,
        tool_failures={},
        success=True,
        completion_score=1.0,
        tokens_used=0,
    )
    learner._apply_real_outcomes([trace])
    assert trace.completion_score == 1.0  # unchanged
    assert trace.task_type == "default"


def test_apply_real_outcomes_ignores_other_learners(db):
    """tool_selector outcomes (tool-efficiency scores) must not be treated as
    task-completion signal for prompt reflection."""
    _seed_rl_outcome(db)
    db.execute(
        f"INSERT INTO {Tables.RL_OUTCOME} "
        "(learner_id, provider, model, task_type, success, quality_score, session_id) "
        "VALUES ('tool_selector','zai','glm','coding',1,0.2,'sX')"
    )
    db.commit()
    learner = PromptOptimizerLearner(name="t", db_connection=db)
    trace = ExecutionTrace(
        session_id="sX",
        task_type="default",
        provider="zai",
        model="glm",
        tool_calls=5,
        tool_failures={},
        success=True,
        completion_score=1.0,
        tokens_used=0,
    )
    learner._apply_real_outcomes([trace])
    assert trace.completion_score == 1.0  # tool_selector ignored; fallback retained
