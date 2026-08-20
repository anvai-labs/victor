# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Closed-loop execution manifest + classifier miner (FEP-0012 closed loop).

Verifies the three roles of the per-task execution manifest:
1. The code-intelligence counter counts the live ``code search``/``grep``
   subcommands (not just the deprecated ``code_search`` stub).
2. Manifest emission joins each task's outcome (reward) + trace + decisions by
   the per-task ``session_id`` correlation spine.
3. The miner projects the manifest into reward-labeled training rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from victor.evaluation.agentic_harness import (
    AgenticExecutionTrace,
    EvalToolCall,
    _is_code_search_call,
)
from victor.evaluation import manifest as manifest_mod
from victor.evaluation.manifest import (
    build_manifest_records,
    emit_execution_manifest,
    load_decisions_by_session,
)
from victor.ml import mining


# --------------------------------------------------------------------------- #
# Glue #3: code-intelligence counter
# --------------------------------------------------------------------------- #
class _Call(EvalToolCall):
    pass


@pytest.mark.parametrize(
    "name,arguments,expected",
    [
        ("code", {"cmd": "search deserialize"}, True),
        ("code", {"cmd": "grep ForeignKey models.py"}, True),
        ("code", {"cmd": "test -x"}, False),  # test is not code intelligence
        ("code", {"cmd": ""}, False),
        ("code_search", {"query": "x"}, True),  # legacy stub still counts
        ("graph", {"mode": "callers"}, False),  # graph is separate (graph_calls)
        ("read", {"path": "a.py"}, False),
        ("shell", {"cmd": "pytest"}, False),
    ],
)
def test_is_code_search_call(name, arguments, expected):
    assert _is_code_search_call(EvalToolCall(name=name, arguments=arguments)) is expected


def test_trace_code_intelligence_counts():
    trace = AgenticExecutionTrace(task_id="t", start_time=0.0)
    trace.tool_calls = [
        EvalToolCall(name="code", arguments={"cmd": "search foo"}),
        EvalToolCall(name="code", arguments={"cmd": "grep bar"}),
        EvalToolCall(name="code", arguments={"cmd": "test -x"}),
        EvalToolCall(name="code_search", arguments={"query": "x"}),
        EvalToolCall(name="graph", arguments={"mode": "callers"}),
        EvalToolCall(name="read", arguments={"path": "a.py"}),
    ]
    assert trace.code_search_calls == 3  # code search + code grep + legacy
    assert trace.graph_calls == 1
    assert trace.code_intelligence_calls == 4


def test_trace_carries_session_id():
    """session_id stamped on the trace joins decisions to the task outcome."""
    trace = AgenticExecutionTrace(task_id="t", start_time=0.0, session_id="abc123")
    assert trace.session_id == "abc123"


# --------------------------------------------------------------------------- #
# Glue #4: manifest emission + session_id join
# --------------------------------------------------------------------------- #
class _TaskResult:
    """Duck-typed TaskResult (manifest reads via getattr)."""

    def __init__(self, **kw):
        self.status = "passed"
        self.tests_passed = 0
        self.tests_total = 0
        self.tool_calls = 0
        self.code_search_calls = 0
        self.graph_calls = 0
        self.turns = 0
        self.session_id = ""
        self.trace = {}
        self.__dict__.update(kw)


class _EvalResult:
    def __init__(self, task_results):
        self.task_results = task_results


def _write_decisions(path: Path, mapping: dict[str, list[dict]]) -> None:
    with path.open("w") as fh:
        for sid, recs in mapping.items():
            for r in recs:
                r = dict(r)
                r["session_id"] = sid
                fh.write(json.dumps(r) + "\n")


def test_load_decisions_buckets_by_session(monkeypatch, tmp_path):
    log = tmp_path / "decisions.jsonl"
    _write_decisions(
        log,
        {
            "S1": [{"type": "task_completion", "input": {"m": "a"}}] * 2,
            "S2": [{"type": "task_completion", "input": {"m": "b"}}],
            "OTHER": [{"type": "task_completion", "input": {"m": "c"}}],
        },
    )
    monkeypatch.setattr(manifest_mod, "_DECISIONS_LOG", log)
    bucketed = load_decisions_by_session({"S1", "S2"})
    assert set(bucketed) == {"S1", "S2"}
    assert len(bucketed["S1"]) == 2
    assert len(bucketed["S2"]) == 1
    assert "OTHER" not in bucketed  # unrelated session filtered out


def test_load_decisions_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest_mod, "_DECISIONS_LOG", tmp_path / "nope.jsonl")
    assert load_decisions_by_session({"S1"}) == {"S1": []}


def test_reward_derivation():
    passed = _TaskResult(status="passed", tests_passed=3, tests_total=3)
    partial = _TaskResult(status="failed", tests_passed=1, tests_total=3)
    failed = _TaskResult(status="failed", tests_passed=0, tests_total=3)
    assert manifest_mod._reward(passed) == "pass"
    assert manifest_mod._reward(partial) == "partial"
    assert manifest_mod._reward(failed) == "fail"


def test_build_manifest_records_joins_decisions(monkeypatch, tmp_path):
    log = tmp_path / "decisions.jsonl"
    _write_decisions(
        log,
        {
            "SP": [{"type": "task_completion", "input": {"m": "done"}}] * 2,
            "SF": [{"type": "task_completion", "input": {"m": "maybe"}}],
        },
    )
    monkeypatch.setattr(manifest_mod, "_DECISIONS_LOG", log)

    tasks = [
        _TaskResult(
            task_id="pass-task",
            status="passed",
            session_id="SP",
            tests_passed=2,
            tests_total=2,
            code_search_calls=2,
            trace={"session_id": "SP", "tool_calls": []},
        ),
        _TaskResult(
            task_id="fail-task",
            status="failed",
            session_id="SF",
            tests_passed=0,
            tests_total=2,
            trace={"session_id": "SF"},
        ),
    ]
    records = build_manifest_records(tasks)
    assert len(records) == 2
    by_task = {r["task_id"]: r for r in records}

    assert by_task["pass-task"]["reward"] == "pass"
    assert by_task["pass-task"]["code_search_calls"] == 2
    assert len(by_task["pass-task"]["decisions"]) == 2
    assert by_task["pass-task"]["trace"]["session_id"] == "SP"

    assert by_task["fail-task"]["reward"] == "fail"
    assert len(by_task["fail-task"]["decisions"]) == 1


def test_emit_execution_manifest_writes_jsonl(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest_mod, "_DECISIONS_LOG", tmp_path / "none.jsonl")
    eval_result = _EvalResult(
        [
            _TaskResult(
                task_id="t1",
                status="passed",
                session_id="S1",
                tests_passed=1,
                tests_total=1,
                trace={"tool_calls": [{"name": "code", "arguments": {"cmd": "search x"}}]},
            )
        ]
    )
    out = emit_execution_manifest(eval_result, output_dir=tmp_path, run_id="run1")
    assert out is not None
    assert out.name == "eval_manifest_run1.jsonl"
    rec = json.loads(out.read_text().strip())
    assert rec["task_id"] == "t1"
    assert rec["reward"] == "pass"
    assert rec["session_id"] == "S1"


def test_emit_execution_manifest_empty_skips(tmp_path):
    assert emit_execution_manifest(_EvalResult([]), output_dir=tmp_path) is None


def test_emit_execution_manifest_never_raises(monkeypatch, tmp_path):
    """Best-effort: attributeless task_results degrade gracefully, never crash."""
    monkeypatch.setattr(manifest_mod, "_DECISIONS_LOG", tmp_path / "none.jsonl")

    class Bad:
        task_results = [object()]  # no attributes — getattr defaults handle it

    # Should not raise; either returns a Path (graceful empty record) or None.
    result = emit_execution_manifest(Bad, output_dir=tmp_path)
    assert result is None or result.exists()


# --------------------------------------------------------------------------- #
# Glue #5: miner
# --------------------------------------------------------------------------- #
def _make_manifest(tmp_path, monkeypatch, *, run_id="r1"):
    log = tmp_path / "decisions.jsonl"
    _write_decisions(
        log,
        {
            "SP": [
                {
                    "type": "task_completion",
                    "input": {"msg": "all tests pass verified"},
                },
                {
                    "type": "task_completion",
                    "input": {"msg": "fix complete and checked"},
                },
            ],
            "SF": [
                {
                    "type": "task_completion",
                    "input": {"msg": "I think this is probably fixed"},
                },
                {"type": "task_completion", "input": {"msg": "looks good maybe"}},
                {"type": "stage_detection", "input": {"msg": "execution"}},
            ],
        },
    )
    monkeypatch.setattr(manifest_mod, "_DECISIONS_LOG", log)
    tasks = [
        _TaskResult(task_id="p", status="passed", session_id="SP", tests_passed=2, tests_total=2),
        _TaskResult(task_id="f", status="failed", session_id="SF", tests_passed=0, tests_total=2),
    ]
    return emit_execution_manifest(_EvalResult(tasks), output_dir=tmp_path, run_id=run_id)


def test_mine_returns_per_type_samples(tmp_path, monkeypatch):
    manifest = _make_manifest(tmp_path, monkeypatch)
    per_type = mining.mine(manifest)
    assert "task_completion" in per_type
    assert len(per_type["task_completion"]) == 4
    labels = {label for _, label in per_type["task_completion"]}
    assert labels == {"pass", "fail"}


def test_mine_detailed_carries_join_keys(tmp_path, monkeypatch):
    manifest = _make_manifest(tmp_path, monkeypatch)
    rows = mining.mine_detailed(manifest)
    assert len(rows) == 5  # 2 pass + 3 fail (incl. 1 stage_detection)
    for row in rows:
        assert {"text", "label", "decision_type", "task_id", "session_id"} <= set(row)
    types = {r["decision_type"] for r in rows}
    assert types == {"task_completion", "stage_detection"}


def test_write_training_rows_includes_features(tmp_path, monkeypatch):
    manifest = _make_manifest(tmp_path, monkeypatch)
    rows = mining.mine_detailed(manifest)
    out = tmp_path / "rows.jsonl"
    mining.write_training_rows(rows, out)
    written = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(written) == len(rows)
    assert all("features" in w and isinstance(w["features"], dict) for w in written)


def test_mine_empty_manifest(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert mining.mine(empty) == {}
    assert mining.mine_detailed(empty) == []
