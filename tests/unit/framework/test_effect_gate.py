# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the effect-grounded completion gate (EVR-4, ADR-010)."""

from types import SimpleNamespace

from victor.framework.effect_gate import (
    EffectClass,
    EffectGate,
    EffectGateConfig,
    EffectLedger,
    GroundedClaimChecker,
    classify_tool_result,
)
from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult


def _turn(*, tool_results=None, content="", is_qa=False, response_metadata=None):
    return SimpleNamespace(
        tool_results=tool_results or [],
        content=content,
        is_qa_response=is_qa,
        response=SimpleNamespace(metadata=response_metadata or {}),
    )


def _complete(score=0.9, metadata=None):
    return EvaluationResult(
        decision=EvaluationDecision.COMPLETE,
        score=score,
        reason="done",
        metadata=dict(metadata or {}),
    )


def _gate(*, enabled=True, max_downgrades=2, workspace=None, verify_artifacts=True):
    return EffectGate(
        EffectGateConfig(
            enabled=enabled, max_downgrades=max_downgrades, verify_artifacts=verify_artifacts
        ),
        workspace_resolver=(lambda state: workspace) if workspace is not None else None,
    )


# --- classify_tool_result ------------------------------------------------------------------------


def test_write_tool_is_workspace_delta():
    ev = classify_tool_result({"tool_name": "write", "args": {"path": "a.py"}, "success": True})
    assert ev is not None
    assert ev.effect_class is EffectClass.WORKSPACE_DELTA
    assert ev.artifact == "a.py"


def test_alias_names_resolve_canonically():
    # write_file → write, edit_file → edit, apply_patch → patch, bash → shell
    for name in ("write_file", "edit_file", "apply_patch"):
        ev = classify_tool_result(
            {"tool_name": name, "args": {"file_path": "x.py"}, "success": True}
        )
        assert ev is not None and ev.effect_class is EffectClass.WORKSPACE_DELTA, name
    ev = classify_tool_result({"tool_name": "bash", "args": {"cmd": "touch x"}, "success": True})
    assert ev is not None and ev.effect_class is EffectClass.WORKSPACE_DELTA


def test_failed_tool_yields_no_evidence():
    assert (
        classify_tool_result({"tool_name": "write", "args": {"path": "a.py"}, "success": False})
        is None
    )


def test_readonly_shell_invocation_is_not_an_effect():
    # readonly=True is enforced by the tool itself → trustworthy read, not a delta
    ev = classify_tool_result(
        {"tool_name": "shell", "args": {"cmd": "ls -la", "readonly": True}, "success": True}
    )
    assert ev is None


def test_readonly_command_heads_are_not_deltas():
    for cmd in ("cat foo.py", "grep -r x .", "git status", "git diff HEAD~1", "ls"):
        assert (
            classify_tool_result({"tool_name": "shell", "args": {"cmd": cmd}, "success": True})
            is None
        ), cmd


def test_successful_test_command_is_verified_check():
    for cmd in ("pytest tests/unit -q", "cargo test", "make lint", "ruff check .", "mypy victor"):
        ev = classify_tool_result({"tool_name": "shell", "args": {"cmd": cmd}, "success": True})
        assert ev is not None and ev.effect_class is EffectClass.VERIFIED_CHECK, cmd


def test_test_tool_is_verified_check():
    ev = classify_tool_result({"tool_name": "test", "args": {}, "success": True})
    assert ev is not None and ev.effect_class is EffectClass.VERIFIED_CHECK


def test_mutating_shell_command_is_workspace_delta():
    ev = classify_tool_result(
        {"tool_name": "shell", "args": {"cmd": "rm -rf build"}, "success": True}
    )
    assert ev is not None and ev.effect_class is EffectClass.WORKSPACE_DELTA


def test_read_tools_yield_no_effect_evidence():
    assert (
        classify_tool_result({"tool_name": "read", "args": {"path": "a"}, "success": True}) is None
    )
    assert classify_tool_result({"tool_name": "grep", "args": {}, "success": True}) is None


def test_accepts_evaluation_trace_key_spelling():
    ev = classify_tool_result({"name": "write", "arguments": {"path": "a.py"}, "success": True})
    assert ev is not None and ev.effect_class is EffectClass.WORKSPACE_DELTA


# --- EffectLedger --------------------------------------------------------------------------------


def test_ledger_accumulates_across_turns():
    ledger = EffectLedger()
    ledger.record_turn(_turn(tool_results=[{"tool_name": "read", "args": {}, "success": True}]), 1)
    ledger.record_turn(
        _turn(tool_results=[{"tool_name": "write", "args": {"path": "a.py"}, "success": True}]), 2
    )
    ledger.record_turn(_turn(), 3)  # no-tool summary turn keeps prior evidence
    effects = ledger.candidate_effects()
    assert len(effects) == 1 and effects[0].turn_index == 2
    assert ledger.has_read_evidence()
    assert ledger.attempted_tool_calls == 2


def test_ledger_reset_clears_everything():
    ledger = EffectLedger()
    ledger.record_turn(
        _turn(tool_results=[{"tool_name": "write", "args": {"path": "a.py"}, "success": True}]), 1
    )
    ledger.reset()
    assert ledger.candidate_effects() == []
    assert not ledger.has_read_evidence()
    assert ledger.attempted_tool_calls == 0


# --- GroundedClaimChecker ------------------------------------------------------------------------


def test_grounded_by_read_evidence():
    ledger = EffectLedger()
    ledger.record_turn(_turn(tool_results=[{"tool_name": "read", "args": {}, "success": True}]), 1)
    assert GroundedClaimChecker().is_grounded("whatever", ledger, None)


def test_grounded_by_verified_file_reference(tmp_path):
    (tmp_path / "real.py").write_text("x = 1\n")
    checker = GroundedClaimChecker()
    assert checker.is_grounded("The bug is in real.py line 1", EffectLedger(), tmp_path)
    assert not checker.is_grounded("The bug is in fictional.py", EffectLedger(), tmp_path)


def test_not_grounded_without_evidence_or_workspace():
    assert not GroundedClaimChecker().is_grounded("Fixed a.py", EffectLedger(), None)


# --- EffectGate.apply ----------------------------------------------------------------------------


async def test_disabled_gate_is_strict_noop():
    gate = _gate(enabled=False)
    evaluation = _complete()
    gate.record(_turn(), {})  # must not mutate anything
    result = await gate.apply(evaluation, _turn(), {"task_type": "edit"})
    assert result is evaluation  # identity: byte-stable flag-off
    assert evaluation.metadata == {}
    assert gate.ledger.attempted_tool_calls == 0


async def test_non_complete_decisions_pass_through():
    gate = _gate()
    evaluation = EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.5, reason="going")
    assert await gate.apply(evaluation, _turn(), {"task_type": "edit"}) is evaluation


async def test_complete_without_effect_downgrades_to_retry():
    gate = _gate()
    gate.record(_turn(tool_results=[{"tool_name": "read", "args": {}, "success": True}]), {})
    result = await gate.apply(
        _complete(score=0.9), _turn(content="I fixed it"), {"task_type": "edit"}
    )
    assert result.decision == EvaluationDecision.RETRY
    assert result.score <= 0.4
    assert result.reason.startswith("completion-without-effect")
    assert result.metadata["completion_without_effect"] is True
    assert result.metadata["expected_effect_class"] == EffectClass.WORKSPACE_DELTA.value
    assert "effect_gate" in result.metadata


async def test_complete_stands_with_recorded_write():
    gate = _gate()
    gate.record(
        _turn(tool_results=[{"tool_name": "write", "args": {"path": "a.py"}, "success": True}]), {}
    )
    evaluation = _complete()
    result = await gate.apply(evaluation, _turn(content="done"), {"task_type": "edit"})
    assert result is evaluation
    assert result.decision == EvaluationDecision.COMPLETE
    assert result.metadata["effect_gate"]["satisfied_by"]["effects"]


async def test_verified_check_satisfies_mutation_task():
    gate = _gate()
    gate.record(
        _turn(tool_results=[{"tool_name": "shell", "args": {"cmd": "pytest -q"}, "success": True}]),
        {},
    )
    result = await gate.apply(_complete(), _turn(content="tests pass"), {"task_type": "bug_fix"})
    assert result.decision == EvaluationDecision.COMPLETE


async def test_session_scope_no_tool_summary_turn_completes():
    # Effectful turn 1, summary turn 2 with no tools — session-scoped ledger accepts it.
    gate = _gate()
    gate.record(
        _turn(tool_results=[{"tool_name": "edit", "args": {"path": "b.py"}, "success": True}]), {}
    )
    gate.record(_turn(), {})
    result = await gate.apply(_complete(), _turn(content="summary"), {"task_type": "refactor"})
    assert result.decision == EvaluationDecision.COMPLETE


async def test_qa_direct_answer_passes():
    gate = _gate()
    result = await gate.apply(
        _complete(), _turn(content="The answer is 42", is_qa=True), {"task_type": "general_query"}
    )
    assert result.decision == EvaluationDecision.COMPLETE
    assert result.metadata["effect_gate"]["satisfied_by"] == {"grounding": "direct_answer_no_tools"}


async def test_qa_with_read_evidence_passes():
    gate = _gate()
    gate.record(_turn(tool_results=[{"tool_name": "read", "args": {}, "success": True}]), {})
    result = await gate.apply(
        _complete(), _turn(content="Per foo.py ..."), {"task_type": "explain"}
    )
    assert result.decision == EvaluationDecision.COMPLETE


async def test_qa_with_only_failed_tools_downgrades():
    gate = _gate()
    gate.record(
        _turn(tool_results=[{"tool_name": "read", "args": {}, "success": False, "error": "x"}]), {}
    )
    result = await gate.apply(_complete(), _turn(content="It works"), {"task_type": "explain"})
    assert result.decision == EvaluationDecision.RETRY
    assert result.metadata["expected_effect_class"] == EffectClass.GROUNDED_CLAIM.value


async def test_forced_completion_routes_through_lenient_path():
    gate = _gate()
    # forced_task_completion metadata + zero tools attempted → lenient direct-answer pass,
    # even for a mutation task type.
    result = await gate.apply(
        _complete(metadata={"forced_task_completion": True}),
        _turn(content="done"),
        {"task_type": "edit"},
    )
    assert result.decision == EvaluationDecision.COMPLETE


async def test_team_execution_bypassed_with_annotation():
    gate = _gate()
    turn = _turn(content="team output", response_metadata={"execution_mode": "team_execution"})
    evaluation = _complete()
    result = await gate.apply(evaluation, turn, {"task_type": "edit"})
    assert result is evaluation
    assert result.metadata["effect_gate"] == {"bypassed": "team_execution"}


# --- downgrade budget ----------------------------------------------------------------------------


async def test_downgrade_budget_then_annotate_and_allow():
    gate = _gate(max_downgrades=2)
    state = {"task_type": "edit"}
    r1 = await gate.apply(_complete(), _turn(content="done"), state)
    r2 = await gate.apply(_complete(), _turn(content="done"), state)
    assert r1.decision == r2.decision == EvaluationDecision.RETRY
    r3 = await gate.apply(_complete(), _turn(content="done"), state)
    assert r3.decision == EvaluationDecision.COMPLETE
    assert r3.metadata["effect_gate_exhausted"] is True


async def test_reset_restores_downgrade_budget():
    gate = _gate(max_downgrades=1)
    state = {"task_type": "edit"}
    r1 = await gate.apply(_complete(), _turn(content="done"), state)
    assert r1.decision == EvaluationDecision.RETRY
    gate.reset()
    r2 = await gate.apply(_complete(), _turn(content="done"), state)
    assert r2.decision == EvaluationDecision.RETRY  # budget available again after reset


# --- artifact stat-verification ------------------------------------------------------------------


async def test_missing_artifact_not_counted_when_workspace_resolvable(tmp_path):
    gate = _gate(workspace=tmp_path)
    gate.record(
        _turn(tool_results=[{"tool_name": "write", "args": {"path": "ghost.py"}, "success": True}]),
        {},
    )
    result = await gate.apply(_complete(), _turn(content="wrote ghost.py"), {"task_type": "edit"})
    assert result.decision == EvaluationDecision.RETRY


async def test_existing_artifact_counts(tmp_path):
    (tmp_path / "real.py").write_text("x = 1\n")
    gate = _gate(workspace=tmp_path)
    gate.record(
        _turn(tool_results=[{"tool_name": "write", "args": {"path": "real.py"}, "success": True}]),
        {},
    )
    result = await gate.apply(_complete(), _turn(content="wrote real.py"), {"task_type": "edit"})
    assert result.decision == EvaluationDecision.COMPLETE


async def test_unresolvable_workspace_counts_not_blocks():
    gate = _gate()  # no workspace resolver
    gate.record(
        _turn(tool_results=[{"tool_name": "write", "args": {"path": "ghost.py"}, "success": True}]),
        {},
    )
    result = await gate.apply(_complete(), _turn(content="wrote it"), {"task_type": "edit"})
    assert result.decision == EvaluationDecision.COMPLETE


async def test_shell_delta_has_no_artifact_and_counts(tmp_path):
    gate = _gate(workspace=tmp_path)
    gate.record(
        _turn(
            tool_results=[{"tool_name": "shell", "args": {"cmd": "mkdir -p out"}, "success": True}]
        ),
        {},
    )
    result = await gate.apply(_complete(), _turn(content="created out/"), {"task_type": "setup"})
    assert result.decision == EvaluationDecision.COMPLETE
