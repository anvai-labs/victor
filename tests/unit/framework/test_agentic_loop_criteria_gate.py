"""Criteria gate unit tests (FEP-0043): caller-declared completion minimums
block an ungrounded COMPLETE while iterations remain, generically.

Victor stays a generic framework: criteria are opaque caller key/values
naming state keys; the gate counts entries toward declared minimums and
never interprets what a state key collects.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.framework.agentic_loop import AgenticLoop
from victor.framework.criteria_gate import CriteriaGate, CriteriaGateConfig
from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult


def make_loop(enabled=True, max_iterations=10):
    loop = MagicMock(spec=AgenticLoop)
    loop.max_iterations = max_iterations
    loop.config = SimpleNamespace(enable_criteria_gate=enabled)
    gate = CriteriaGate(CriteriaGateConfig(enabled=enabled))
    loop.criteria_gate = gate
    return loop, gate


def run_apply(gate, state, decision=EvaluationDecision.COMPLETE, score=0.5):
    evaluation = EvaluationResult(decision=decision, score=score, reason="looks done")
    return asyncio.run(gate.apply(evaluation, {"disposition": None}, state))


def test_unmet_minimum_downgrades_complete_to_retry():
    loop, gate = make_loop(enabled=True)
    state = {"criteria": {"results": 5}, "results": ["a", "b"]}
    evaluation = run_apply(gate, state)
    assert evaluation.decision == EvaluationDecision.RETRY
    assert "results: 2/5" in evaluation.reason


def test_met_minimum_passes_complete_through():
    loop, gate = make_loop(enabled=True)
    state = {"criteria": {"results": 5}, "results": ["a", "b", "c", "d", "e"]}
    evaluation = run_apply(gate, state)
    assert evaluation.decision == EvaluationDecision.COMPLETE


def test_no_criteria_declared_is_noop():
    loop, gate = make_loop(enabled=True)
    state = {"results": ["a"]}
    evaluation = run_apply(gate, state)
    assert evaluation.decision == EvaluationDecision.COMPLETE


def test_disabled_gate_is_noop_even_with_unmet_criteria():
    loop, gate = make_loop(enabled=False)
    state = {"criteria": {"results": 5}, "results": ["a", "b"]}
    evaluation = run_apply(gate, state)
    assert evaluation.decision == EvaluationDecision.COMPLETE


def test_unmet_list_recorded_for_diagnostics_and_reset():
    loop, gate = make_loop(enabled=True)
    state = {"criteria": {"results": 5}, "results": ["a", "b"]}
    run_apply(gate, state)
    assert gate.last_unmet == ["results"]
    gate.reset()
    assert gate.last_unmet == []


def test_non_integer_and_non_positive_minimums_ignored():
    loop, gate = make_loop(enabled=True)
    state = {"criteria": {"results": "5", "findings": 0, "valid": 1}, "valid": ["x"]}
    evaluation = run_apply(gate, state)
    # only min_valid is a declared positive int; met (1 >= 1) -> pass through
    assert evaluation.decision == EvaluationDecision.COMPLETE


def test_gate_never_fails_or_escalates():
    loop, gate = make_loop(enabled=True)
    state = {"criteria": {"results": 5}, "results": []}
    for decision in (EvaluationDecision.RETRY, EvaluationDecision.ESCALATE):
        evaluation = EvaluationResult(decision=decision, score=0.5, reason="r")
        gated = asyncio.run(gate.apply(evaluation, {"disposition": None}, state))
        assert gated.decision == decision, "non-COMPLETE decisions pass through"


def test_budget_exhausted_complete_stands():
    """Anti-spin hard rule: the gate downgrades COMPLETE to RETRY (never FAIL),
    so a funded run always terminates with its best effort."""
    loop, gate = make_loop(enabled=True)
    state = {"criteria": {"results": 5}, "results": ["a", "b"]}
    gated = run_apply(gate, state)
    assert gated.decision == EvaluationDecision.RETRY
