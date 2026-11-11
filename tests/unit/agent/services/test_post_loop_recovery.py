"""The shared post-loop recovery rule (buffered run() epilogue + run_streaming)."""

from types import SimpleNamespace

import pytest

from victor.agent.services import post_loop_recovery as rule
from victor.agent.services.turn_execution_runtime import TurnDisposition, TurnExecutor, TurnResult
from victor.framework.evaluation_nodes import EvaluationDecision
from victor.providers.base import CompletionResponse


def _turn(disposition=TurnDisposition.CONTINUE):
    result = TurnResult(response=CompletionResponse(content=""), tool_results=[])
    result.disposition = disposition
    return result


@pytest.mark.parametrize(
    "spin, action_result, decision, expected",
    [
        (False, _turn(), EvaluationDecision.FAIL, True),
        (False, None, None, True),
        (True, _turn(), EvaluationDecision.FAIL, False),
        (False, _turn(next(d for d in TurnDisposition if d.stops_run)), None, False),
        (False, _turn(), EvaluationDecision.ESCALATE, False),
        (False, _turn(), "escalate", False),
        # Disposition is read only from real TurnResults.
        (False, SimpleNamespace(disposition=SimpleNamespace(stops_run=True)), None, True),
    ],
    ids=["fail", "nothing", "spin", "stops-run", "escalate", "escalate-str", "duck"],
)
def test_recovery_allowed(spin, action_result, decision, expected):
    assert (
        rule.recovery_allowed(
            spin_guard_stopped=spin, action_result=action_result, last_decision=decision
        )
        is expected
    )


@pytest.mark.parametrize(
    "allowed, success, response, evidence, expected",
    [
        (True, False, CompletionResponse(content=""), True, "synthesize"),
        (True, False, None, False, "complete"),
        (True, True, None, False, "complete"),
        (True, True, CompletionResponse(content=""), True, None),
        (True, False, CompletionResponse(content="text"), True, None),
        (True, False, CompletionResponse(content=""), False, None),
        (False, False, CompletionResponse(content=""), True, None),
        (False, False, None, True, None),
    ],
)
def test_recovery_kind(allowed, success, response, evidence, expected):
    assert (
        rule.recovery_kind(
            recovery_allowed=allowed,
            success=success,
            response=response,
            has_tool_evidence=evidence,
        )
        == expected
    )


@pytest.mark.parametrize(
    "action_result, expected",
    [
        (None, False),
        (SimpleNamespace(successful_tool_count=2, tool_results=[]), True),
        (SimpleNamespace(tool_results=[{"success": True}]), True),
        (SimpleNamespace(tool_results=[{"success": False}, "not-a-dict"]), False),
        (SimpleNamespace(), False),
    ],
)
def test_turn_has_tool_evidence(action_result, expected):
    assert rule.turn_has_tool_evidence(action_result) is expected


def test_loop_has_tool_evidence_scans_every_iteration():
    hit = SimpleNamespace(action_result=SimpleNamespace(tool_results=[{"success": True}]))
    miss = SimpleNamespace(action_result=None)
    assert rule.loop_has_tool_evidence(SimpleNamespace(iterations=[miss, hit])) is True
    assert rule.loop_has_tool_evidence(SimpleNamespace(iterations=[miss])) is False
    assert rule.loop_has_tool_evidence(SimpleNamespace()) is False


def test_turn_executor_exposes_the_same_rule():
    # The framework loop reaches the rule through its turn executor.
    assert TurnExecutor.post_loop_recovery_allowed is rule.recovery_allowed
    assert TurnExecutor.post_loop_recovery_kind is rule.recovery_kind
    assert TurnExecutor.turn_has_successful_tool_evidence is rule.turn_has_tool_evidence
