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
    """FEP-0043 promise: when the run context marks the iteration budget
    exhausted, COMPLETE stands (best-effort retention over forced RETRY)."""
    loop, gate = make_loop(enabled=True)
    state = {"criteria": {"results": 5}, "results": ["a", "b"], "_criteria_budget_exhausted": True}
    gated = run_apply(gate, state)
    assert gated.decision == EvaluationDecision.COMPLETE
    assert gated.metadata["criteria_gate"]["unmet"] == ["results"]


def test_downgrades_bounded_by_max_downgrades():
    """After max_downgrades forced RETRYs, COMPLETE stands annotated with
    the unmet minimums (EffectGate max_downgrades convention)."""
    loop, gate = make_loop(enabled=True)
    gate.config.max_downgrades = 1
    state = {"criteria": {"results": 5}, "results": ["a", "b"]}
    first = run_apply(gate, state)
    assert first.decision == EvaluationDecision.RETRY
    assert first.metadata["criteria_gate"]["downgrade"] == 1
    second = run_apply(gate, state)
    assert second.decision == EvaluationDecision.COMPLETE
    assert second.metadata["criteria_gate"]["downgrades_exhausted"] is True


def test_metrics_and_metadata_preserved_on_downgrade():
    loop, gate = make_loop(enabled=True)
    state = {"criteria": {"results": 5}, "results": ["a", "b"]}
    evaluation = EvaluationResult(
        decision=EvaluationDecision.COMPLETE,
        score=0.9,
        reason="r",
        metrics={"fulfillment_score": 0.8},
        metadata={"successful_tool_progress": True},
    )
    gated = asyncio.run(gate.apply(evaluation, {"disposition": None}, state))
    assert gated.metrics == {"fulfillment_score": 0.8}
    assert gated.metadata.get("successful_tool_progress") is True
    assert gated.metadata["criteria_gate"]["downgrade"] == 1


# --- generic state mirroring (FEP-0043) -------------------------------------------------
# The gate counts state entries; mirroring is the opt-in seam that lets the run's
# own tool results populate those state keys. The framework never interprets what
# a declared key collects: same-named list payloads extend (deduped), dict
# payloads merge by key, everything else is ignored.


def _tool_result(payload=None, *, success=True, nested=None, tool_name="consumer_tool"):
    result = {"tool_name": tool_name, "args": {}, "success": success}
    if payload is not None:
        result.update(payload)
    if nested is not None:
        result["result"] = nested
    return result


def test_mirror_extends_declared_list_from_successful_results():
    from victor.framework.criteria_gate import mirror_criteria_state

    state = {"criteria": {"findings": 4}}
    mirrored = mirror_criteria_state(
        state, [_tool_result({"findings": ["f1", "f2"]}), _tool_result({"findings": ["f3"]})]
    )
    assert state["findings"] == ["f1", "f2", "f3"]
    assert mirrored == {"findings": 3}


def test_mirror_dedupes_repeated_list_entries():
    from victor.framework.criteria_gate import mirror_criteria_state

    state = {"criteria": {"findings": 4}}
    # A corrected re-proposal re-reports prior entries: the floor counts them once.
    mirror_criteria_state(state, [_tool_result({"findings": ["f1", "f2"]})])
    mirror_criteria_state(state, [_tool_result({"findings": ["f1", "f3"]})])
    assert state["findings"] == ["f1", "f2", "f3"]


def test_mirror_merges_dict_payloads_by_key():
    from victor.framework.criteria_gate import mirror_criteria_state

    state = {"criteria": {"findings": 2}}
    mirror_criteria_state(state, [_tool_result({"findings": {"F1": {"id": "F1"}}})])
    mirror_criteria_state(
        state, [_tool_result({"findings": {"F1": {"id": "F1"}, "F2": {"id": "F2"}}})]
    )
    assert sorted(state["findings"]) == ["F1", "F2"]


def test_mirror_reads_nested_result_payload():
    from victor.framework.criteria_gate import mirror_criteria_state

    state = {"criteria": {"findings": 1}}
    mirror_criteria_state(state, [_tool_result(nested={"findings": ["f1"]})])
    assert state["findings"] == ["f1"]


def test_mirror_ignores_undeclared_keys_and_failed_results():
    from victor.framework.criteria_gate import mirror_criteria_state

    state = {"criteria": {"findings": 1}}
    mirror_criteria_state(
        state,
        [
            _tool_result({"results": ["undeclared"]}),
            _tool_result({"findings": ["failed"]}, success=False),
            _tool_result({"findings": 3}),  # non-collection value
        ],
    )
    assert "results" not in state
    assert "findings" not in state


def test_mirror_without_declared_criteria_is_noop():
    from victor.framework.criteria_gate import mirror_criteria_state

    state = {"findings": ["preseeded"]}
    mirrored = mirror_criteria_state(state, [_tool_result({"findings": ["f1"]})])
    assert mirrored == {}
    assert state["findings"] == ["preseeded"]


def test_gate_apply_never_mutates_state():
    """The gate is a pure decision post-filter: mirroring is the loop's job."""
    loop, gate = make_loop(enabled=True)
    state = {"criteria": {"findings": 1}}
    run_apply(gate, state)
    assert "findings" not in state


# --- buffered loop wiring: ACT mirrors tool evidence into gate-visible state ------------


async def test_buffered_run_mirrors_tool_evidence_and_gate_sees_it():
    """A real buffered run seeds criteria from run(context=...), mirrors the
    turn's tool evidence into state, and the enabled gate passes a met
    COMPLETE through — the full generic chain without domain knowledge."""
    from types import SimpleNamespace as NS
    from unittest.mock import AsyncMock

    from victor.agent.action_authorizer import ActionIntent
    from victor.agent.services.turn_execution_runtime import TurnResult
    from victor.framework.perception_integration import Perception
    from victor.framework.task.protocols import TaskComplexity
    from victor.providers.base import CompletionResponse

    turn = TurnResult(
        response=CompletionResponse(content="collected"),
        tool_results=[
            {"tool_name": "consumer_tool", "args": {}, "success": True, "findings": ["f1", "f2"]}
        ],
        has_tool_calls=True,
        tool_calls_count=1,
    )
    executor = NS(execute_turn=AsyncMock(return_value=turn))
    loop = AgenticLoop(
        orchestrator=MagicMock(spec=[]),
        turn_executor=executor,
        max_iterations=3,
        enable_adaptive_iterations=False,
        enable_fulfillment_check=False,
        config={"disable_enhanced_completion": True, "enable_criteria_gate": True},
    )
    loop._analyze_turn = AsyncMock(
        return_value=Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.MEDIUM,
            task_analysis=MagicMock(task_type="code_generation"),
            confidence=0.8,
        )
    )
    loop._plan = AsyncMock(return_value={})
    loop._evaluate = AsyncMock(
        return_value=EvaluationResult(
            decision=EvaluationDecision.COMPLETE, score=0.9, reason="done"
        )
    )
    result = await loop.run("collect", context={"criteria": {"findings": 2}})
    assert result.success is True
    assert result.final_state["criteria"] == {"findings": 2}
    assert result.final_state["findings"] == ["f1", "f2"]
    assert loop.criteria_gate.last_unmet == []


async def test_buffered_run_unmet_floor_downgrades_then_exhaustion_yields():
    """The gate holds an ungrounded COMPLETE for at most max_downgrades rounds
    and lets the budget-exhausted terminal COMPLETE stand: a funded run always
    terminates with its best effort."""
    from types import SimpleNamespace as NS
    from unittest.mock import AsyncMock

    from victor.agent.action_authorizer import ActionIntent
    from victor.agent.services.turn_execution_runtime import TurnResult
    from victor.framework.perception_integration import Perception
    from victor.framework.task.protocols import TaskComplexity
    from victor.providers.base import CompletionResponse

    turn = TurnResult(
        response=CompletionResponse(content="thin"),
        tool_results=[{"tool_name": "t", "args": {}, "success": True, "findings": ["f1"]}],
        has_tool_calls=True,
        tool_calls_count=1,
    )
    executor = NS(execute_turn=AsyncMock(return_value=turn))
    loop = AgenticLoop(
        orchestrator=MagicMock(spec=[]),
        turn_executor=executor,
        max_iterations=2,
        enable_adaptive_iterations=False,
        enable_fulfillment_check=False,
        config={"disable_enhanced_completion": True, "enable_criteria_gate": True},
    )
    loop._analyze_turn = AsyncMock(
        return_value=Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.MEDIUM,
            task_analysis=MagicMock(task_type="code_generation"),
            confidence=0.8,
        )
    )
    loop._plan = AsyncMock(return_value={})
    decisions = []
    real_gate_apply = loop.criteria_gate.apply

    async def spying_apply(evaluation, action_result, state):
        gated = await real_gate_apply(evaluation, action_result, state)
        decisions.append((gated.decision, dict(gated.metadata)))
        return gated

    loop.criteria_gate.apply = spying_apply
    # Mock the core cascade, not the _evaluate seam: the seam is what applies
    # effect_gate -> per_turn_auditor -> criteria_gate, and this test spies on it.
    loop._evaluate_core = AsyncMock(
        return_value=EvaluationResult(
            decision=EvaluationDecision.COMPLETE, score=0.9, reason="done"
        )
    )
    result = await loop.run("collect", context={"criteria": {"findings": 5}})
    # Iteration 1: downgrade 1/2; iteration 2 (budget exhausted): COMPLETE stands.
    assert [d[0] for d in decisions] == [
        EvaluationDecision.RETRY,
        EvaluationDecision.COMPLETE,
    ]
    assert decisions[0][1]["criteria_gate"]["downgrade"] == 1
    assert decisions[1][1]["criteria_gate"]["budget_exhausted"] is True
    assert result.success is True
    assert result.final_state["findings"] == ["f1"]


def test_last_unmet_clears_once_criteria_are_met():
    loop, gate = make_loop(enabled=True)
    unmet_state = {"criteria": {"findings": 2}, "findings": ["a"]}
    run_apply(gate, unmet_state)
    assert gate.last_unmet == ["findings"]
    met_state = {"criteria": {"findings": 2}, "findings": ["a", "b"]}
    run_apply(gate, met_state)
    assert gate.last_unmet == []


# --- caller-declared completion marker (FEP-0043 extension) ------------------------------
# Some consumers model completion as a state their own tools emit ("the corpus is
# closed"), not as a countable collection. The caller declares one state key via
# ``run(context={"complete_on_key": ...})``; a successful tool result carrying
# that same-named key with a truthy value of any JSON type records the marker,
# and an enabled gate upgrades any non-COMPLETE decision to COMPLETE. The
# framework never interprets what the key means. Default-off parity: without the
# declaration, or with the gate disabled, every path is byte-identical.


def test_marker_upgrades_non_complete_decision():
    loop, gate = make_loop(enabled=True)
    state = {"complete_on_key": "done", "done": True}
    gated = run_apply(gate, state, decision=EvaluationDecision.RETRY)
    assert gated.decision == EvaluationDecision.COMPLETE
    assert gated.reason == "Caller completion marker present (state key 'done')"
    assert gated.metadata["criteria_gate"] == {"completed_by_marker": "done"}


def test_marker_upgrade_preserves_score_metrics_and_metadata():
    loop, gate = make_loop(enabled=True)
    state = {"complete_on_key": "done", "done": True}
    evaluation = EvaluationResult(
        decision=EvaluationDecision.RETRY,
        score=0.7,
        reason="r",
        metrics={"fulfillment_score": 0.6},
        metadata={"successful_tool_progress": True},
    )
    gated = asyncio.run(gate.apply(evaluation, {"disposition": None}, state))
    assert gated.decision == EvaluationDecision.COMPLETE
    assert gated.score == 0.7
    assert gated.metrics == {"fulfillment_score": 0.6}
    assert gated.metadata.get("successful_tool_progress") is True
    assert gated.metadata["criteria_gate"]["completed_by_marker"] == "done"


def test_marker_absent_leaves_retry_unchanged():
    loop, gate = make_loop(enabled=True)
    state = {"complete_on_key": "done"}
    gated = run_apply(gate, state, decision=EvaluationDecision.RETRY)
    assert gated.decision == EvaluationDecision.RETRY
    assert "criteria_gate" not in (gated.metadata or {})


def test_marker_falsy_value_does_not_upgrade():
    loop, gate = make_loop(enabled=True)
    for falsy in (False, "", [], {}, 0, None):
        state = {"complete_on_key": "done", "done": falsy}
        gated = run_apply(gate, state, decision=EvaluationDecision.RETRY)
        assert gated.decision == EvaluationDecision.RETRY, falsy


def test_disabled_gate_never_upgrades_marker():
    loop, gate = make_loop(enabled=False)
    state = {"complete_on_key": "done", "done": True}
    gated = run_apply(gate, state, decision=EvaluationDecision.RETRY)
    assert gated.decision == EvaluationDecision.RETRY


def test_non_string_complete_on_key_is_ignored():
    loop, gate = make_loop(enabled=True)
    for declaration in (5, ["done"], {"done": True}, True, None):
        state = {"complete_on_key": declaration, "done": True}
        gated = run_apply(gate, state, decision=EvaluationDecision.RETRY)
        assert gated.decision == EvaluationDecision.RETRY, declaration


def test_complete_decision_is_never_rewritten_by_marker():
    """The marker only rescues non-COMPLETE decisions; the COMPLETE paths
    (including the criteria downgrade) are unchanged."""
    loop, gate = make_loop(enabled=True)
    state = {"complete_on_key": "done", "done": True, "criteria": {"results": 5}, "results": ["a"]}
    gated = run_apply(gate, state)
    assert gated.decision == EvaluationDecision.RETRY
    assert gated.metadata["criteria_gate"]["downgrade"] == 1


def test_mirror_sets_marker_from_truthy_tool_result():
    from victor.framework.criteria_gate import mirror_criteria_state

    state = {"complete_on_key": "corpus_closed"}
    mirrored = mirror_criteria_state(
        state,
        [
            _tool_result({"corpus_closed": False}),  # falsy: not yet
            _tool_result({"corpus_closed": True}),
        ],
    )
    assert state["corpus_closed"] is True
    assert mirrored == {"corpus_closed": True}


def test_mirror_accepts_marker_of_any_truthy_json_type():
    from victor.framework.criteria_gate import mirror_criteria_state

    for truthy in ("closed", ["x"], {"at": 1}, 3):
        state = {"complete_on_key": "corpus_closed"}
        mirror_criteria_state(state, [_tool_result({"corpus_closed": truthy})])
        assert state["corpus_closed"] is True, truthy


def test_mirror_reads_nested_marker_payload():
    from victor.framework.criteria_gate import mirror_criteria_state

    state = {"complete_on_key": "corpus_closed"}
    mirror_criteria_state(state, [_tool_result(nested={"corpus_closed": True})])
    assert state["corpus_closed"] is True


def test_mirror_ignores_marker_from_failed_results_and_undeclared_keys():
    from victor.framework.criteria_gate import mirror_criteria_state

    state = {"complete_on_key": "corpus_closed"}
    mirrored = mirror_criteria_state(
        state,
        [
            _tool_result({"corpus_closed": True}, success=False),
            _tool_result({"other_key": True}),
        ],
    )
    assert "corpus_closed" not in state
    assert mirrored == {}


def test_mirror_without_complete_on_key_is_noop_for_marker():
    from victor.framework.criteria_gate import mirror_criteria_state

    state = {}
    mirrored = mirror_criteria_state(state, [_tool_result({"corpus_closed": True})])
    assert mirrored == {}
    assert "corpus_closed" not in state


def test_gate_and_mirror_integration_rescues_retry_from_tool_evidence():
    from victor.framework.criteria_gate import mirror_criteria_state

    loop, gate = make_loop(enabled=True)
    state = {"criteria": {"findings": 4}, "complete_on_key": "corpus_closed"}
    mirror_criteria_state(state, [_tool_result({"findings": ["f1"], "corpus_closed": True})])
    gated = run_apply(gate, state, decision=EvaluationDecision.RETRY)
    assert gated.decision == EvaluationDecision.COMPLETE
    assert gated.metadata["criteria_gate"]["completed_by_marker"] == "corpus_closed"


async def test_buffered_run_marker_closes_the_loop_without_a_further_turn():
    """A buffered run whose own tool emits the caller's completion marker
    finishes COMPLETE even though the evaluation seam returned RETRY — the
    generic chain, no domain knowledge in the loop."""
    from types import SimpleNamespace as NS
    from unittest.mock import AsyncMock

    from victor.agent.action_authorizer import ActionIntent
    from victor.agent.services.turn_execution_runtime import TurnResult
    from victor.framework.perception_integration import Perception
    from victor.framework.task.protocols import TaskComplexity
    from victor.providers.base import CompletionResponse

    turn = TurnResult(
        response=CompletionResponse(content="proposed"),
        tool_results=[
            {"tool_name": "consumer_tool", "args": {}, "success": True, "corpus_closed": True}
        ],
        has_tool_calls=True,
        tool_calls_count=1,
    )
    executor = NS(execute_turn=AsyncMock(return_value=turn))
    loop = AgenticLoop(
        orchestrator=MagicMock(spec=[]),
        turn_executor=executor,
        max_iterations=3,
        enable_adaptive_iterations=False,
        enable_fulfillment_check=False,
        config={"disable_enhanced_completion": True, "enable_criteria_gate": True},
    )
    loop._analyze_turn = AsyncMock(
        return_value=Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.MEDIUM,
            task_analysis=MagicMock(task_type="code_generation"),
            confidence=0.8,
        )
    )
    loop._plan = AsyncMock(return_value={})
    loop._evaluate_core = AsyncMock(
        return_value=EvaluationResult(
            decision=EvaluationDecision.RETRY, score=0.9, reason="not done"
        )
    )
    result = await loop.run("propose", context={"complete_on_key": "corpus_closed"})
    assert result.success is True
    assert result.final_state["corpus_closed"] is True
    assert loop.criteria_gate.last_unmet == []
