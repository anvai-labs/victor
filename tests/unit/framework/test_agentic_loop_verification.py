"""Verification evidence must govern completion on both execution paths."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.turn_execution_runtime import TurnResult
from victor.agent.task_analyzer import TaskAnalysis, TaskAnalyzer
from victor.framework.agentic_loop import AgenticLoop
from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
from victor.framework.task.protocols import TaskComplexity
from victor.framework.verification import VerificationResult
from victor.providers.base import CompletionResponse


@pytest.fixture(autouse=True)
def deterministic_analysis(monkeypatch):
    monkeypatch.setattr(
        TaskAnalyzer,
        "analyze",
        lambda *args, **kwargs: TaskAnalysis(
            complexity=TaskComplexity.SIMPLE, tool_budget=10, complexity_confidence=1.0
        ),
    )


def make_loop(verifier, retries, iterations=6):
    turn = TurnResult(response=CompletionResponse(content="", role="assistant"))

    class Port:
        async def stream_turn_act(self, *, outcome, **kwargs):
            outcome.turn_result = turn
            if False:
                yield

    loop = AgenticLoop(
        orchestrator=MagicMock(spec=[]),
        turn_executor=MagicMock(),
        streaming_act_port=Port(),
        verifier=verifier,
        max_verify_retries=retries,
        max_iterations=iterations,
        enable_fulfillment_check=False,
        enable_adaptive_iterations=False,
        config={"enable_topology_routing": False, "disable_enhanced_completion": True},
    )
    loop._act = AsyncMock(return_value=turn)
    loop._evaluate = AsyncMock(
        side_effect=lambda *args: EvaluationResult(EvaluationDecision.COMPLETE, 1.0, "done")
    )
    loop._emit_prompt_reward_outcome = MagicMock()
    return loop


async def execute(loop, streaming):
    if streaming:
        _ = [chunk async for chunk in loop.run_streaming("Write the requested files")]
        return None
    return await loop.run("Write the requested files")


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("retries", [0, 1, 2])
@pytest.mark.parametrize(
    "outcome",
    [VerificationResult(0, 1), VerificationResult(0, 0), RuntimeError("verifier unavailable")],
)
async def test_unverified_work_never_completes(streaming, retries, outcome):
    verify = (
        AsyncMock(side_effect=outcome)
        if isinstance(outcome, Exception)
        else AsyncMock(return_value=outcome)
    )
    loop = make_loop(MagicMock(verify=verify), retries)
    result = await execute(loop, streaming)
    assert verify.await_count == retries + 1
    assert loop._evaluate.await_count == retries + 1
    final, state = loop._emit_prompt_reward_outcome.call_args.args
    assert final.decision == EvaluationDecision.FAIL
    assert final.score == 0
    assert state["verification"]["verified"] is False
    assert state["verification"]["retries_used"] == retries
    if result is not None:
        assert result.success is False
        assert result.iterations[-1].evaluation is final


@pytest.mark.parametrize("streaming", [False, True])
async def test_last_allowed_retry_is_verified_and_can_succeed(streaming):
    verify = AsyncMock(side_effect=[VerificationResult(0, 1), VerificationResult(1, 1)])
    loop = make_loop(MagicMock(verify=verify), retries=1)
    result = await execute(loop, streaming)
    assert verify.await_count == 2
    final, state = loop._emit_prompt_reward_outcome.call_args.args
    assert final.decision == EvaluationDecision.COMPLETE
    assert state["verification"]["verified"] is True
    if result is not None:
        assert result.success is True


@pytest.mark.parametrize("streaming", [False, True])
async def test_iteration_limit_cannot_turn_pending_verification_into_success(streaming):
    loop = make_loop(
        MagicMock(verify=AsyncMock(return_value=VerificationResult(0, 1))), retries=2, iterations=1
    )
    result = await execute(loop, streaming)
    final, _ = loop._emit_prompt_reward_outcome.call_args.args
    assert final.decision == EvaluationDecision.RETRY
    assert final.score == 0
    if result is not None:
        assert result.success is False


@pytest.mark.parametrize("streaming", [False, True])
async def test_default_without_verifier_retains_completion(streaming):
    loop = make_loop(None, retries=0)
    result = await execute(loop, streaming)
    final, state = loop._emit_prompt_reward_outcome.call_args.args
    assert final.decision == EvaluationDecision.COMPLETE
    assert final.score == 1
    assert "verification" not in state
    if result is not None:
        assert result.success is True


async def test_stategraph_cannot_silently_ignore_verifier(monkeypatch):
    monkeypatch.setattr(
        "victor.framework.agentic_loop_executor.use_stategraph_executor", lambda: True
    )
    loop = make_loop(MagicMock(), retries=1)
    with pytest.raises(NotImplementedError, match="StateGraph.*verification"):
        await loop.run("task")


async def test_iteration_stream_cannot_silently_ignore_verifier():
    loop = make_loop(MagicMock(), retries=1)
    with pytest.raises(NotImplementedError, match="Iteration streaming.*verification"):
        _ = [iteration async for iteration in loop.stream("task")]


async def test_semantic_cache_cannot_bypass_workspace_verification(monkeypatch):
    from victor.core.feature_flags import FeatureFlag

    monkeypatch.setattr(
        "victor.core.feature_flags.is_feature_enabled",
        lambda flag: flag == FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE,
    )
    cache = MagicMock()
    cache.get.return_value = {"response": "claimed done", "similarity": 1.0}
    monkeypatch.setattr("victor.agent.semantic_response_cache.get_semantic_cache", lambda: cache)
    loop = make_loop(MagicMock(verify=AsyncMock(return_value=VerificationResult(0, 1))), retries=0)
    result = await loop.run("task")
    assert result.success is False
    cache.get.assert_not_called()


@pytest.mark.parametrize("streaming", [False, True])
async def test_retry_budget_resets_when_loop_is_reused(streaming):
    verifier = MagicMock(verify=AsyncMock(return_value=VerificationResult(0, 1)))
    loop = make_loop(verifier, retries=1)
    for _ in range(2):
        await execute(loop, streaming)
    assert verifier.verify.await_count == 4


async def test_controller_completion_cannot_leave_stale_success_after_verification():
    loop = make_loop(MagicMock(verify=AsyncMock(return_value=VerificationResult(0, 1))), retries=0)
    loop.turn_evaluation_controller.evaluate = MagicMock(
        return_value=SimpleNamespace(
            stop=True, terminal_success=True, stop_reason="search_saturation"
        )
    )
    result = await loop.run("task")
    assert result.success is False
    assert result.iterations[-1].evaluation.decision == EvaluationDecision.FAIL


async def test_backslide_downgrade_cannot_leave_stale_complete_iteration():
    verifier = MagicMock(verify=AsyncMock())
    loop = make_loop(verifier, retries=0, iterations=1)
    loop._apply_backslide_guard = lambda evaluation: EvaluationResult(
        EvaluationDecision.CONTINUE, 0.5
    )
    result = await loop.run("task")
    assert result.success is False
    assert result.iterations[-1].evaluation.decision == EvaluationDecision.CONTINUE
    verifier.verify.assert_not_awaited()
