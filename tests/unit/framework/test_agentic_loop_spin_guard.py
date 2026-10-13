"""Reproduction + regression tests for the agentic-loop spin (loop guards).

Context: on a real repo the loop ran ~5.5h effectively spinning. These tests
lock the two axes that matter:

- B1: the *iteration* axis is bounded (the loop honors max_iterations even when
  the executor never signals COMPLETE). This axis was never the problem.
- Invariant: in a healthy run perception is called EXACTLY once per iteration —
  the premise that makes ``perception_calls >> iteration_count`` a valid spin
  signal. If this drifts, the guard's threshold logic needs revisiting.

The blocking-perception reproduction (sync perceive starving the event loop so a
timeout can never fire) lands with the C1 non-blocking fix, where it goes
red-before / green-after.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.framework.agentic_loop import AgenticLoop, LoopResult
from victor.framework.loop import guards as loop_guards
from victor.agent.task_analyzer import TaskAnalysis, TaskAnalyzer
from victor.framework.task.protocols import TaskComplexity
from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
from victor.agent.services.turn_execution_runtime import TurnResult
from victor.providers.base import CompletionResponse


@pytest.fixture(autouse=True)
def deterministic_task_analysis(monkeypatch):
    """Keep real perception/offloading, without loading classifiers or plugins."""

    def analyze(self, query, context=None):
        return TaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=1.0,
        )

    monkeypatch.setattr(TaskAnalyzer, "analyze", analyze)


def continuing_loop(max_iterations):
    """Exercise real loop termination with an explicitly nonterminal workload."""
    loop = AgenticLoop(
        orchestrator=MagicMock(spec=[]),
        max_iterations=max_iterations,
        enable_fulfillment_check=False,
        enable_adaptive_iterations=False,
        config={"enable_topology_routing": False, "disable_enhanced_completion": True},
    )
    loop._act = AsyncMock(
        return_value=TurnResult(response=CompletionResponse(content="", role="assistant"))
    )
    loop._evaluate = AsyncMock(
        side_effect=lambda *args: EvaluationResult(
            decision=EvaluationDecision.CONTINUE, score=0.5, reason="Work remains"
        )
    )
    return loop


@pytest.mark.timeout(30)
class TestIterationBoundHolds:
    @pytest.mark.parametrize("max_iterations", [1, 4])
    async def test_never_complete_loop_stops_at_max_iterations(self, max_iterations):
        # A stub orchestrator that never signals COMPLETE must still terminate
        # within max_iterations (the iteration axis is bounded).
        loop = continuing_loop(max_iterations)
        result = await loop.run("Do something that never completes")
        assert isinstance(result, LoopResult)
        # Adaptive extension is off, so iterations must not exceed the cap.
        assert len(result.iterations) == max_iterations
        assert loop._act.await_count == max_iterations
        assert loop._evaluate.await_count == max_iterations
        assert all(
            iteration.evaluation.decision == EvaluationDecision.CONTINUE
            for iteration in result.iterations
        )
        assert result.success is False
        assert result.total_duration < 20  # fast, not a spin


@pytest.mark.timeout(30)
class TestPerceptionOncePerIteration:
    async def test_healthy_run_counts_perception_once_per_iteration(self, monkeypatch):
        # Capture the per-run guard that agentic_loop installs, then assert the
        # once-per-iteration invariant that the spin signal relies on.
        captured = {}
        real_set = loop_guards.set_current_guards

        def _capture(g):
            captured["guards"] = g
            return real_set(g)

        monkeypatch.setattr(loop_guards, "set_current_guards", _capture)
        # agentic_loop imports the names into its own namespace at call time via
        # a local import, so patch there too if needed — the module-level symbol
        # is what the local `from ... import set_current_guards` resolves to.
        monkeypatch.setattr(
            "victor.framework.loop.guards.set_current_guards", _capture, raising=False
        )

        loop = continuing_loop(3)
        await loop.run("A healthy bounded task")

        g = captured.get("guards")
        assert g is not None, "loop must install LoopGuards"
        assert g.iteration_count == 3
        # Healthy invariant: perception is called exactly once per iteration.
        assert g.perception_calls == g.iteration_count, (
            f"perception_calls={g.perception_calls} != iterations={g.iteration_count} "
            "— the once-per-iteration premise of the spin signal has drifted"
        )


@pytest.mark.timeout(30)
class TestPerceptionDoesNotBlockEventLoop:
    """C1: perception's CPU-bound analyze/embed must run off the event-loop thread.

    Reproduction of the root enabler — while a synchronous perception blocked the
    loop thread, asyncio timeouts (per-turn 240s, total) could never fire, so a
    spin ran unbounded. With to_thread, the loop stays responsive.
    """

    async def test_slow_sync_perception_keeps_event_loop_responsive(self, monkeypatch):
        import asyncio
        import time as _time

        from victor.framework.perception_integration import TaskAnalyzer, perceive

        real_analyze = TaskAnalyzer.analyze

        def _slow_analyze(self, query, context=None):
            _time.sleep(0.5)  # mimic the blocking model.encode under the lock
            return real_analyze(self, query, context=context)

        monkeypatch.setattr(TaskAnalyzer, "analyze", _slow_analyze)

        ticks = {"n": 0}

        async def _heartbeat():
            while True:
                ticks["n"] += 1
                await asyncio.sleep(0.02)

        hb = asyncio.create_task(_heartbeat())
        try:
            await perceive("a query whose perception blocks for 0.5s")
        finally:
            hb.cancel()

        # With C1 (to_thread) the 0.5s sync analyze runs off-thread, so the
        # heartbeat kept ticking (~25). WITHOUT C1 it blocks the loop → ~0-1 ticks.
        assert ticks["n"] >= 5, f"event loop was blocked during perception (ticks={ticks['n']})"


@pytest.mark.timeout(30)
class TestSpinGuardTerminatesLoop:
    async def test_loop_spin_error_produces_fail_result(self, monkeypatch):
        # When a guard raises LoopSpinError, the loop must terminate with a FAIL
        # result tagged terminated_by="loop_spin_guard" (not spin, not a raw crash).
        from victor.framework.loop.guards import LoopGuards, LoopSpinError

        def _raise(self, **kwargs):
            raise LoopSpinError("test: wall-clock deadline exceeded")

        monkeypatch.setattr(LoopGuards, "enforce", _raise)

        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=50,
            enable_fulfillment_check=False,
            enable_adaptive_iterations=False,
        )
        result = await loop.run("a task the guard cuts short")
        assert result.success is False
        assert result.metadata.get("terminated_by") == "loop_spin_guard"
        assert "wall-clock deadline exceeded" in result.metadata.get("error", "")
        # Terminated at the very first iteration's guard — nowhere near 50.
        assert len(result.iterations) < 50


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("phase", ["_analyze_turn", "_plan", "_act", "_evaluate"])
async def test_deadline_interrupts_blocked_phase(phase, streaming):
    import asyncio

    loop = continuing_loop(3)
    loop.config.total_timeout = 0.1
    from types import SimpleNamespace

    loop._analyze_turn = AsyncMock(
        return_value=SimpleNamespace(intent=None, task_analysis=None, to_dict=lambda: {})
    )
    loop._plan = AsyncMock(return_value={})
    closed = asyncio.Event()

    async def blocked(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    setattr(loop, phase, blocked)
    if streaming:

        class Port:
            async def stream_turn_act(self, *, outcome, plan, state, **kwargs):
                outcome.turn_result = await loop._act(plan, state)
                if False:
                    yield

        loop.streaming_act_port = Port()

        async def collect():
            return [chunk async for chunk in loop.run_streaming("bounded task")]

        chunks = await asyncio.wait_for(collect(), timeout=1)
        assert chunks[-1].metadata["agentic_loop_success"] is False
        assert "deadline" in chunks[-1].metadata["agentic_loop_error"]
    else:
        result = await asyncio.wait_for(loop.run("bounded task"), timeout=1)
        assert result.success is False
        assert result.metadata["terminated_by"] == "loop_spin_guard"
        assert "deadline" in result.metadata["error"]
    assert closed.is_set()
    assert loop_guards.current_guards() is None


@pytest.mark.parametrize("timeout", [None, 0, 1])
async def test_semantic_cache_respects_deadline_and_disabled_default(monkeypatch, timeout):
    loop = continuing_loop(3)
    loop.config.total_timeout = timeout
    captured = {}
    real_guard = loop_guards.LoopGuards

    def guard(**kwargs):
        captured["guard"] = real_guard(**kwargs)
        return captured["guard"]

    def cache_lookup(query):
        captured["guard"].run_started_monotonic -= 2
        return {"response": "cached answer", "similarity": 1.0}

    cache = MagicMock()
    cache.get.side_effect = cache_lookup
    monkeypatch.setattr(loop_guards, "LoopGuards", guard)
    from victor.core.feature_flags import FeatureFlag

    monkeypatch.setattr(
        "victor.core.feature_flags.is_feature_enabled",
        lambda flag: flag == FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE,
    )
    monkeypatch.setattr("victor.agent.semantic_response_cache.get_semantic_cache", lambda: cache)
    result = await loop.run("q")
    if timeout is None:
        assert result.success is True
        assert result.final_state["response"] == "cached answer"
        assert result.metadata["source"] == "semantic_cache"
    else:
        assert result.success is False
        assert "deadline" in result.metadata["error"]
    assert cache.get.call_count == (0 if timeout == 0 else 1)
    loop._act.assert_not_called()
