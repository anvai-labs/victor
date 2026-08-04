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

from unittest.mock import MagicMock

import pytest

from victor.framework.agentic_loop import AgenticLoop, LoopResult
from victor.framework.loop import guards as loop_guards


@pytest.mark.timeout(30)
class TestIterationBoundHolds:
    async def test_never_complete_loop_stops_at_max_iterations(self):
        # A stub orchestrator that never signals COMPLETE must still terminate
        # within max_iterations (the iteration axis is bounded).
        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=4,
            enable_fulfillment_check=False,
            enable_adaptive_iterations=False,
        )
        result = await loop.run("Do something that never completes")
        assert isinstance(result, LoopResult)
        # Adaptive extension is off, so iterations must not exceed the cap.
        assert len(result.iterations) <= 4
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

        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=3,
            enable_fulfillment_check=False,
            enable_adaptive_iterations=False,
        )
        await loop.run("A healthy bounded task")

        g = captured.get("guards")
        assert g is not None, "loop must install LoopGuards"
        assert g.iteration_count >= 1
        # Healthy invariant: perception is called exactly once per iteration.
        assert g.perception_calls == g.iteration_count, (
            f"perception_calls={g.perception_calls} != iterations={g.iteration_count} "
            "— the once-per-iteration premise of the spin signal has drifted"
        )
