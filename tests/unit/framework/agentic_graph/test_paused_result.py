# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""FOU-D4: a graph-interrupted run tells callers "paused, resume here".

FEP-0032 paused-result surfacing: the executor owns a per-run thread id,
``LoopResult`` carries the paused position (thread_id + next_node), and
``CompiledGraph.resume`` is the resume-at entry point.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from victor.framework.agentic_graph.executor import AgenticLoopGraphExecutor, LoopResult
from victor.framework.graph import StateGraph
from victor.framework.graph_checkpoint import MemoryCheckpointer


class _GraphResultStub:
    """Minimal GraphExecutionResult stand-in for from_graph_result."""

    def __init__(self, interrupted: bool, next_node: str = None):
        self.interrupted = interrupted
        self.next_node = next_node
        self.state_history = []


def _final_state(interrupted: bool):
    from types import SimpleNamespace

    return SimpleNamespace(
        action_result=None,
        evaluation=None,
        iteration=1,
        max_iterations=10,
        to_dict=lambda: {},
    )


class TestLoopResultPaused:
    def test_from_graph_result_surfaces_paused_position(self):
        result = LoopResult.from_graph_result(
            _GraphResultStub(interrupted=True, next_node="b"),
            _final_state(True),
            thread_id="run-abc",
        )
        assert result.paused is True
        assert result.metadata["paused"] == {"thread_id": "run-abc", "next_node": "b"}

    def test_from_graph_result_without_thread_id_has_no_paused_claim(self):
        result = LoopResult.from_graph_result(
            _GraphResultStub(interrupted=True, next_node="b"),
            _final_state(True),
        )
        # Without a thread id a "paused" claim is unresumable — do not surface it.
        assert result.paused is False
        assert "paused" not in result.metadata

    def test_completed_run_is_not_paused(self):
        result = LoopResult.from_graph_result(
            _GraphResultStub(interrupted=False, next_node=None),
            _final_state(False),
            thread_id="run-abc",
        )
        assert result.paused is False
        assert "paused" not in result.metadata


class TestExecutorOwnsRunThreadId:
    """The executor mints a stable per-run thread id and hands it to invoke."""

    async def test_run_passes_thread_id_and_surfaces_pause(self, monkeypatch):
        executor = AgenticLoopGraphExecutor(execution_context=MagicMock())
        captured: Dict[str, Any] = {}

        class _FakeGraphResult:
            interrupted = True
            next_node = "act"
            state_history = []
            action_result = None
            evaluation = None
            iteration = 1
            max_iterations = 10

            def to_dict(self):
                return {}

        async def fake_invoke(state, **kwargs):
            captured.update(kwargs)
            return _FakeGraphResult()

        monkeypatch.setattr(executor.compiled, "invoke", fake_invoke)

        result = await executor.run("hello")

        assert isinstance(captured.get("thread_id"), str) and captured["thread_id"]
        assert result.paused is True
        assert result.metadata["paused"]["thread_id"] == captured["thread_id"]
        assert result.metadata["paused"]["next_node"] == "act"

    async def test_run_thread_id_is_unique_per_run(self, monkeypatch):
        executor = AgenticLoopGraphExecutor(execution_context=MagicMock())
        seen: List[str] = []

        class _Done:
            interrupted = False
            next_node = None
            state_history = []
            action_result = None
            evaluation = None
            iteration = 1
            max_iterations = 10

            def to_dict(self):
                return {}

        async def fake_invoke(state, **kwargs):
            seen.append(kwargs["thread_id"])
            return _Done()

        monkeypatch.setattr(executor.compiled, "invoke", fake_invoke)
        await executor.run("one")
        await executor.run("two")
        assert len(set(seen)) == 2


class TestCompiledGraphResume:
    """``resume(thread_id)`` is the resume-at entry point (FEP-0032)."""

    async def test_resume_completes_an_interrupted_thread(self):
        calls: List[str] = []

        def make(name):
            async def node(state):
                calls.append(name)
                return state

            return node

        graph = StateGraph(dict)
        graph.add_node("a", make("a"))
        graph.add_node("b", make("b"))
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "__end__")
        compiled = graph.compile(checkpointer=MemoryCheckpointer(), interrupt_after=["a"])

        paused = await compiled.invoke({}, thread_id="t")
        assert paused.interrupted is True

        resumed = await compiled.resume("t")
        assert resumed.interrupted is False
        assert calls == ["a", "b"]

    async def test_resume_unknown_thread_is_a_loud_error(self):
        graph = StateGraph(dict)
        graph.add_node("a", make := (lambda s: s))
        graph.set_entry_point("a")
        graph.add_edge("a", "__end__")
        compiled = graph.compile(checkpointer=MemoryCheckpointer())

        with pytest.raises(ValueError, match="resume"):
            await compiled.resume("no-such-thread")
