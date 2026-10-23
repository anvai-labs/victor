"""FEP-0032: graph interrupt/resume signal and resume-at semantics."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from victor.framework.graph import END, StateGraph
from victor.framework.graph_checkpoint import MemoryCheckpointer, WorkflowCheckpoint


def _graph(calls: List[str], *, fail_once: Dict[str, int] | None = None, **edges: Any):
    """a -> b -> END by default; each node appends its name and bumps a counter."""
    fail_once = fail_once if fail_once is not None else {}

    def make(name: str):
        async def node(state):
            calls.append(name)
            if fail_once.get(name, 0) > 0:
                fail_once[name] -= 1
                raise RuntimeError(f"{name} crashed")
            state["count"] = state.get("count", 0) + 1
            return state

        return node

    graph = StateGraph(dict)
    names = ("a", "b", "c") if "router" in edges else ("a", "b")
    for name in names:
        graph.add_node(name, make(name))
    graph.set_entry_point("a")
    if "router" in edges:
        graph.add_conditional_edge("a", edges["router"], {"b": "b", "c": "c"})
        graph.add_edge("c", END)
    else:
        graph.add_edge("a", "b")
    graph.add_edge("b", END)
    return graph


async def _run(graph, *, thread_id="t", **config):
    checkpointer = config.pop("checkpointer", None) or MemoryCheckpointer()
    compiled = graph.compile(checkpointer=checkpointer, **config)
    return compiled, checkpointer


@pytest.mark.asyncio
async def test_interrupt_after_resumes_at_the_successor_without_rerunning():
    calls: List[str] = []
    compiled, _ = await _run(_graph(calls), interrupt_after=["a"])

    paused = await compiled.invoke({}, thread_id="t")
    assert (paused.success, paused.interrupted, paused.next_node) == (True, True, "b")
    assert calls == ["a"]

    resumed = await compiled.invoke({}, thread_id="t")
    assert calls == ["a", "b"]  # completed "a" is not executed again
    assert (resumed.success, resumed.interrupted, resumed.next_node) == (True, False, None)
    assert resumed.state["count"] == 2


@pytest.mark.asyncio
async def test_interrupt_before_signals_the_pending_node():
    calls: List[str] = []
    compiled, _ = await _run(_graph(calls), interrupt_before=["b"])

    paused = await compiled.invoke({}, thread_id="t")
    assert (paused.interrupted, paused.next_node) == (True, "b")
    assert calls == ["a"]

    resumed = await compiled.invoke({}, thread_id="t")
    assert calls == ["a", "b"]
    assert resumed.interrupted is False and resumed.state["count"] == 2


@pytest.mark.asyncio
async def test_conditional_successor_is_resolved_once_and_persisted():
    calls: List[str] = []
    routed: List[str] = []

    def router(state):
        routed.append("route")
        return "c"

    compiled, checkpointer = await _run(_graph(calls, router=router), interrupt_after=["a"])
    paused = await compiled.invoke({}, thread_id="t")
    assert paused.next_node == "c"
    latest = await checkpointer.load("t")
    assert latest.node_id == "a" and latest.next_node == "c"

    await compiled.invoke({}, thread_id="t")
    assert calls == ["a", "c"]
    assert routed == ["route"]  # the router is not re-evaluated on resume


@pytest.mark.asyncio
async def test_interrupt_after_a_terminal_node_resumes_to_completion_without_rerun():
    calls: List[str] = []
    compiled, _ = await _run(_graph(calls), interrupt_after=["b"])

    first = await compiled.invoke({}, thread_id="t")
    assert (first.interrupted, first.next_node) == (True, END)
    resumed = await compiled.invoke({}, thread_id="t")
    assert calls == ["a", "b"]
    assert resumed.success and resumed.state["count"] == 2


@pytest.mark.asyncio
async def test_crash_resume_continues_after_the_last_completed_node():
    calls: List[str] = []
    compiled, _ = await _run(_graph(calls, fail_once={"b": 1}))

    crashed = await compiled.invoke({}, thread_id="t")
    assert crashed.success is False and calls == ["a", "b"]

    resumed = await compiled.invoke({}, thread_id="t")
    assert calls == ["a", "b", "b"]  # "a" completed before the crash; only "b" re-runs
    assert resumed.success and resumed.state["count"] == 2


@pytest.mark.asyncio
async def test_re_invoking_a_completed_thread_executes_nothing():
    calls: List[str] = []
    compiled, _ = await _run(_graph(calls))
    await compiled.invoke({}, thread_id="t")
    again = await compiled.invoke({}, thread_id="t")
    assert calls == ["a", "b"]
    assert again.success and again.node_history == [] and again.state["count"] == 2


@pytest.mark.asyncio
async def test_legacy_checkpoint_without_resume_metadata_resumes_at_node_id():
    calls: List[str] = []
    checkpointer = MemoryCheckpointer()
    await checkpointer.save(
        WorkflowCheckpoint(
            checkpoint_id="old",
            thread_id="t",
            node_id="a",
            state={"count": 1},
            timestamp=0.0,
        )
    )
    compiled, _ = await _run(_graph(calls), checkpointer=checkpointer)
    await compiled.invoke({}, thread_id="t")
    assert calls == ["a", "b"]  # pre-FEP-0032 behavior retained for old data


@pytest.mark.asyncio
async def test_failing_router_at_interrupt_time_still_pauses():
    calls: List[str] = []

    def router(state):
        raise RuntimeError("router down")

    compiled, checkpointer = await _run(_graph(calls, router=router), interrupt_after=["a"])
    paused = await compiled.invoke({}, thread_id="t")
    assert (paused.success, paused.interrupted, paused.next_node) == (True, True, None)
    latest = await checkpointer.load("t")
    assert latest.next_node is None and latest.metadata["checkpoint_boundary"] == "after"


@pytest.mark.asyncio
async def test_uninterrupted_runs_report_no_pause():
    calls: List[str] = []
    compiled, _ = await _run(_graph(calls))
    result = await compiled.invoke({}, thread_id="fresh")
    assert (result.interrupted, result.next_node) == (False, None)


def test_checkpoint_next_node_is_metadata_backed_and_round_trips():
    cp = WorkflowCheckpoint("id", "t", "a", {}, 0.0, metadata={"next_node": "b"})
    assert cp.next_node == "b"
    assert WorkflowCheckpoint.from_dict(cp.to_dict()).next_node == "b"
    assert WorkflowCheckpoint("id", "t", "a", {}, 0.0).next_node is None


@pytest.mark.asyncio
async def test_interrupt_before_is_bypassed_only_for_the_resumed_node():
    """A cycle that reaches the interrupted node again pauses again."""
    calls: List[str] = []

    async def loop(state):
        calls.append("loop")
        state["n"] = state.get("n", 0) + 1
        return state

    graph = StateGraph(dict)
    graph.add_node("loop", loop)
    graph.add_conditional_edge(
        "loop", lambda s: "again" if s["n"] < 2 else "done", {"again": "loop", "done": END}
    )
    graph.set_entry_point("loop")
    compiled = graph.compile(checkpointer=MemoryCheckpointer(), interrupt_before=["loop"])

    first = await compiled.invoke({}, thread_id="t")
    assert (first.interrupted, first.next_node, calls) == (True, "loop", [])
    second = await compiled.invoke({}, thread_id="t")
    assert (second.interrupted, second.next_node, calls) == (True, "loop", ["loop"])
    third = await compiled.invoke({}, thread_id="t")
    assert (third.interrupted, calls) == (False, ["loop", "loop"])


@pytest.mark.asyncio
async def test_explicit_start_node_does_not_consume_the_bypass_elsewhere():
    calls: List[str] = []
    compiled, _ = await _run(_graph(calls), interrupt_before=["b"])
    await compiled.invoke({}, thread_id="t")
    rerun = await compiled.invoke({}, thread_id="t", start_node="a")
    assert calls == ["a", "a"]
    assert (rerun.interrupted, rerun.next_node) == (True, "b")


@pytest.mark.asyncio
async def test_stream_resume_matches_invoke_resume():
    calls: List[str] = []
    compiled, _ = await _run(_graph(calls), interrupt_after=["a"])
    first = [node async for node, _ in compiled.stream({}, thread_id="t")]
    second = [node async for node, _ in compiled.stream({}, thread_id="t")]
    assert (first, second, calls) == (["a"], ["b"], ["a", "b"])


@pytest.mark.asyncio
async def test_sequential_fanout_interrupt_after_reports_the_pending_branch():
    seen: List[str] = []
    graph = StateGraph(dict)
    for name in ("a", "b", "c"):

        def node(state, name=name):
            seen.append(name)
            return state

        graph.add_node(name, node)
    graph.add_edge("a", "b")
    graph.add_edge("a", "c")
    graph.set_entry_point("a")
    compiled = graph.compile(checkpointer=MemoryCheckpointer(), interrupt_after=["a"])
    paused = await compiled.invoke({}, thread_id="t")
    assert (paused.interrupted, paused.next_node) == (True, "b")
    await compiled.invoke({}, thread_id="t")
    assert seen == ["a", "b", "c"]


class TestResumeNodeFallbacks:
    """GraphCheckpointManager._resume_node: every fallback keeps a usable resume point."""

    @staticmethod
    def _cp(**metadata):
        return WorkflowCheckpoint("id", "t", "a", {}, 0.0, metadata=metadata)

    def test_completed_checkpoint_without_a_resolver_resumes_at_node_id(self):
        from victor.framework.graph_execution import GraphCheckpointManager

        cp = self._cp(checkpoint_boundary="after")
        assert GraphCheckpointManager._resume_node(cp, {}, None) == "a"

    def test_failing_resolver_falls_back_to_rerunning_the_node(self):
        from victor.framework.graph_execution import GraphCheckpointManager

        def boom(node, state):
            raise RuntimeError("router down")

        cp = self._cp(checkpoint_boundary="after")
        assert GraphCheckpointManager._resume_node(cp, {}, boom) == "a"

    def test_fan_out_successor_falls_back_to_rerunning_the_node(self):
        from victor.framework.graph_execution import GraphCheckpointManager

        cp = self._cp(checkpoint_boundary="after")
        assert GraphCheckpointManager._resume_node(cp, {}, lambda n, s: ["send"]) == "a"

    def test_policy_errors_are_not_swallowed(self):
        from victor.core.errors import ProviderPolicyError
        from victor.framework.graph_execution import GraphCheckpointManager

        def policy(node, state):
            raise ProviderPolicyError("blocked")

        with pytest.raises(ProviderPolicyError):
            GraphCheckpointManager._resume_node(self._cp(checkpoint_boundary="after"), {}, policy)

    def test_explicit_next_node_wins_over_resolution(self):
        from victor.framework.graph_execution import GraphCheckpointManager

        cp = self._cp(checkpoint_boundary="after", next_node="z")
        assert GraphCheckpointManager._resume_node(cp, {}, lambda n, s: "b") == "z"
