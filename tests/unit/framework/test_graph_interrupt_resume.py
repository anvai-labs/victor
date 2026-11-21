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

    resumed = await compiled.invoke(None, thread_id="t")
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

    resumed = await compiled.invoke(None, thread_id="t")
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

    await compiled.invoke(None, thread_id="t")
    assert calls == ["a", "c"]
    assert routed == ["route"]  # the router is not re-evaluated on resume


@pytest.mark.asyncio
async def test_interrupt_after_a_terminal_node_resumes_to_completion_without_rerun():
    calls: List[str] = []
    compiled, _ = await _run(_graph(calls), interrupt_after=["b"])

    first = await compiled.invoke({}, thread_id="t")
    assert (first.interrupted, first.next_node) == (True, END)
    resumed = await compiled.invoke(None, thread_id="t")
    assert calls == ["a", "b"]
    assert resumed.success and resumed.state["count"] == 2


@pytest.mark.asyncio
async def test_crash_resume_continues_after_the_last_completed_node():
    calls: List[str] = []
    compiled, _ = await _run(_graph(calls, fail_once={"b": 1}))

    crashed = await compiled.invoke({}, thread_id="t")
    assert crashed.success is False and calls == ["a", "b"]

    resumed = await compiled.invoke(None, thread_id="t")
    assert calls == ["a", "b", "b"]  # "a" completed before the crash; only "b" re-runs
    assert resumed.success and resumed.state["count"] == 2


@pytest.mark.asyncio
async def test_re_invoking_a_completed_thread_executes_nothing():
    calls: List[str] = []
    compiled, _ = await _run(_graph(calls))
    await compiled.invoke({}, thread_id="t")
    again = await compiled.invoke(None, thread_id="t")
    assert calls == ["a", "b"]
    assert again.success and again.node_history == [] and again.state["count"] == 2


def _durable_checkpointers(tmp_path):
    """FOU-C2 item 24: the FEP-0032 contract holds on every checkpointer.

    Besides MemoryCheckpointer, the production durable adapters (FOU-D2
    default seam) — resume depends on their ``next_node`` /
    ``checkpoint_boundary`` metadata round-trip.
    """
    from victor.storage.checkpoints.backends.sqlite_backend import SQLiteCheckpointBackend
    from victor.storage.checkpoints.graph_adapter import (
        GraphSQLiteCheckpointer,
        ProjectDatabaseGraphCheckpointer,
    )

    return [
        ("memory", MemoryCheckpointer),
        ("project-db", lambda: ProjectDatabaseGraphCheckpointer(tmp_path / "project.db")),
        (
            "sqlite-backend",
            lambda: GraphSQLiteCheckpointer(
                SQLiteCheckpointBackend(storage_path=tmp_path / "lite")
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_interrupt_after_resumes_at_the_successor_on_every_durable_checkpointer(tmp_path):
    """Acceptance at the real seam: interrupt-after resume was only ever
    exercised against MemoryCheckpointer; the durable adapters must carry the
    resume position end-to-end through CompiledGraph, not just round-trip it."""
    for label, factory in _durable_checkpointers(tmp_path):
        calls: List[str] = []
        compiled, _ = await _run(_graph(calls), checkpointer=factory(), interrupt_after=["a"])

        paused = await compiled.invoke({}, thread_id="t")
        assert (paused.interrupted, paused.next_node) == (True, "b"), label
        assert calls == ["a"], label

        resumed = await compiled.invoke(None, thread_id="t")
        assert calls == ["a", "b"], label  # completed "a" is not executed again
        assert resumed.success and resumed.state["count"] == 2, label


@pytest.mark.asyncio
async def test_completed_thread_with_sparse_after_nodes_executes_nothing_on_reinvoke(tmp_path):
    """FOU-C2 item 24 (the completed-node resume defect): with executor-style
    ``checkpoint_after_nodes`` — ``AgenticLoopGraphExecutor`` compiles with
    ``{"act"}`` — the last durable checkpoint of a *completed* run sat
    mid-graph, so re-invoking the thread re-executed the completed tail node.
    The run's completion must itself be checkpointed (next_node=END)."""
    for label, factory in _durable_checkpointers(tmp_path):
        calls: List[str] = []
        compiled, _ = await _run(
            _graph(calls), checkpointer=factory(), checkpoint_after_nodes={"a"}
        )

        done = await compiled.invoke({}, thread_id="t")
        assert done.success and calls == ["a", "b"], label

        again = await compiled.invoke(None, thread_id="t")
        assert calls == ["a", "b"], label  # the completed tail is not re-executed
        assert again.success and again.node_history == [] and again.state["count"] == 2, label


@pytest.mark.asyncio
async def test_sparse_after_nodes_completion_checkpoint_records_the_end_boundary(tmp_path):
    calls: List[str] = []
    checkpointer = MemoryCheckpointer()
    compiled, _ = await _run(_graph(calls), checkpointer=checkpointer, checkpoint_after_nodes={"a"})
    await compiled.invoke({}, thread_id="t")

    latest = await checkpointer.load("t")
    assert latest.node_id == "b"
    assert latest.metadata["checkpoint_boundary"] == "after"
    assert latest.next_node == END


@pytest.mark.asyncio
async def test_resume_api_of_completed_sparse_thread_executes_nothing():
    calls: List[str] = []
    compiled, _ = await _run(
        _graph(calls), checkpointer=MemoryCheckpointer(), checkpoint_after_nodes={"a"}
    )
    await compiled.invoke({}, thread_id="t")

    resumed = await compiled.resume("t")
    assert calls == ["a", "b"]
    assert resumed.success and resumed.node_history == []


@pytest.mark.asyncio
async def test_sparse_crash_resume_then_reinvoke_stops_at_the_end_boundary():
    """Crash in the non-checkpointed tail, resume (tail re-runs — it never
    completed), finish — and a further re-invoke must now execute nothing
    because the completion is durable."""
    calls: List[str] = []
    compiled, _ = await _run(
        _graph(calls, fail_once={"b": 1}),
        checkpointer=MemoryCheckpointer(),
        checkpoint_after_nodes={"a"},
    )

    crashed = await compiled.invoke({}, thread_id="t")
    assert crashed.success is False and calls == ["a", "b"]

    resumed = await compiled.invoke(None, thread_id="t")
    assert calls == ["a", "b", "b"] and resumed.success

    again = await compiled.invoke(None, thread_id="t")
    assert calls == ["a", "b", "b"]  # done stays done
    assert again.success and again.node_history == []


@pytest.mark.asyncio
async def test_full_checkpointing_gains_no_extra_completion_checkpoint():
    """Default (full) checkpointing already ends at the final node's own
    checkpoint; the completion record must not duplicate it."""
    calls: List[str] = []
    checkpointer = MemoryCheckpointer()
    compiled, _ = await _run(_graph(calls), checkpointer=checkpointer)
    await compiled.invoke({}, thread_id="t")

    checkpoints = await checkpointer.list("t")
    assert len(checkpoints) == 2  # one per node, unchanged


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
    await compiled.invoke(None, thread_id="t")
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
    second = await compiled.invoke(None, thread_id="t")
    assert (second.interrupted, second.next_node, calls) == (True, "loop", ["loop"])
    third = await compiled.invoke(None, thread_id="t")
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
    second = [node async for node, _ in compiled.stream(None, thread_id="t")]
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
    await compiled.invoke(None, thread_id="t")
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


@pytest.mark.asyncio
async def test_resume_with_fresh_input_is_a_loud_error_not_silent_discard():
    """FEP-0032 phase 2: supplying input to a checkpointed thread used to be
    silently discarded; it must raise naming the thread. ``None`` is the
    resume signal ("omit fresh input_state" in the FEP's flow diagram)."""
    calls: List[str] = []
    compiled, _ = await _run(_graph(calls), interrupt_after=["a"])

    paused = await compiled.invoke({}, thread_id="t")
    assert paused.interrupted is True

    with pytest.raises(ValueError, match="input_state.*t|t.*input_state"):
        await compiled.invoke({"count": 99}, thread_id="t")

    # None is the resume convention: no new input, checkpoint state wins.
    resumed = await compiled.invoke(None, thread_id="t")
    assert calls == ["a", "b"]
    assert resumed.state["count"] == 2


@pytest.mark.asyncio
async def test_start_node_override_acknowledges_input_on_a_checkpointed_thread():
    """The deliberate override path: explicit input + start_node is an
    acknowledged position override, not a silent discard."""
    calls: List[str] = []
    compiled, _ = await _run(_graph(calls))
    await compiled.invoke({}, thread_id="t")
    rerun = await compiled.invoke({}, thread_id="t", start_node="a")
    assert rerun.success is True


@pytest.mark.asyncio
async def test_stream_with_fresh_input_is_a_loud_error():
    calls: List[str] = []
    compiled, _ = await _run(_graph(calls), interrupt_after=["a"])
    first = [node async for node, _ in compiled.stream({}, thread_id="t")]
    assert first == ["a"]

    with pytest.raises(ValueError, match="t"):
        async for _ in compiled.stream({"count": 99}, thread_id="t"):
            pass

    resumed = [node async for node, _ in compiled.stream(None, thread_id="t")]
    assert resumed == ["b"]


@pytest.mark.asyncio
async def test_first_invoke_accepts_none_state_schema_short_circuit():
    """input_state=None only ever reaches execution when a checkpoint exists;
    _validate_input_state must short-circuit instead of dict(None)-crashing."""
    calls: List[str] = []
    compiled, _ = await _run(_graph(calls), interrupt_after=["a"])
    # First call on a fresh thread with None: no checkpoint, deep-copied None
    # is a caller error the guard names precisely.
    with pytest.raises(ValueError, match="input_state"):
        await compiled.invoke(None, thread_id="fresh")


@pytest.mark.asyncio
async def test_compiled_resume_is_the_resume_at_entry_point():
    """FOU-D4: compiled.resume(thread_id) == invoke(None, thread_id), with a
    loud error naming the remedy when there is nothing to resume."""
    calls: List[str] = []
    compiled, _ = await _run(_graph(calls), interrupt_after=["a"])
    paused = await compiled.invoke({}, thread_id="t")
    assert paused.interrupted is True

    resumed = await compiled.resume("t")
    assert resumed.interrupted is False and calls == ["a", "b"]

    with pytest.raises(ValueError, match="t"):
        await compiled.resume("never-started")


class TestRuntimeHandleSnapshots:
    """FOU-D5f-S6c: graph states carry runtime handles in context
    (``_execution_context``); deep-copying them (result snapshots,
    MemoryCheckpointer.save) fails with 'cannot pickle _asyncio.Future'
    and kills the run. Snapshots must strip runtime handles instead."""

    @staticmethod
    def _state_with_runtime_handle():
        import asyncio
        from types import SimpleNamespace

        loop = asyncio.new_event_loop()
        future = loop.create_future()
        handle = SimpleNamespace(future=future, services={"chat": object()})
        state = {
            "query": "task",
            "context": {"_execution_context": handle, "session_id": "s-1"},
        }
        return state, handle, loop

    def test_snapshot_strips_runtime_handles_instead_of_crashing(self):
        from victor.framework.graph_execution import snapshot_state_for_result

        state, handle, loop = self._state_with_runtime_handle()
        try:
            snapshot = snapshot_state_for_result(state)
            assert snapshot["context"]["session_id"] == "s-1"
            assert "_execution_context" not in snapshot["context"]
            # The live handle is untouched (only the snapshot is sanitized).
            assert "_execution_context" in state["context"]
        finally:
            loop.close()

    async def test_memory_checkpointer_save_survives_runtime_handles(self):
        import asyncio
        import time
        from types import SimpleNamespace

        from victor.framework.agentic_graph.state import AgenticLoopStateModel
        from victor.framework.graph_checkpoint import (
            MemoryCheckpointer,
            WorkflowCheckpoint,
        )

        loop = asyncio.new_event_loop()
        future = loop.create_future()
        handle = SimpleNamespace(future=future)
        try:
            checkpointer = MemoryCheckpointer()
            checkpoint = WorkflowCheckpoint(
                checkpoint_id="cp-1",
                thread_id="t",
                node_id="a",
                state=AgenticLoopStateModel(
                    query="task",
                    max_iterations=5,
                    context={"_execution_context": handle},
                ),
                timestamp=time.time(),
            )
            await checkpointer.save(checkpoint)
            loaded = await checkpointer.load("t")
            assert loaded is not None
            assert "_execution_context" not in (loaded.state.context or {})
        finally:
            loop.close()

    async def test_checkpointer_survives_tasks_at_arbitrary_keys(self):
        """Beyond _execution_context: any runtime object (asyncio.Task, ...)
        anywhere in state must not kill the checkpoint save (tolerant
        deepcopy substitutes repr for uncopiable leaves)."""
        import asyncio
        import time
        from types import SimpleNamespace

        from victor.framework.agentic_graph.state import AgenticLoopStateModel
        from victor.framework.graph_checkpoint import (
            MemoryCheckpointer,
            WorkflowCheckpoint,
        )

        loop = asyncio.new_event_loop()
        try:
            task = loop.create_task(asyncio.sleep(0))
            handle = SimpleNamespace(background_task=task)
            checkpointer = MemoryCheckpointer()
            checkpoint = WorkflowCheckpoint(
                checkpoint_id="cp-2",
                thread_id="t2",
                node_id="a",
                state=AgenticLoopStateModel(
                    query="task",
                    max_iterations=5,
                    context={"background_task": task, "owner": handle},
                ),
                timestamp=time.time(),
            )
            await checkpointer.save(checkpoint)
            loaded = await checkpointer.load("t2")
            assert loaded is not None
            # The save survived; copyable leaves round-trip, uncopiable
            # leaves were substituted (repr) rather than killing the run.
            assert "background_task" in (loaded.state.context or {})
            assert "owner" in (loaded.state.context or {})
        finally:
            loop.close()


class TestDeepcopyPreservingHandles:
    """FOU-D5f-S6c: load_initial_state deep-copies the caller's input state;
    the live runtime execution context must survive BY REFERENCE (nodes reach
    services through it), not be deep-copied (its futures fail pickling)."""

    def test_handles_survive_by_reference_rest_is_copied(self):
        import asyncio
        from types import SimpleNamespace

        from victor.framework.graph_checkpoint import (
            deepcopy_preserving_runtime_handles,
        )

        loop = asyncio.new_event_loop()
        try:
            future = loop.create_future()
            handle = SimpleNamespace(future=future)
            state = {
                "query": "task",
                "context": {"_execution_context": handle, "session_id": "s-1"},
                "count": 1,
            }
            copied = deepcopy_preserving_runtime_handles(state)
            # Other values are private copies
            assert copied["count"] == 1
            assert copied["context"]["session_id"] == "s-1"
            copied["count"] = 2
            assert state["count"] == 1
            # The live handle is shared BY REFERENCE
            assert copied["context"]["_execution_context"] is handle
        finally:
            loop.close()

    def test_nested_handles_inside_models_and_dicts(self):
        import asyncio
        from types import SimpleNamespace

        from victor.framework.agentic_graph.state import AgenticLoopStateModel
        from victor.framework.graph_checkpoint import (
            deepcopy_preserving_runtime_handles,
        )

        loop = asyncio.new_event_loop()
        try:
            future = loop.create_future()
            handle = SimpleNamespace(future=future)
            state = AgenticLoopStateModel(
                query="task",
                max_iterations=5,
                context={"_execution_context": handle, "session_id": "s-2"},
            )
            copied = deepcopy_preserving_runtime_handles(state)
            assert copied.context["_execution_context"] is handle
            assert copied.context["session_id"] == "s-2"
        finally:
            loop.close()
