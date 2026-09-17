"""Ordinary edges execute sequential DAG branches through the canonical runtime."""

import pytest

from victor.framework.graph import END, MemoryCheckpointer, Send, StateGraph


def make_graph(edges, *, failing=None):
    graph = StateGraph(dict)
    nodes = set(edges) | {
        target for targets in edges.values() for target in targets if target != END
    }
    for node in nodes:

        def execute(state, node_id=node):
            if node_id == failing:
                raise RuntimeError(f"failed {node_id}")
            state.setdefault("trace", []).append(node_id)
            return state

        graph.add_node(node, execute)
    for source, targets in edges.items():
        for target in targets:
            graph.add_edge(source, target)
    graph.set_entry_point("a")
    return graph


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_multiple_successors_and_shared_descendant_execute_once_in_bfs_order(stream):
    graph = make_graph({"a": ["b", "c"], "b": ["d"], "c": ["e"], "e": ["d"]})
    compiled = graph.compile()
    if stream:
        events = [event async for event in compiled.stream({})]
        assert [node for node, _ in events] == ["a", "b", "c", "d", "e"]
        state = events[-1][1]
    else:
        result = await compiled.invoke({})
        assert result.success
        assert result.node_history == ["a", "b", "c", "d", "e"]
        state = result.state
    assert state["trace"] == ["a", "b", "c", "d", "e"]


@pytest.mark.asyncio
async def test_early_end_branch_does_not_drop_pending_sibling():
    result = await make_graph({"a": [END, "b"], "b": ["c"]}).compile().invoke({})
    assert result.success
    assert result.state["trace"] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_failed_branch_stops_remaining_nodes():
    result = await make_graph({"a": ["b", "c"]}, failing="b").compile().invoke({})
    assert not result.success
    assert result.state["trace"] == ["a"]
    assert "failed b" in result.error


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["before", "after"])
async def test_checkpoint_keeps_pending_siblings_and_completed_descendants(interrupt):
    graph = make_graph({"a": ["b", "c"], "b": ["d"], "c": ["d"]})
    checkpointer = MemoryCheckpointer()
    compiled = graph.compile(checkpointer=checkpointer, **{f"interrupt_{interrupt}": ["b"]})
    await compiled.invoke({}, thread_id="dag")
    checkpoint = await checkpointer.load("dag")
    frontier = checkpoint.metadata["sequential_frontier"]
    assert frontier["pending_nodes"] == (["b", "c"] if interrupt == "before" else ["c", "d"])

    resumed = graph.compile(checkpointer=checkpointer)
    result = await resumed.invoke({}, thread_id="dag")
    assert result.success
    assert result.state["trace"] == ["a", "b", "c", "d"]
    # A finished checkpoint is terminal, so a second resume cannot repeat the last node.
    repeated = await resumed.invoke({}, thread_id="dag")
    assert repeated.node_history == []
    assert repeated.state["trace"] == result.state["trace"]


def test_cycles_combined_with_ordinary_fanout_are_rejected_before_execution():
    with pytest.raises(ValueError, match="sequential fan-out requires an acyclic graph"):
        make_graph({"a": ["b", "c"], "b": ["a"]}).compile()


@pytest.mark.asyncio
async def test_single_edge_conditional_cycle_keeps_existing_iteration_semantics():
    graph = make_graph({"a": ["b"]})
    graph.add_conditional_edge(
        "b",
        lambda state: "again" if len(state["trace"]) < 6 else "done",
        {"again": "a", "done": END},
    )
    result = await graph.compile().invoke({})
    assert result.success
    assert result.state["trace"] == ["a", "b", "a", "b", "a", "b"]


@pytest.mark.asyncio
async def test_send_fanout_still_merges_and_joins():
    graph = StateGraph(dict)
    graph.add_node("a", lambda state: state)
    graph.add_node("b", lambda state: {"b": True})
    graph.add_node("c", lambda state: {"c": True})
    graph.add_node("join", lambda state: {**state, "joined": True})
    graph.add_conditional_edge(
        "a",
        lambda state: [Send("b", {}, "join"), Send("c", {}, "join")],
        {"b": "b", "c": "c", "join": "join"},
    )
    graph.add_edge("join", END)
    graph.set_entry_point("a")
    result = await graph.compile().invoke({})
    assert result.success
    assert result.state == {"b": True, "c": True, "joined": True}
    assert result.node_history == ["a", "send:b", "send:c", "join"]


@pytest.mark.asyncio
async def test_mixed_dynamic_send_is_rejected_before_branch_side_effects():
    graph = make_graph({"a": ["b", "c"]})
    calls = []
    graph.add_node("sent", lambda state: calls.append("sent") or state)
    graph.add_conditional_edge("b", lambda state: [Send("sent", {})], {"sent": "sent"})
    result = await graph.compile().invoke({})
    assert not result.success
    assert "Dynamic Send cannot be combined with sequential fan-out" in result.error
    assert result.state["trace"] == ["a", "b"]
    assert calls == []


@pytest.mark.asyncio
async def test_checkpoint_replay_and_start_override_cannot_discard_pending_sibling():
    graph = make_graph({"a": ["b", "c"], "b": ["d"], "c": ["d"]})
    checkpointer = MemoryCheckpointer()
    compiled = graph.compile(checkpointer=checkpointer, interrupt_after=["b"])
    await compiled.invoke({}, thread_id="original")
    checkpoint = await checkpointer.load("original")
    resumed = graph.compile(checkpointer=checkpointer)
    with pytest.raises(ValueError, match="Cannot replay a sequential fan-out checkpoint"):
        await resumed.replay_from("original", checkpoint.checkpoint_id)
    with pytest.raises(ValueError, match="Cannot override start_node"):
        await resumed.invoke({}, thread_id="original", start_node="b")
    assert len(await checkpointer.list("original")) == 2
    result = await resumed.invoke({}, thread_id="original")
    assert result.state["trace"] == ["a", "b", "c", "d"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["node", "mixed_send"])
async def test_stream_propagates_failure_after_partial_events(failure):
    graph = make_graph({"a": ["b", "c"]}, failing="b" if failure == "node" else None)
    calls = []
    if failure == "mixed_send":
        graph.add_node("sent", lambda state: calls.append("sent") or state)
        graph.add_conditional_edge("b", lambda state: [Send("sent", {})], {"sent": "sent"})
    expected = "failed b" if failure == "node" else "Dynamic Send cannot be combined"
    completed = []
    with pytest.raises(RuntimeError, match=expected):
        async for node_id, state in graph.compile().stream({}):
            completed.append(node_id)
    assert completed == ["a"]
    assert calls == []
