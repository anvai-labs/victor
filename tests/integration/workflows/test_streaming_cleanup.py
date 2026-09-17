"""Public streaming boundaries close the graph task before returning control."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.coordinators.yaml_coordinator import YAMLWorkflowCoordinator
from victor.framework.graph import END, StateGraph
from victor.framework.workflow_engine import WorkflowEngine
from victor.workflows.definition import AgentNode, ParallelNode, TransformNode, WorkflowDefinition
from victor.workflows.state_graph_adapter import StateGraphWorkflowExecutor
from victor.workflows.streaming_executor import StreamingWorkflowExecutor, WorkflowEventType
from victor.workflows.unified_compiler import CachedCompiledGraph


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["close", "cancel"])
@pytest.mark.parametrize(
    "surface",
    [
        "graph",
        "adapter",
        "streaming_graph",
        "streaming_definition",
        "cached",
        "yaml_compiler",
        "yaml_definition",
        "engine_yaml_compiler",
        "engine_yaml_definition",
        "engine_graph",
    ],
)
async def test_stream_stops_owned_runner_before_returning(surface, stop, monkeypatch):
    entered = asyncio.Event()
    stopped = asyncio.Event()
    release = asyncio.Event()

    async def first(state):
        return dict(state)

    async def blocked(state):
        entered.set()
        try:
            await release.wait()
        finally:
            stopped.set()
        return dict(state)

    graph = StateGraph(dict)
    graph.add_node("first", first)
    graph.add_node("blocked", blocked)
    graph.add_edge("first", "blocked")
    graph.add_edge("blocked", END)
    graph.set_entry_point("first")
    compiled = graph.compile()
    definition = WorkflowDefinition(
        name="cleanup",
        start_node="first",
        nodes={
            "first": TransformNode(
                id="first", name="first", transform=dict, next_nodes=["blocked"]
            ),
            "blocked": TransformNode(id="blocked", name="blocked", transform=dict),
        },
    )

    async def execute_transform(self, node, state):
        return await (first(state) if node.id == "first" else blocked(state))

    monkeypatch.setattr(
        "victor.workflows.executors.transform.TransformNodeExecutor.execute", execute_transform
    )
    cached = CachedCompiledGraph(compiled_graph=compiled, workflow_name="cleanup")
    compiler = SimpleNamespace(compile_yaml=lambda *args, **kwargs: cached)
    coordinator = YAMLWorkflowCoordinator(
        unified_compiler=compiler,
        use_unified_compiler=surface not in ("yaml_definition", "engine_yaml_definition"),
    )
    monkeypatch.setattr(coordinator, "load_workflow", lambda *args, **kwargs: definition)
    engine = WorkflowEngine()
    monkeypatch.setattr(engine, "_get_unified_compiler", lambda: compiler)
    monkeypatch.setattr(engine, "_get_yaml_coordinator", lambda: coordinator)

    if surface == "graph":
        stream = compiled.stream({})
    elif surface == "adapter":
        stream = StateGraphWorkflowExecutor().stream(compiled, {})
    elif surface == "streaming_graph":
        stream = StreamingWorkflowExecutor().stream(compiled, {})
    elif surface == "streaming_definition":
        stream = StreamingWorkflowExecutor().stream(definition, {})
    elif surface == "cached":
        stream = cached.stream({})
    elif surface.startswith("yaml_"):
        stream = coordinator.stream("unused.yaml", {})
    elif surface.startswith("engine_yaml_"):
        stream = engine.stream_yaml(
            "unused.yaml", {}, use_unified_compiler=surface == "engine_yaml_compiler"
        )
    else:
        stream = engine.stream_graph(compiled, {})

    consumer = None
    try:
        await asyncio.wait_for(anext(stream), timeout=2)
        await asyncio.wait_for(entered.wait(), timeout=2)
        if stop == "close":
            await asyncio.wait_for(stream.aclose(), timeout=2)
        else:
            consumer = asyncio.create_task(anext(stream))
            # Let the consumer reach its wait for the blocked node's event.
            await asyncio.sleep(0)
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(consumer, timeout=2)
        assert stopped.is_set(), "stream returned before its downstream node was stopped"
    finally:
        release.set()
        if consumer is not None and not consumer.done():
            consumer.cancel()
        await stream.aclose()


@pytest.mark.asyncio
async def test_parallel_lifecycle_tool_calls_sum_children_without_inflating_run_total():
    spawn = AsyncMock(
        return_value=SimpleNamespace(success=True, summary="done", error=None, tool_calls_used=3)
    )
    workflow = WorkflowDefinition(
        name="parallel_usage",
        start_node="before",
        nodes={
            "before": AgentNode(
                id="before", name="before", role="researcher", goal="prepare", next_nodes=["fan"]
            ),
            "fan": ParallelNode(
                id="fan", name="fan", parallel_nodes=["a", "b"], next_nodes=["after"]
            ),
            "a": AgentNode(id="a", name="a", role="researcher", goal="first"),
            "b": AgentNode(id="b", name="b", role="researcher", goal="second"),
            "after": TransformNode(id="after", name="after", transform=dict),
        },
    )
    with patch(
        "victor.agent.subagents.orchestrator.SubAgentOrchestrator",
        return_value=SimpleNamespace(spawn=spawn),
    ):
        executor = StreamingWorkflowExecutor(MagicMock())
        chunks = [chunk async for chunk in executor.astream(workflow)]
        result = await executor.execute(workflow)

    usage = {
        chunk.node_id: chunk.metadata["tool_calls_used"]
        for chunk in chunks
        if chunk.event_type == WorkflowEventType.NODE_COMPLETE
    }
    assert chunks[-1].event_type == WorkflowEventType.WORKFLOW_COMPLETE
    assert usage == {"before": 3, "fan": 6, "after": 0}
    assert result.success
    assert result.total_tool_calls == 9
    assert sum(usage.values()) == result.total_tool_calls


@pytest.mark.asyncio
async def test_engine_yaml_stream_preserves_compiled_node_and_state(monkeypatch):
    graph = StateGraph(dict)
    graph.add_node("answer", lambda state: {**state, "answer": 42})
    graph.add_edge("answer", END)
    graph.set_entry_point("answer")
    cached = CachedCompiledGraph(compiled_graph=graph.compile(), workflow_name="answer")
    engine = WorkflowEngine()
    monkeypatch.setattr(
        engine,
        "_get_unified_compiler",
        lambda: SimpleNamespace(compile_yaml=lambda *args, **kwargs: cached),
    )

    events = [event async for event in engine.stream_yaml("unused.yaml", {"input": 7})]

    assert len(events) == 1
    assert events[0].event_type == "state_update"
    assert events[0].node_id == "answer"
    assert events[0].state_snapshot["input"] == 7
    assert events[0].state_snapshot["answer"] == 42
