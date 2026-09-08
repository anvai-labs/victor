"""Canonical streaming lifecycle, cancellation, isolation, and persistence contracts."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.graph import MemoryCheckpointer
from victor.workflows.context import WorkflowContext, WorkflowResult
from victor.workflows.definition import TransformNode, WorkflowBuilder, WorkflowDefinition
from victor.workflows.streaming_executor import StreamingWorkflowExecutor, WorkflowEventType


def linear(effects):
    return (
        WorkflowBuilder("linear")
        .add_transform("a", lambda state: effects.append("a") or {"a": True})
        .add_transform("b", lambda state: effects.append("b") or {"b": True})
        .build()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary,expected", [("workflow_start", []), ("node_start", []), ("node_complete", ["a"])]
)
async def test_cancellation_at_delivered_boundary_prevents_further_effects(boundary, expected):
    effects = []
    executor = StreamingWorkflowExecutor()
    chunks = []
    async for chunk in executor.astream(linear(effects)):
        chunks.append(chunk)
        if chunk.event_type.value == boundary:
            assert executor.cancel_workflow(chunk.workflow_id)
    assert effects == expected
    assert chunks[-1].is_final
    assert chunks[-1].metadata["success"] is False
    assert chunks[-1].error == "Workflow cancelled"
    assert executor.get_active_workflows() == []


@pytest.mark.asyncio
async def test_close_at_node_start_awaits_runner_and_prevents_effects():
    effects = []
    executor = StreamingWorkflowExecutor()
    stream = executor.astream(linear(effects))
    assert (await anext(stream)).event_type == WorkflowEventType.WORKFLOW_START
    assert (await anext(stream)).event_type == WorkflowEventType.NODE_START
    await stream.aclose()
    await asyncio.sleep(0)
    assert effects == []
    assert executor.get_active_workflows() == []


@pytest.mark.asyncio
async def test_two_concurrent_streams_keep_cancellation_and_ids_isolated():
    executor = StreamingWorkflowExecutor()
    effects_a, effects_b = [], []

    async def collect(workflow, cancel):
        chunks = []
        async for chunk in executor.astream(workflow):
            chunks.append(chunk)
            if cancel and chunk.event_type == WorkflowEventType.NODE_COMPLETE:
                executor.cancel_workflow(chunk.workflow_id)
        return chunks

    cancelled, completed = await asyncio.gather(
        collect(linear(effects_a), True), collect(linear(effects_b), False)
    )
    assert effects_a == ["a"]
    assert effects_b == ["a", "b"]
    assert cancelled[-1].metadata["success"] is False
    assert completed[-1].metadata["success"] is True
    assert {c.workflow_id for c in cancelled}.isdisjoint({c.workflow_id for c in completed})
    assert executor.get_active_workflows() == []


@pytest.mark.asyncio
async def test_real_async_node_timeout_reports_node_and_terminal_errors():
    entered = asyncio.Event()
    stopped = asyncio.Event()

    async def spawn(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    workflow = WorkflowBuilder("timeout").add_agent("agent", "executor", "work").build()
    with patch("victor.agent.subagents.SubAgentOrchestrator.spawn", side_effect=spawn):
        chunks = [
            c async for c in StreamingWorkflowExecutor(MagicMock()).astream(workflow, timeout=0.1)
        ]
    assert entered.is_set() and stopped.is_set()
    assert any(c.event_type == WorkflowEventType.NODE_ERROR for c in chunks)
    assert chunks[-1].event_type == WorkflowEventType.WORKFLOW_ERROR
    assert chunks[-1].metadata["success"] is False
    assert "timeout" in chunks[-1].error.lower()


@pytest.mark.asyncio
async def test_external_consumer_cancellation_propagates_and_cleans_up():
    entered = asyncio.Event()
    stopped = asyncio.Event()

    async def spawn(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    executor = StreamingWorkflowExecutor(MagicMock())
    workflow = WorkflowBuilder("cancel").add_agent("agent", "executor", "work").build()

    async def consume():
        return [c async for c in executor.astream(workflow)]

    with patch("victor.agent.subagents.SubAgentOrchestrator.spawn", side_effect=spawn):
        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), 2)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
    assert stopped.is_set()
    assert executor.get_active_workflows() == []


@pytest.mark.asyncio
async def test_checkpoint_failure_never_announces_node_completion():
    checkpointer = MagicMock(
        load=AsyncMock(return_value=None), save=AsyncMock(side_effect=OSError("disk full"))
    )
    executor = StreamingWorkflowExecutor(checkpointer=checkpointer)
    chunks = [c async for c in executor.astream(linear([]))]
    assert not any(c.event_type == WorkflowEventType.NODE_COMPLETE for c in chunks)
    assert any(c.event_type == WorkflowEventType.NODE_ERROR for c in chunks)
    assert chunks[-1].error == "disk full"
    assert chunks[-1].metadata["success"] is False


@pytest.mark.asyncio
async def test_supplied_checkpointer_retains_dag_frontier_for_terminal_resume():
    checkpointer = MemoryCheckpointer()
    effects = []
    nodes = {
        name: TransformNode(
            id=name,
            name=name,
            transform=lambda state, name=name: effects.append(name) or state,
            next_nodes=targets,
        )
        for name, targets in {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}.items()
    }
    workflow = WorkflowDefinition(name="dag", nodes=nodes, start_node="a")
    executor = StreamingWorkflowExecutor(checkpointer=checkpointer)
    first = [c async for c in executor.astream(workflow, thread_id="persisted")]
    second = [c async for c in executor.astream(workflow, thread_id="persisted")]
    assert first[-1].metadata["success"] is True
    assert effects == ["a", "b", "c", "d"]
    assert [c.event_type for c in second] == [
        WorkflowEventType.WORKFLOW_START,
        WorkflowEventType.WORKFLOW_COMPLETE,
    ]
    checkpoint = await checkpointer.load("persisted")
    assert checkpoint.metadata["sequential_frontier"]["pending_nodes"] == []


@pytest.mark.asyncio
async def test_paused_result_keeps_metadata_and_does_not_claim_completion():
    result = WorkflowResult(
        "paused", True, WorkflowContext(), interrupted=True, interrupt_node="approval"
    )
    executor = StreamingWorkflowExecutor()
    with patch.object(executor._runtime, "execute", AsyncMock(return_value=result)):
        chunks = [c async for c in executor.astream(linear([]))]
    final = chunks[-1]
    assert final.is_final and final.event_type == WorkflowEventType.WORKFLOW_PAUSED
    assert final.metadata["status"] == "paused"
    assert final.metadata["interrupted"] is True
    assert final.metadata["interrupt_node"] == "approval"
    assert final.progress == 0


@pytest.mark.asyncio
async def test_subscriber_can_unsubscribe_without_skipping_other_subscriber():
    executor = StreamingWorkflowExecutor()
    seen = []
    unsubscribe = executor.subscribe([WorkflowEventType.NODE_START], lambda chunk: unsubscribe())
    executor.subscribe([WorkflowEventType.NODE_START], lambda chunk: seen.append(chunk.node_id))
    async for _ in executor.astream(linear([])):
        pass
    assert seen == ["a", "b"]


@pytest.mark.asyncio
async def test_yaml_fallback_streams_real_definition_and_surfaces_failure():
    from victor.framework.coordinators.yaml_coordinator import YAMLWorkflowCoordinator

    coordinator = YAMLWorkflowCoordinator(use_unified_compiler=False)
    with patch.object(coordinator, "load_workflow", return_value=linear([])):
        events = [
            event async for event in coordinator.stream("unused.yaml", initial_state={"input": 7})
        ]
    assert [event.node_id for event in events] == ["a", "b"]
    assert events[-1].state_snapshot["input"] == 7

    def fail(state):
        raise ValueError("broken transform")

    failing = WorkflowBuilder("bad").add_transform("bad", fail).build()
    with patch.object(coordinator, "load_workflow", return_value=failing):
        events = [event async for event in coordinator.stream("unused.yaml")]
    assert events[-1].event_type == "error"
    assert "broken transform" in events[-1].data["error"]


@pytest.mark.asyncio
async def test_parallel_partial_failure_respects_successful_any_join():
    from victor.workflows.definition import ParallelNode

    def fail(state):
        raise ValueError("optional branch failed")

    workflow = WorkflowDefinition(
        name="any-join",
        nodes={
            "group": ParallelNode(
                id="group", name="group", parallel_nodes=["good", "bad"], join_strategy="any"
            ),
            "good": TransformNode(id="good", name="good", transform=lambda state: {"value": 7}),
            "bad": TransformNode(id="bad", name="bad", transform=fail),
        },
        start_node="group",
    )
    chunks = [chunk async for chunk in StreamingWorkflowExecutor().astream(workflow)]
    assert chunks[-1].metadata["success"] is True
    assert chunks[-1].event_type == WorkflowEventType.WORKFLOW_COMPLETE
    assert [c.node_id for c in chunks if c.event_type == WorkflowEventType.NODE_COMPLETE] == [
        "group"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["definition", "continue_on_failure"])
async def test_invalid_execution_returns_one_failed_terminal_chunk_without_effects(invalid):
    effects = []
    workflow = linear(effects)
    if invalid == "definition":
        workflow.start_node = "missing"
    else:
        workflow.metadata["continue_on_failure"] = True
    chunks = [chunk async for chunk in StreamingWorkflowExecutor().astream(workflow)]
    assert effects == []
    assert chunks[-1].event_type == WorkflowEventType.WORKFLOW_ERROR
    assert chunks[-1].error
    assert sum(c.is_final for c in chunks) == 1


def test_streaming_cache_configuration_is_rejected_instead_of_ignored():
    with pytest.raises(ValueError, match="Node-result caching is unsupported"):
        StreamingWorkflowExecutor(cache=object())


@pytest.mark.asyncio
async def test_execute_context_metadata_is_per_run_and_keeps_resume_thread():
    executor = StreamingWorkflowExecutor()
    workflow = linear([])
    first, second = await asyncio.gather(
        executor.execute(workflow, thread_id="same-thread"),
        executor.execute(workflow, thread_id="same-thread"),
    )
    assert first.context.metadata["workflow_name"] == "linear"
    assert (
        first.context.metadata["thread_id"] == second.context.metadata["thread_id"] == "same-thread"
    )
    assert first.context.metadata["execution_id"] != second.context.metadata["execution_id"]
    implicit = await executor.execute(workflow)
    assert implicit.context.metadata["execution_id"] == implicit.context.metadata["thread_id"]
