"""Regression coverage for the ADR-030 runtime construction seam."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.framework.graph import MemoryCheckpointer, StateGraph
from victor.workflows.context import WorkflowContext
from victor.workflows.definition import TransformNode, WorkflowDefinition
from victor.workflows.runtime_executor_factory import create_legacy_workflow_executor
from victor.workflows.runtime_types import GraphNodeResult
from victor.workflows.state_graph_adapter import StateGraphWorkflowExecutor, to_workflow_result
from victor.workflows.unified_executor import ExecutorResult


def workflow(transform=lambda state: {"answer": state["input"] * 2}):
    return WorkflowDefinition(
        name="adapter",
        start_node="transform",
        nodes={"transform": TransformNode(id="transform", name="Transform", transform=transform)},
    )


async def test_seam_runs_definition_and_preserves_results():
    executor = create_legacy_workflow_executor()
    assert isinstance(executor, StateGraphWorkflowExecutor)
    initial = {"input": 4}
    result = await executor.execute(workflow(), initial_context=initial)
    assert result.success, result.error
    assert result.final_state == {"input": 4, "answer": 8}
    assert result.nodes_executed == ["transform"]
    assert result.context.node_results["transform"].success
    assert result.to_dict()["node_results"]["transform"]["status"] == "completed"
    assert initial == {"input": 4}


async def test_failed_node_result_survives_conversion():
    def fail(state):
        raise ValueError("invalid input")

    result = await create_legacy_workflow_executor().execute(workflow(fail), {"input": 4})
    assert not result.success
    assert "invalid input" in result.error
    assert not result.context.node_results["transform"].success
    assert result.context.has_failures()


def test_conversion_preserves_pause_and_tool_accounting():
    result = to_workflow_result(
        "paused",
        ExecutorResult(
            success=True,
            state={"answer": 8},
            interrupted=True,
            interrupt_node="review",
            nodes_executed=["agent"],
            duration_seconds=2,
            node_results={
                "agent": GraphNodeResult("agent", True, output="done", tool_calls_used=3)
            },
        ),
    )
    assert result.interrupted and result.interrupt_node == "review"
    assert result.total_tool_calls == 3
    assert result.get_output("agent") == "done"
    assert result.total_duration == 2
    assert result.to_dict()["interrupted"] is True


async def test_compiled_graph_honors_initial_context_alias():
    graph = StateGraph(dict)
    graph.add_node("double", lambda state: {**state, "answer": state["input"] * 2})
    graph.set_entry_point("double")
    graph.set_finish_point("double")
    result = await create_legacy_workflow_executor().execute(
        graph.compile(), {"input": 1}, initial_context={"input": 5}
    )
    assert result.success and result.state["answer"] == 10


async def test_invalid_context_as_definition_fails_loudly():
    with pytest.raises(TypeError, match="WorkflowDefinition"):
        await create_legacy_workflow_executor().execute(WorkflowContext())


async def test_explicit_checkpoint_id_is_not_silently_ignored():
    with pytest.raises(ValueError, match="thread_id"):
        await create_legacy_workflow_executor().execute(workflow(), checkpoint="old-bfs-id")


async def test_graph_checkpoint_is_saved():
    store = MemoryCheckpointer()
    executor = StateGraphWorkflowExecutor(checkpointer=store)
    result = await executor.execute(workflow(), {"input": 4}, thread_id="run")
    assert result.success, result.error
    saved = await store.load("run")
    assert saved is not None and saved.node_id == "transform"


async def test_batch_invocations_do_not_share_configuration(monkeypatch):
    seen = []

    async def execute(runtime, definition, state, **kwargs):
        await asyncio.sleep(0)
        seen.append((state["input"], runtime.config.timeout, runtime.config.max_iterations))
        return ExecutorResult(success=True, state=state)

    monkeypatch.setattr("victor.workflows.state_graph_adapter.StateGraphExecutor.execute", execute)
    executor = StateGraphWorkflowExecutor()
    first = workflow()
    first.max_iterations = 7
    second = workflow()
    second.max_iterations = 13
    await asyncio.gather(
        executor.execute(first, {"input": 1}, timeout=1),
        executor.execute(second, {"input": 2}, timeout=2),
    )
    assert sorted(seen) == [(1, 1, 7), (2, 2, 13)]


async def test_empty_initial_context_overrides_positional_state():
    result = await StateGraphWorkflowExecutor().execute(
        workflow(lambda state: {"empty": "input" not in state}),
        {"input": 4},
        initial_context={},
    )
    assert result.success and result.final_state == {"empty": True}


def test_unsupported_node_cache_is_rejected():
    with pytest.raises(ValueError, match="caching"):
        StateGraphWorkflowExecutor(cache=MagicMock())


async def test_continue_on_failure_rejected_before_side_effects():
    transform = MagicMock(return_value={"ran": True})
    definition = workflow(transform)
    definition.metadata["continue_on_failure"] = True
    with pytest.raises(ValueError, match="continue_on_failure"):
        await StateGraphWorkflowExecutor().execute(definition)
    transform.assert_not_called()


@pytest.mark.parametrize("success", [True, False])
async def test_terminal_agent_preserves_flat_state_and_result(monkeypatch, success):
    from victor.workflows.definition import AgentNode

    spawn = AsyncMock(
        return_value=MagicMock(
            success=success,
            summary="complete",
            error=None if success else "failed",
            tool_calls_used=2,
        )
    )
    monkeypatch.setattr("victor.agent.subagents.orchestrator.SubAgentOrchestrator.spawn", spawn)
    node = AgentNode(id="agent", name="Agent", goal="Do work", output_key="answer")
    definition = WorkflowDefinition(name="terminal", start_node="agent", nodes={"agent": node})
    result = await StateGraphWorkflowExecutor(MagicMock()).execute(definition, {"input": 4})
    assert result.success is success
    assert result.final_state["input"] == 4
    assert "data" not in result.final_state
    assert result.context.node_results["agent"].success is success
    if success:
        assert result.final_state["answer"] == "complete"
        assert result.total_tool_calls == 2
    assert node.timeout_seconds is None
    assert spawn.await_args.kwargs["timeout_seconds"] == 300


async def test_service_provider_pool_does_not_fabricate_agent_success():
    from victor.core.container import ServiceContainer
    from victor.workflows.definition import AgentNode
    from victor.workflows.orchestrator_pool import OrchestratorPool
    from victor.workflows.services.workflow_service_provider import WorkflowServiceProvider

    container = ServiceContainer()
    pool = OrchestratorPool(MagicMock())
    container.register_instance(OrchestratorPool, pool)
    provider = WorkflowServiceProvider(MagicMock())
    provider.container = container
    executor = provider._create_workflow_executor()
    node = AgentNode(id="agent", name="Agent", goal="Run work")
    result = await executor.execute(
        WorkflowDefinition(name="pool", start_node="agent", nodes={"agent": node})
    )
    assert not result.success
    assert "orchestrator" in result.error.lower()


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_parallel_limit_rejected(limit):
    with pytest.raises(ValueError, match="max_parallel"):
        StateGraphWorkflowExecutor(max_parallel=limit)


async def test_batch_executes_real_adapter_with_successes_and_failure():
    from victor.workflows.batch_executor import BatchConfig, BatchWorkflowExecutor

    def transform(state):
        if state["input"] < 0:
            raise ValueError("negative input")
        return {"answer": state["input"] * 2}

    inputs = [{"input": 1}, {"input": -1}, {"input": 3}]
    executor = BatchWorkflowExecutor(
        create_legacy_workflow_executor(), BatchConfig(max_concurrent=2)
    )
    result = await executor.execute_batch(workflow(transform), inputs)
    assert result.total_successful == 2
    assert result.total_failed == 1
    assert [item.result.final_state["answer"] for item in result.items if item.result.success] == [
        2,
        6,
    ]
    assert "negative input" in result.items[1].error
    assert inputs == [{"input": 1}, {"input": -1}, {"input": 3}]
