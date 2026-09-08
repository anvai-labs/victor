"""Tests for WorkflowExecutor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.workflows.context import WorkflowContext, WorkflowResult
from victor.workflows.state_graph_adapter import StateGraphWorkflowExecutor as WorkflowExecutor
from victor_contracts.workflows import ExecutorNodeStatus, NodeResult


@pytest.fixture
def mock_orchestrator():
    orch = MagicMock()
    orch.settings = MagicMock()
    return orch


@pytest.fixture
def executor(mock_orchestrator):
    return WorkflowExecutor(mock_orchestrator, max_parallel=2, default_timeout=60.0)


class TestNodeResult:
    def test_success_property(self):
        r = NodeResult(node_id="n1", status=ExecutorNodeStatus.COMPLETED, output="ok")
        assert r.success is True

    def test_failed_not_success(self):
        r = NodeResult(node_id="n1", status=ExecutorNodeStatus.FAILED, error="err")
        assert r.success is False

    def test_to_dict(self):
        r = NodeResult(node_id="n1", status=ExecutorNodeStatus.COMPLETED, output="data")
        d = r.to_dict()
        assert d["node_id"] == "n1"
        assert d["status"] == "completed"
        assert d["output"] == "data"


class TestWorkflowContext:
    def test_get_set(self):
        ctx = WorkflowContext()
        ctx.set("key", "value")
        assert ctx.get("key") == "value"

    def test_get_default(self):
        ctx = WorkflowContext()
        assert ctx.get("missing", "default") == "default"

    def test_update(self):
        ctx = WorkflowContext()
        ctx.update({"a": 1, "b": 2})
        assert ctx.get("a") == 1
        assert ctx.get("b") == 2

    def test_add_and_get_result(self):
        ctx = WorkflowContext()
        result = NodeResult(node_id="n1", status=ExecutorNodeStatus.COMPLETED, output="x")
        ctx.add_result(result)
        assert ctx.get_result("n1") is result

    def test_has_failures(self):
        ctx = WorkflowContext()
        ctx.add_result(NodeResult(node_id="n1", status=ExecutorNodeStatus.COMPLETED))
        assert ctx.has_failures() is False
        ctx.add_result(NodeResult(node_id="n2", status=ExecutorNodeStatus.FAILED, error="e"))
        assert ctx.has_failures() is True

    def test_get_outputs(self):
        ctx = WorkflowContext()
        ctx.add_result(
            NodeResult(node_id="n1", status=ExecutorNodeStatus.COMPLETED, output="data1")
        )
        ctx.add_result(NodeResult(node_id="n2", status=ExecutorNodeStatus.FAILED, error="e"))
        outputs = ctx.get_outputs()
        assert "n1" in outputs
        assert "n2" not in outputs


class TestWorkflowResult:
    def test_get_output(self):
        ctx = WorkflowContext()
        ctx.add_result(
            NodeResult(node_id="analyze", status=ExecutorNodeStatus.COMPLETED, output="result")
        )
        wr = WorkflowResult(workflow_name="test", success=True, context=ctx)
        assert wr.get_output("analyze") == "result"
        assert wr.get_output("missing") is None

    def test_to_dict(self):
        ctx = WorkflowContext()
        wr = WorkflowResult(workflow_name="wf", success=True, context=ctx, total_duration=1.5)
        d = wr.to_dict()
        assert d["workflow_name"] == "wf"
        assert d["success"] is True
        assert d["total_duration"] == 1.5


class TestWorkflowExecutorInit:
    def test_default_attributes(self, executor):
        assert executor.max_parallel == 2
        assert executor.default_timeout == 60.0

    @pytest.mark.asyncio
    async def test_execute_empty_workflow_raises(self, executor):
        from victor.workflows.definition import WorkflowDefinition

        result = await executor.execute(WorkflowDefinition(name="empty", nodes={}), {})
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_execute_passes_initial_context(self, executor):
        from victor.workflows.definition import WorkflowBuilder

        captured = {}

        def capture(state):
            captured.update(state)
            return {"done": True}

        workflow = WorkflowBuilder("initial_context").add_transform("capture", capture).build()
        result = await executor.execute(workflow, initial_context={"files": ["main.py"]})
        assert result.success is True
        assert captured["files"] == ["main.py"]
        assert result.context.data["done"] is True


class TestWorkflowExecutorChainHandlers:
    @pytest.mark.asyncio
    async def test_execute_chain_handler_uses_asyncio_to_thread_for_sync_invoke(self):
        from types import SimpleNamespace
        from victor.workflows.definition import ComputeNode
        from victor.workflows.executors.compute import ComputeNodeExecutor

        node = ComputeNode(
            id="compute_sync_invoke",
            name="Invoke",
            handler="chain:analysis_chain",
            output_key="chain_output",
            input_mapping={"payload": "value"},
        )
        chain = SimpleNamespace(invoke=MagicMock(return_value={"result": "ok"}))

        async def run_in_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch("victor.framework.chain_registry.create_chain", return_value=chain),
            patch("asyncio.to_thread", side_effect=run_in_thread) as offload,
        ):
            state = await ComputeNodeExecutor().execute(node, {"value": 42})
        assert state["_node_results"][node.id].success is True
        assert state["chain_output"] == {"result": "ok"}
        offload.assert_awaited_once()
        assert offload.await_args.args == (chain.invoke, {"payload": 42})

    @pytest.mark.asyncio
    async def test_execute_chain_handler_uses_asyncio_to_thread_for_sync_callable(self):
        from victor.workflows.definition import ComputeNode
        from victor.workflows.executors.compute import ComputeNodeExecutor

        node = ComputeNode(
            id="compute_sync_callable",
            name="Callable",
            handler="chain:callable_chain",
            output_key="callable_output",
            input_mapping={"payload": "value"},
        )

        def chain(**kwargs):
            return {"seen": kwargs}

        async def run_in_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch("victor.framework.chain_registry.create_chain", return_value=chain),
            patch("asyncio.to_thread", side_effect=run_in_thread) as offload,
        ):
            state = await ComputeNodeExecutor().execute(node, {"value": "repo"})
        assert state["_node_results"][node.id].success is True
        assert state["callable_output"] == {"seen": {"payload": "repo"}}
        offload.assert_awaited_once()
        assert offload.await_args.args[0] is chain
        assert offload.await_args.kwargs == {"payload": "repo"}


# ---------------------------------------------------------------------------
# Phase 3.1 boundary tests: compute registry and SDK type migration
# ---------------------------------------------------------------------------


class TestComputeRegistryBoundary:
    """Enforce that compute handler registry is canonical in compute_registry.py."""

    def test_compute_registry_module_is_canonical_source(self):
        from victor.workflows.compute_registry import (
            ComputeHandler,
            register_compute_handler,
            get_compute_handler,
            list_compute_handlers,
        )

        assert callable(register_compute_handler)
        assert callable(get_compute_handler)
        assert callable(list_compute_handlers)
        assert ComputeHandler is not None

    def test_executor_reexports_from_compute_registry(self):
        from victor.workflows import compute_registry, executor

        assert executor.register_compute_handler is compute_registry.register_compute_handler
        assert executor.get_compute_handler is compute_registry.get_compute_handler
        assert executor.list_compute_handlers is compute_registry.list_compute_handlers
        assert executor.ComputeHandler is compute_registry.ComputeHandler

    def test_shared_registry_dict(self):
        from victor.workflows import compute_registry, executor

        assert executor._compute_handlers is compute_registry._compute_handlers

    def test_register_via_registry_visible_in_executor(self):
        from victor.workflows.compute_registry import register_compute_handler
        from victor.workflows.compute_registry import (
            get_compute_handler,
            list_compute_handlers,
        )

        async def stub_handler(node, context, tool_registry):
            pass

        register_compute_handler("_test_boundary_stub", stub_handler)
        assert "_test_boundary_stub" in list_compute_handlers()
        assert get_compute_handler("_test_boundary_stub") is stub_handler

    def test_benchmark_handlers_use_sdk_for_node_result(self):
        source = open("victor/benchmark/handlers.py").read()
        assert "from victor_contracts.workflows import NodeResult" in source or (
            "from victor_contracts.workflows import" in source and "NodeResult" in source
        )
        assert "from victor.workflows.executor import NodeResult" not in source

    def test_benchmark_handlers_use_compute_registry(self):
        source = open("victor/benchmark/handlers.py").read()
        assert "from victor.workflows.compute_registry import register_compute_handler" in source
        assert "from victor.workflows.executor import register_compute_handler" not in source

    def test_benchmark_escape_hatches_use_sdk_for_node_result(self):
        source = open("victor/benchmark/escape_hatches.py").read()
        assert "from victor.workflows.executor import NodeResult" not in source

    def test_benchmark_escape_hatches_use_compute_registry(self):
        source = open("victor/benchmark/escape_hatches.py").read()
        assert "from victor.workflows.compute_registry import register_compute_handler" in source
        assert "from victor.workflows.executor import register_compute_handler" not in source

    def test_framework_integration_uses_compute_registry(self):
        source = open("victor/framework/framework_integration_registry_service.py").read()
        assert "from victor.workflows.compute_registry import register_compute_handler" in source
        assert "from victor.workflows.executor import register_compute_handler" not in source

    def test_workflow_handlers_use_compute_registry(self):
        source = open("victor/workflows/handlers.py").read()
        assert "from victor.workflows.compute_registry import register_compute_handler" in source
        assert "from victor.workflows.executor import register_compute_handler" not in source


# ---------------------------------------------------------------------------
# Phase 3.2 boundary tests: WorkflowContext/WorkflowResult/TemporalContext migration
# ---------------------------------------------------------------------------


class TestWorkflowContextBoundary:
    """Enforce that WorkflowContext/WorkflowResult/TemporalContext are canonical in context.py."""

    def test_context_module_is_canonical_source(self):
        from victor.workflows.context import (
            WorkflowContext,
            WorkflowResult,
            TemporalContext,
        )

        assert WorkflowContext is not None
        assert WorkflowResult is not None
        assert TemporalContext is not None

    def test_executor_reexports_from_context(self):
        from victor.workflows import context, executor

        assert executor.WorkflowContext is context.WorkflowContext
        assert executor.WorkflowResult is context.WorkflowResult
        assert executor.TemporalContext is context.TemporalContext

    def test_executor_does_not_define_workflow_context(self):
        import ast
        import inspect

        import victor.workflows.executor as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)
        defined_classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        assert "WorkflowContext" not in defined_classes
        assert "WorkflowResult" not in defined_classes
        assert "TemporalContext" not in defined_classes

    def test_temporal_context_preserves_date_range_behavior(self):
        from victor.workflows.context import TemporalContext

        temporal = TemporalContext(
            as_of_date="2024-12-31",
            lookback_periods=2,
            period_type="quarters",
        )

        start_date, end_date = temporal.get_date_range()

        assert end_date == "2024-12-31"
        assert start_date < end_date

    def test_context_module_exports_all_three(self):
        from victor.workflows.context import __all__

        assert "WorkflowContext" in __all__
        assert "WorkflowResult" in __all__
        assert "TemporalContext" in __all__

    def test_from_workflow_context_no_executor_import(self):
        import inspect
        from victor.workflows import context

        source = inspect.getsource(context.from_workflow_context)
        assert "from victor.workflows.executor import" not in source

    def test_to_workflow_context_no_executor_import(self):
        import inspect
        from victor.workflows import context

        source = inspect.getsource(context.to_workflow_context)
        assert "from victor.workflows.executor import" not in source


class TestWorkflowExecutorCompatibilityBoundaries:
    """Enforce type-only callers stay off the legacy executor module."""

    def test_batch_executor_does_not_import_legacy_workflow_executor(self):
        source = open("victor/workflows/batch_executor.py").read()
        assert "from victor.workflows.executor import WorkflowExecutor" not in source

    def test_yaml_coordinator_does_not_directly_import_legacy_executors(self):
        source = open("victor/framework/coordinators/yaml_coordinator.py").read()
        assert "from victor.workflows.executor import WorkflowExecutor" not in source
        assert (
            "from victor.workflows.streaming_executor import StreamingWorkflowExecutor"
            not in source
        )

    def test_workflow_engine_does_not_directly_import_legacy_executors(self):
        source = open("victor/framework/workflow_engine.py").read()
        assert "from victor.workflows.executor import WorkflowExecutor" not in source
        assert (
            "from victor.workflows.streaming_executor import StreamingWorkflowExecutor"
            not in source
        )

    def test_streaming_executor_does_not_subclass_legacy_workflow_executor(self):
        source = open("victor/workflows/streaming_executor.py").read()
        assert "from victor.workflows.executor import WorkflowExecutor" not in source
        assert "class StreamingWorkflowExecutor(WorkflowExecutor)" not in source

    def test_workflow_handlers_use_canonical_result_types(self):
        source = open("victor/workflows/handlers.py").read()
        assert "from victor.workflows.executor import NodeResult" not in source
        assert "from victor.workflows.executor import ExecutorNodeStatus" not in source

    def test_workflow_handlers_use_canonical_compute_registry(self):
        source = open("victor/workflows/executors/compute.py").read()
        assert "from victor.workflows.executor import get_compute_handler" not in source

    def test_node_runners_use_canonical_compute_registry(self):
        source = open("victor/workflows/node_runners.py").read()
        assert "from victor.workflows.executor import get_compute_handler" not in source

    def test_workflow_cache_uses_canonical_result_types(self):
        source = open("victor/workflows/cache.py").read()
        assert "from victor.workflows.executor import NodeResult" not in source
