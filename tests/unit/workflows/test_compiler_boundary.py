from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from victor.workflows.compiler.boundary import (
    LegacyWorkflowDslCompiler,
    LegacyWorkflowGraphCompiler,
    NativeWorkflowGraphCompiler,
    ParsedWorkflowDefinition,
    WorkflowCompilationRequest,
    WorkflowDefinitionValidator,
    WorkflowParser,
)
from victor.workflows.compiler.workflow_compiler_impl import WorkflowCompilerImpl
from victor.workflows.definition import (
    AgentNode,
    ComputeNode,
    ConditionNode,
    ParallelNode,
    TransformNode,
    WorkflowDefinition,
)
from victor.workflows.executors.factory import NodeExecutorFactory


def _make_workflow(name: str) -> WorkflowDefinition:
    return WorkflowDefinition(
        name=name,
        nodes={
            "start": AgentNode(
                id="start",
                name="Start",
                role="researcher",
                goal="Analyze",
                output_key="result",
            )
        },
        start_node="start",
    )


def _make_condition_workflow(name: str) -> WorkflowDefinition:
    return WorkflowDefinition(
        name=name,
        nodes={
            "start": TransformNode(
                id="start",
                name="Start",
                transform=lambda ctx: {"seed": ctx.get("seed", 0) + 1},
                next_nodes=["decide"],
            ),
            "decide": ConditionNode(
                id="decide",
                name="Decide",
                condition=lambda ctx: "high" if ctx.get("flag") else "low",
                branches={"high": "high", "low": "low"},
            ),
            "high": TransformNode(
                id="high",
                name="High",
                transform=lambda ctx: {"branch": "high"},
            ),
            "low": TransformNode(
                id="low",
                name="Low",
                transform=lambda ctx: {"branch": "low"},
            ),
        },
        start_node="start",
    )


def _make_parallel_workflow(name: str) -> WorkflowDefinition:
    return WorkflowDefinition(
        name=name,
        nodes={
            "start": TransformNode(
                id="start",
                name="Start",
                transform=lambda ctx: {"started": True},
                next_nodes=["parallel"],
            ),
            "worker_a": ComputeNode(
                id="worker_a",
                name="Worker A",
                output_key="a",
            ),
            "worker_b": ComputeNode(
                id="worker_b",
                name="Worker B",
                output_key="b",
            ),
            "parallel": ParallelNode(
                id="parallel",
                name="Parallel",
                parallel_nodes=["worker_a", "worker_b"],
                next_nodes=["finish"],
            ),
            "finish": TransformNode(
                id="finish",
                name="Finish",
                transform=lambda ctx: {"finished": True},
            ),
        },
        start_node="start",
    )


class TestWorkflowParser:
    def test_parser_selects_single_workflow_from_mapping(self) -> None:
        workflow = _make_workflow("alpha")
        loader = Mock()
        loader.load.return_value = {"alpha": workflow}
        parser = WorkflowParser(loader)
        request = WorkflowCompilationRequest(source="workflows:\n  alpha: {}", validate=True)

        parsed = parser.parse(request)

        loader.load.assert_called_once_with(request.source, workflow_name=None)
        assert parsed.workflow is workflow
        assert parsed.workflow_name == "alpha"
        assert parsed.source_path is None

    def test_parser_requires_workflow_name_for_multi_workflow_mapping(self) -> None:
        loader = Mock()
        loader.load.return_value = {
            "alpha": _make_workflow("alpha"),
            "beta": _make_workflow("beta"),
        }
        parser = WorkflowParser(loader)
        request = WorkflowCompilationRequest(
            source="workflows:\n  alpha: {}\n  beta: {}", validate=True
        )

        with pytest.raises(ValueError, match="Multiple workflows found"):
            parser.parse(request)


class TestWorkflowDefinitionValidator:
    def test_validator_raises_for_invalid_workflow(self) -> None:
        invalid_workflow = WorkflowDefinition(name="broken")
        parsed = ParsedWorkflowDefinition(
            request=WorkflowCompilationRequest(source="broken.yaml"),
            workflow=invalid_workflow,
        )
        validator = Mock()

        with pytest.raises(ValueError, match="Workflow must have at least one node"):
            WorkflowDefinitionValidator(validator).validate(parsed)

        validator.validate.assert_not_called()

    def test_validator_normalizes_result_objects(self) -> None:
        workflow = _make_workflow("alpha")
        parsed = ParsedWorkflowDefinition(
            request=WorkflowCompilationRequest(source="workflow.yaml"),
            workflow=workflow,
        )
        result = type("ValidationResult", (), {"is_valid": True, "errors": []})()
        validator = Mock()
        validator.validate.return_value = result

        validated = WorkflowDefinitionValidator(validator).validate(parsed)

        validator.validate.assert_called_once_with(workflow)
        assert validated is parsed


class TestLegacyWorkflowGraphCompiler:
    def test_legacy_graph_compiler_uses_backend_factory(self) -> None:
        workflow = _make_workflow("alpha")
        parsed = ParsedWorkflowDefinition(
            request=WorkflowCompilationRequest(source="workflow.yaml"),
            workflow=workflow,
        )
        compiled = object()
        backend = Mock()
        backend.compile.return_value = compiled
        backend_factory = Mock(return_value=backend)

        result = LegacyWorkflowGraphCompiler(compiler_factory=backend_factory).compile(parsed)

        backend_factory.assert_called_once_with()
        backend.compile.assert_called_once_with(workflow)
        assert result is compiled


class TestLegacyWorkflowDslCompiler:
    def test_legacy_workflow_dsl_compiler_uses_backend_factory(self) -> None:
        graph = Mock(name="workflow_graph")
        compiled = object()
        backend = Mock()
        backend.compile.return_value = compiled
        backend_factory = Mock(return_value=backend)

        result = LegacyWorkflowDslCompiler(compiler_factory=backend_factory).compile(
            graph,
            name="graph-alpha",
        )

        backend_factory.assert_called_once_with()
        backend.compile.assert_called_once_with(graph, "graph-alpha")
        assert result is compiled


class TestNativeWorkflowGraphCompiler:
    def test_native_graph_compiler_matches_legacy_condition_schema(self) -> None:
        workflow = _make_condition_workflow("conditional")
        parsed = ParsedWorkflowDefinition(
            request=WorkflowCompilationRequest(source="workflow.yaml"),
            workflow=workflow,
        )
        native = NativeWorkflowGraphCompiler(node_executor_factory=NodeExecutorFactory())
        legacy = LegacyWorkflowGraphCompiler()

        native_graph = native.compile(parsed)
        legacy_graph = legacy.compile(parsed)

        assert native_graph.get_graph_schema() == legacy_graph.get_graph_schema()

    def test_native_graph_compiler_matches_legacy_parallel_schema(self) -> None:
        workflow = _make_parallel_workflow("parallel")
        parsed = ParsedWorkflowDefinition(
            request=WorkflowCompilationRequest(source="workflow.yaml"),
            workflow=workflow,
        )
        native = NativeWorkflowGraphCompiler(node_executor_factory=NodeExecutorFactory())
        legacy = LegacyWorkflowGraphCompiler()

        native_graph = native.compile(parsed)
        legacy_graph = legacy.compile(parsed)

        assert native_graph.get_graph_schema() == legacy_graph.get_graph_schema()

    def test_native_graph_compiler_matches_legacy_parallel_execution(self) -> None:
        workflow = _make_parallel_workflow("parallel")
        parsed = ParsedWorkflowDefinition(
            request=WorkflowCompilationRequest(source="workflow.yaml"),
            workflow=workflow,
        )
        native = NativeWorkflowGraphCompiler(node_executor_factory=NodeExecutorFactory())
        legacy = LegacyWorkflowGraphCompiler()

        native_result = asyncio.run(native.compile(parsed).invoke({"value": 3}))
        legacy_result = asyncio.run(legacy.compile(parsed).invoke({"value": 3}))

        assert native_result.success is True
        assert legacy_result.success is True
        assert native_result.state["a"] == legacy_result.state["a"]
        assert native_result.state["b"] == legacy_result.state["b"]
        assert native_result.state["finished"] == legacy_result.state["finished"] is True
        assert (
            native_result.state["_parallel_results"].keys()
            == legacy_result.state["_parallel_results"].keys()
        )

    def test_native_graph_compiler_rejects_unregistered_node_types(self) -> None:
        unknown = SimpleNamespace(
            id="custom",
            name="Custom",
            node_type=SimpleNamespace(value="custom_unknown"),
            next_nodes=[],
        )
        workflow = WorkflowDefinition(
            name="unknown",
            nodes={"custom": unknown},
            start_node="custom",
        )
        parsed = ParsedWorkflowDefinition(
            request=WorkflowCompilationRequest(source="workflow.yaml"),
            workflow=workflow,
        )

        native = NativeWorkflowGraphCompiler(node_executor_factory=NodeExecutorFactory())

        with pytest.raises(ValueError, match="Unsupported workflow node type 'custom_unknown'"):
            native.compile(parsed)


class TestWorkflowCompilerFacade:
    def test_workflow_compiler_impl_uses_explicit_boundary_stages(self) -> None:
        parsed = ParsedWorkflowDefinition(
            request=WorkflowCompilationRequest(source="workflow.yaml", workflow_name="alpha"),
            workflow=_make_workflow("alpha"),
        )
        parser = Mock()
        parser.parse.return_value = parsed
        validator = Mock()
        validator.validate.return_value = parsed
        graph_compiler = Mock()
        compiled = object()
        graph_compiler.compile.return_value = compiled

        compiler = WorkflowCompilerImpl(
            yaml_loader=Mock(),
            validator=Mock(),
            node_factory=Mock(),
        )
        # Inject mocks for testing
        compiler._workflow_parser = parser
        compiler._workflow_definition_validator = validator
        compiler._graph_compiler = graph_compiler

        result = compiler.compile("workflow.yaml", workflow_name="alpha", validate=True)

        parser.parse.assert_called_once()
        request = parser.parse.call_args.args[0]
        assert request == WorkflowCompilationRequest(
            source="workflow.yaml",
            workflow_name="alpha",
            validate=True,
        )
        validator.validate.assert_called_once_with(parsed)
        graph_compiler.compile.assert_called_once_with(parsed)
        assert result is compiled

    def test_workflow_compiler_impl_skips_validation_when_disabled(self) -> None:
        parsed = ParsedWorkflowDefinition(
            request=WorkflowCompilationRequest(source="workflow.yaml", validate=False),
            workflow=_make_workflow("alpha"),
        )
        parser = Mock(return_value=parsed)
        parser.parse.return_value = parsed
        validator = Mock()
        graph_compiler = Mock(return_value=object())
        graph_compiler.compile.return_value = object()

        compiler = WorkflowCompilerImpl(
            yaml_loader=Mock(),
            validator=Mock(),
            node_factory=Mock(),
        )
        # Inject mocks for testing
        compiler._workflow_parser = parser
        compiler._workflow_definition_validator = validator
        compiler._graph_compiler = graph_compiler

        compiler.compile("workflow.yaml", validate=False)

        validator.validate.assert_not_called()
        graph_compiler.compile.assert_called_once_with(parsed)

    def test_workflow_compiler_impl_has_expected_interface(self) -> None:
        compiler = WorkflowCompilerImpl(yaml_loader=Mock(), validator=Mock(), node_factory=Mock())

        assert hasattr(compiler, "compile")
        assert callable(compiler.compile)

    def test_workflow_compiler_impl_defaults_to_native_backend(self) -> None:
        workflow = WorkflowDefinition(
            name="alpha",
            nodes={
                "start": TransformNode(
                    id="start",
                    name="Start",
                    transform=lambda ctx: {"ok": True},
                )
            },
            start_node="start",
        )
        loader = Mock()
        loader.load.return_value = workflow
        node_executor_factory = Mock()
        validator = Mock()
        validator.validate.return_value = None

        async def passthrough(state):
            return state

        node_executor_factory.create_executor.return_value = passthrough
        compiler = WorkflowCompilerImpl(
            yaml_loader=loader,
            validator=validator,
            node_factory=node_executor_factory,
        )

        compiled = compiler.compile("workflow.yaml")

        node_executor_factory.create_executor.assert_called_once_with(workflow.nodes["start"])
        assert compiled.get_graph_schema()["entry_point"] == "start"


@pytest.mark.asyncio
@pytest.mark.parametrize("limit,expected_peak", [(1, 1), (2, 2), (None, 2)])
async def test_native_parallel_compiler_honors_concurrency_limit(limit, expected_peak):
    active = 0
    peak = 0

    class AsyncFactory:
        def create_executor(self, node):
            async def execute(state):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                try:
                    await asyncio.sleep(0)
                    return {**state, node.id: True}
                finally:
                    active -= 1

            return execute

    workflow = _make_parallel_workflow("bounded_parallel")
    parsed = ParsedWorkflowDefinition(
        request=WorkflowCompilationRequest(source="test"), workflow=workflow
    )
    compiled = NativeWorkflowGraphCompiler(
        node_executor_factory=AsyncFactory(), max_parallel=limit, enable_checkpointing=False
    ).compile(parsed)
    result = await compiled.invoke({})
    assert result.success is True
    assert result.state["worker_a"] is True
    assert result.state["worker_b"] is True
    assert peak == expected_peak
    assert active == 0


@pytest.mark.parametrize("limit", [0, -1])
def test_native_parallel_compiler_rejects_invalid_concurrency_limit(limit):
    with pytest.raises(ValueError, match="max_parallel must be positive"):
        NativeWorkflowGraphCompiler(node_executor_factory=Mock(), max_parallel=limit)


@pytest.mark.parametrize(
    "left,right,expected", [(float("nan"), float("nan"), True), (float("nan"), 1.0, False)]
)
def test_snapshot_equality_treats_scalar_nan_as_unchanged(left, right, expected):
    from victor.workflows.compiler.boundary import _state_values_equal

    assert _state_values_equal(left, right) is expected
