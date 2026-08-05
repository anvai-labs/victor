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

"""Tests for YAMLToStateGraphCompiler."""

import asyncio
import logging
import pytest
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from victor.workflows.executors.registry import (
    clear_registered_workflow_node_executors,
    register_workflow_node_executor,
)


@pytest.fixture(autouse=True)
def suppress_compiler_logging(caplog):
    """Suppress expected warnings during edge-case tests."""
    # Set workflow compiler logging to ERROR to reduce noise
    # from intentional edge-case testing (no orchestrator, unknown branch, etc.)
    logging.getLogger("victor.workflows.yaml_to_graph_compiler").setLevel(logging.CRITICAL)


from victor.workflows.definition import (
    AgentNode,
    ComputeNode,
    ConditionNode,
    ParallelNode,
    TaskConstraints,
    TransformNode,
    WorkflowBuilder,
    WorkflowDefinition,
)
from victor.workflows.yaml_to_graph_compiler import (
    CompilerConfig,
    ConditionEvaluator,
    GraphNodeResult,
    NodeExecutorFactory,
    WorkflowState,
    YAMLToStateGraphCompiler,
    compile_yaml_workflow,
    execute_yaml_workflow,
)
from victor.framework.graph import END, MemoryCheckpointer


class TestWorkflowState:
    """Tests for WorkflowState TypedDict."""

    def test_workflow_state_creation(self):
        """Test creating a WorkflowState."""
        state: WorkflowState = {
            "_workflow_id": "test-123",
            "_current_node": "start",
            "_node_results": {},
            "_error": None,
            "_iteration": 0,
            "_parallel_results": {},
            "_hitl_pending": False,
            "_hitl_response": None,
            "custom_data": "value",
        }

        assert state["_workflow_id"] == "test-123"
        assert state["custom_data"] == "value"

    def test_workflow_state_with_node_results(self):
        """Test WorkflowState with node results."""
        state: WorkflowState = {
            "_node_results": {
                "node1": GraphNodeResult(
                    node_id="node1",
                    success=True,
                    output={"key": "value"},
                )
            }
        }

        assert state["_node_results"]["node1"].success is True


class TestGraphNodeResult:
    """Tests for GraphNodeResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful result."""
        result = GraphNodeResult(
            node_id="test_node",
            success=True,
            output={"data": "value"},
            duration_seconds=1.5,
            tool_calls_used=3,
        )

        assert result.node_id == "test_node"
        assert result.success is True
        assert result.output == {"data": "value"}
        assert result.error is None
        assert result.duration_seconds == 1.5
        assert result.tool_calls_used == 3

    def test_failed_result(self):
        """Test creating a failed result."""
        result = GraphNodeResult(
            node_id="test_node",
            success=False,
            error="Something went wrong",
        )

        assert result.success is False
        assert result.error == "Something went wrong"


class TestNodeExecutorFactory:
    """Tests for NodeExecutorFactory."""

    def test_create_agent_executor(self):
        """Test creating an agent node executor."""
        factory = NodeExecutorFactory()
        node = AgentNode(
            id="analyze",
            name="Analyze Data",
            role="researcher",
            goal="Analyze the input data",
            tool_budget=10,
        )

        executor = factory.create_executor(node)
        assert callable(executor)

    def test_create_compute_executor(self):
        """Test creating a compute node executor."""
        factory = NodeExecutorFactory()
        node = ComputeNode(
            id="fetch",
            name="Fetch Data",
            tools=["read_file"],
            constraints=TaskConstraints(llm_allowed=False),
        )

        executor = factory.create_executor(node)
        assert callable(executor)

    def test_create_transform_executor(self):
        """Test creating a transform node executor."""
        factory = NodeExecutorFactory()
        node = TransformNode(
            id="transform",
            name="Transform",
            transform=lambda ctx: {**ctx, "transformed": True},
        )

        executor = factory.create_executor(node)
        assert callable(executor)

    def test_create_parallel_executor(self):
        """Test creating a parallel node executor."""
        factory = NodeExecutorFactory()
        node = ParallelNode(
            id="parallel",
            name="Parallel",
            parallel_nodes=["node1", "node2"],
        )

        executor = factory.create_executor(node)
        assert callable(executor)

    def test_create_condition_passthrough(self):
        """Test that condition nodes get passthrough executors."""
        factory = NodeExecutorFactory()
        node = ConditionNode(
            id="decide",
            name="Decision",
            condition=lambda ctx: "yes" if ctx.get("flag") else "no",
            branches={"yes": "next1", "no": "next2"},
        )

        executor = factory.create_executor(node)
        assert callable(executor)

    def test_factory_exposes_profile_aware_orchestrator_pool(self):
        """Compatibility context should preserve orchestrator-pool lookup."""
        default_orchestrator = object()
        profile_orchestrator = object()
        factory = NodeExecutorFactory(
            orchestrator=default_orchestrator,
            orchestrators={"analysis": profile_orchestrator},
        )

        context = factory._resolve_execution_context()

        assert context.orchestrator is default_orchestrator
        assert context.orchestrator_pool.get_default_orchestrator() is default_orchestrator
        assert context.orchestrator_pool.get_orchestrator("analysis") is profile_orchestrator
        assert context.orchestrator_pool.get_orchestrator("missing") is default_orchestrator

    @pytest.mark.asyncio
    async def test_agent_executor_without_orchestrator(self):
        """Test agent executor runs in placeholder mode without orchestrator."""
        factory = NodeExecutorFactory()
        node = AgentNode(
            id="analyze",
            name="Analyze",
            role="researcher",
            goal="Test goal",
        )

        executor = factory.create_executor(node)
        state: WorkflowState = {"input_data": "test"}

        result = await executor(state)

        assert "analyze" in result
        assert result["analyze"]["status"] == "placeholder"
        assert result["_node_results"]["analyze"].success is True

    @pytest.mark.asyncio
    async def test_transform_executor_execution(self):
        """Test transform executor actually transforms state."""
        factory = NodeExecutorFactory()

        def add_computed(ctx):
            return {"computed": ctx.get("input", 0) * 2}

        node = TransformNode(
            id="double",
            name="Double",
            transform=add_computed,
        )

        executor = factory.create_executor(node)
        state: WorkflowState = {"input": 5}

        result = await executor(state)

        assert result["computed"] == 10
        assert result["_node_results"]["double"].success is True

    @pytest.mark.asyncio
    async def test_compute_executor_without_tools(self):
        """Test compute executor handles missing tools gracefully."""
        factory = NodeExecutorFactory()
        node = ComputeNode(
            id="compute",
            name="Compute",
            tools=["nonexistent_tool"],
        )

        executor = factory.create_executor(node)
        state: WorkflowState = {}

        result = await executor(state)

        assert "compute" in result
        assert result["_node_results"]["compute"].success is True

    @pytest.mark.asyncio
    async def test_factory_loads_registered_custom_node_executors(self):
        """Test YAML compiler factory uses the shared custom executor registry."""

        class CustomExecutor:
            def __init__(self, context=None):
                self.context = context

            async def execute(self, node, state):
                state = dict(state)
                state["custom"] = node.id
                return state

        class CustomNode:
            def __init__(self):
                self.id = "custom"
                self.node_type = "custom_plugin"

        try:
            clear_registered_workflow_node_executors()
            register_workflow_node_executor("custom_plugin", CustomExecutor)
            factory = NodeExecutorFactory()
            executor = factory.create_executor(CustomNode())

            result = await executor({})

            assert factory.supports_node_type("custom_plugin") is True
            assert result["custom"] == "custom"
        finally:
            clear_registered_workflow_node_executors()


class TestConditionEvaluator:
    """Tests for ConditionEvaluator."""

    def test_create_router_simple_condition(self):
        """Test creating a router for a simple condition.

        Note: The router returns BRANCH NAMES, not node IDs.
        The StateGraph Edge.get_target() uses the branch name to look up
        the actual target node from the branches dict.
        """
        node = ConditionNode(
            id="check",
            name="Check Flag",
            condition=lambda ctx: "yes" if ctx.get("flag") else "no",
            branches={"yes": "node_a", "no": "node_b"},
        )

        router = ConditionEvaluator.create_router(node)

        # Router returns branch names, not node IDs
        assert router({"flag": True}) == "yes"
        assert router({"flag": False}) == "no"

    def test_router_with_default_branch(self):
        """Test router uses default branch for unknown values."""
        node = ConditionNode(
            id="check",
            name="Check",
            condition=lambda ctx: ctx.get("status", "unknown"),
            branches={"success": "node_a", "default": "node_b"},
        )

        router = ConditionEvaluator.create_router(node)

        assert router({"status": "success"}) == "success"
        assert router({"status": "failure"}) == "default"  # Falls to default
        assert router({}) == "default"  # Falls to default

    def test_router_returns_end_when_no_match(self):
        """Test router returns __END__ when no branch matches and no default."""
        node = ConditionNode(
            id="check",
            name="Check",
            condition=lambda ctx: "unknown",
            branches={"yes": "node_a", "no": "node_b"},
        )

        router = ConditionEvaluator.create_router(node)

        # Returns a special marker that won't match any branch
        assert router({}) == "__END__"

    def test_router_handles_exception_with_default(self):
        """Test router falls to default on exception."""

        def failing_condition(ctx):
            raise ValueError("Test error")

        node = ConditionNode(
            id="check",
            name="Check",
            condition=failing_condition,
            branches={"yes": "node_a", "default": "fallback"},
        )

        router = ConditionEvaluator.create_router(node)

        # Should fall to "default" branch on exception
        assert router({}) == "default"


class TestCompilerConfig:
    """Tests for CompilerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CompilerConfig()

        assert config.max_iterations == 25
        assert config.timeout is None
        assert config.enable_checkpointing is True
        assert config.checkpointer is None
        assert config.interrupt_on_hitl is True

    def test_custom_config(self):
        """Test custom configuration."""
        checkpointer = MemoryCheckpointer()
        config = CompilerConfig(
            max_iterations=50,
            timeout=300.0,
            enable_checkpointing=True,
            checkpointer=checkpointer,
            interrupt_on_hitl=False,
        )

        assert config.max_iterations == 50
        assert config.timeout == 300.0
        assert config.checkpointer is checkpointer
        assert config.interrupt_on_hitl is False


class TestYAMLToStateGraphCompiler:
    """Tests for YAMLToStateGraphCompiler."""

    def test_compiler_initialization(self):
        """Test compiler initialization."""
        compiler = YAMLToStateGraphCompiler()

        assert compiler.orchestrator is None
        assert compiler.config is not None
        assert compiler.config.max_iterations == 25

    def test_compiler_with_custom_config(self):
        """Test compiler with custom configuration."""
        config = CompilerConfig(max_iterations=100)
        compiler = YAMLToStateGraphCompiler(config=config)

        assert compiler.config.max_iterations == 100

    def test_compile_uses_shared_native_backend(self, monkeypatch):
        """Compile should route through the shared native workflow backend."""
        workflow = WorkflowBuilder("test").add_transform("step", lambda ctx: ctx).build()
        checkpointer = MemoryCheckpointer()
        config = CompilerConfig(
            max_iterations=77,
            timeout=33.0,
            checkpointer=checkpointer,
            interrupt_on_hitl=False,
        )
        captured: Dict[str, Any] = {}

        class FakeNativeWorkflowGraphCompiler:
            def __init__(
                self,
                node_executor_factory,
                *,
                checkpointer_factory=None,
                enable_checkpointing=True,
                interrupt_on_hitl=True,
            ):
                captured["node_executor_factory"] = node_executor_factory
                captured["checkpointer_factory"] = checkpointer_factory
                captured["enable_checkpointing"] = enable_checkpointing
                captured["interrupt_on_hitl"] = interrupt_on_hitl

            def compile(self, parsed):
                captured["parsed"] = parsed
                return "compiled"

        monkeypatch.setattr(
            "victor.workflows.yaml_to_graph_compiler.NativeWorkflowGraphCompiler",
            FakeNativeWorkflowGraphCompiler,
        )

        compiler = YAMLToStateGraphCompiler(config=config)
        compiled = compiler.compile(workflow)

        assert compiled == "compiled"
        assert captured["enable_checkpointing"] is True
        assert captured["interrupt_on_hitl"] is False
        assert captured["checkpointer_factory"]() is checkpointer
        assert captured["parsed"].workflow.name == "test"
        assert captured["parsed"].workflow.max_iterations == 77
        assert captured["parsed"].workflow.max_execution_timeout_seconds == 33.0

    def test_compile_simple_workflow(self):
        """Test compiling a simple linear workflow."""
        workflow = (
            WorkflowBuilder("test")
            .add_agent("step1", "researcher", "Do step 1")
            .add_agent("step2", "executor", "Do step 2")
            .build()
        )

        compiler = YAMLToStateGraphCompiler()
        compiled = compiler.compile(workflow)

        assert compiled is not None

    def test_compile_workflow_with_condition(self):
        """Test compiling a workflow with condition node."""
        workflow = (
            WorkflowBuilder("test")
            .add_agent("analyze", "researcher", "Analyze")
            .add_condition(
                "decide",
                lambda ctx: "yes" if ctx.get("success") else "no",
                {"yes": "proceed", "no": "retry"},
            )
            .add_agent("proceed", "executor", "Proceed", next_nodes=[])
            .add_agent("retry", "executor", "Retry", next_nodes=["analyze"])
            .build()
        )

        compiler = YAMLToStateGraphCompiler()
        compiled = compiler.compile(workflow)

        assert compiled is not None

    def test_compile_invalid_workflow_raises(self):
        """Test that compiling invalid workflow raises ValueError."""
        # Create workflow with validation errors
        workflow = WorkflowDefinition(
            name="invalid",
            description="Invalid workflow",
            nodes={},  # Empty nodes
            start_node="nonexistent",
        )

        compiler = YAMLToStateGraphCompiler()

        with pytest.raises(ValueError) as exc_info:
            compiler.compile(workflow)

        assert "Invalid workflow" in str(exc_info.value)

    def test_compile_workflow_with_parallel(self):
        """Test compiling a workflow with parallel node."""
        workflow = WorkflowDefinition(
            name="parallel_test",
            description="Test parallel execution",
            nodes={
                "start": AgentNode(
                    id="start",
                    name="Start",
                    role="researcher",
                    goal="Initialize",
                    next_nodes=["parallel"],
                ),
                "parallel": ParallelNode(
                    id="parallel",
                    name="Parallel",
                    parallel_nodes=["task_a", "task_b"],
                    next_nodes=["end"],
                ),
                "task_a": AgentNode(
                    id="task_a",
                    name="Task A",
                    role="executor",
                    goal="Do A",
                ),
                "task_b": AgentNode(
                    id="task_b",
                    name="Task B",
                    role="executor",
                    goal="Do B",
                ),
                "end": TransformNode(
                    id="end",
                    name="End",
                    transform=lambda ctx: ctx,
                ),
            },
            start_node="start",
        )

        compiler = YAMLToStateGraphCompiler()
        compiled = compiler.compile(workflow)

        assert compiled is not None

    @pytest.mark.asyncio
    async def test_compile_and_execute(self):
        """Test compile_and_execute convenience method."""
        workflow = (
            WorkflowBuilder("simple")
            .add_transform("init", lambda ctx: {**ctx, "initialized": True})
            .add_transform("process", lambda ctx: {**ctx, "processed": True})
            .build()
        )

        compiler = YAMLToStateGraphCompiler()
        result = await compiler.compile_and_execute(
            workflow,
            initial_state={"input": "test"},
        )

        assert result.success is True
        assert result.state.get("initialized") is True
        assert result.state.get("processed") is True

    @pytest.mark.asyncio
    async def test_execute_with_checkpointing(self):
        """Test execution with checkpointing enabled."""
        workflow = (
            WorkflowBuilder("checkpoint_test")
            .add_transform("step1", lambda ctx: {**ctx, "step1": True})
            .add_transform("step2", lambda ctx: {**ctx, "step2": True})
            .build()
        )

        checkpointer = MemoryCheckpointer()
        config = CompilerConfig(
            enable_checkpointing=True,
            checkpointer=checkpointer,
        )

        compiler = YAMLToStateGraphCompiler(config=config)
        result = await compiler.compile_and_execute(
            workflow,
            initial_state={"input": "test"},
            thread_id="test-thread-123",
        )

        assert result.success is True

        # Verify checkpoint was saved
        checkpoints = await checkpointer.list("test-thread-123")
        assert len(checkpoints) > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compile_yaml_workflow(self):
        """Test compile_yaml_workflow function."""
        workflow = WorkflowBuilder("test").add_transform("step", lambda ctx: ctx).build()

        compiled = compile_yaml_workflow(workflow)
        assert compiled is not None

    def test_compile_yaml_workflow_passes_keyword_dependencies(self, monkeypatch):
        """Test compile helper preserves tool registry and config keyword binding."""
        workflow = WorkflowBuilder("test").add_transform("step", lambda ctx: ctx).build()
        orchestrator = object()
        tool_registry = object()
        config = CompilerConfig(max_iterations=77)
        captured = {}

        class FakeCompiler:
            def __init__(
                self,
                orchestrator=None,
                orchestrators=None,
                tool_registry=None,
                config=None,
            ):
                captured["orchestrator"] = orchestrator
                captured["orchestrators"] = orchestrators
                captured["tool_registry"] = tool_registry
                captured["config"] = config

            def compile(self, workflow):
                captured["workflow"] = workflow
                return "compiled"

        monkeypatch.setattr(
            "victor.workflows.yaml_to_graph_compiler.YAMLToStateGraphCompiler",
            FakeCompiler,
        )

        compiled = compile_yaml_workflow(
            workflow,
            orchestrator=orchestrator,
            tool_registry=tool_registry,
            config=config,
        )

        assert compiled == "compiled"
        assert captured["workflow"] is workflow
        assert captured["orchestrator"] is orchestrator
        assert captured["orchestrators"] is None
        assert captured["tool_registry"] is tool_registry
        assert captured["config"] is config

    @pytest.mark.asyncio
    async def test_execute_yaml_workflow(self):
        """Test execute_yaml_workflow function."""
        workflow = (
            WorkflowBuilder("test").add_transform("step", lambda ctx: {**ctx, "done": True}).build()
        )

        result = await execute_yaml_workflow(
            workflow,
            initial_state={"input": "test"},
        )

        assert result.success is True
        assert result.state.get("done") is True

    @pytest.mark.asyncio
    async def test_execute_yaml_workflow_passes_keyword_dependencies(self, monkeypatch):
        """Test execute helper preserves tool registry and config keyword binding."""
        workflow = WorkflowBuilder("test").add_transform("step", lambda ctx: ctx).build()
        orchestrator = object()
        tool_registry = object()
        config = CompilerConfig(timeout=12.0)
        captured = {}

        class FakeCompiler:
            def __init__(
                self,
                orchestrator=None,
                orchestrators=None,
                tool_registry=None,
                config=None,
            ):
                captured["orchestrator"] = orchestrator
                captured["orchestrators"] = orchestrators
                captured["tool_registry"] = tool_registry
                captured["config"] = config

            async def compile_and_execute(self, workflow, initial_state, thread_id):
                captured["workflow"] = workflow
                captured["initial_state"] = initial_state
                captured["thread_id"] = thread_id
                return "executed"

        monkeypatch.setattr(
            "victor.workflows.yaml_to_graph_compiler.YAMLToStateGraphCompiler",
            FakeCompiler,
        )

        result = await execute_yaml_workflow(
            workflow,
            initial_state={"x": 1},
            orchestrator=orchestrator,
            tool_registry=tool_registry,
            thread_id="thread-1",
            config=config,
        )

        assert result == "executed"
        assert captured["workflow"] is workflow
        assert captured["initial_state"] == {"x": 1}
        assert captured["thread_id"] == "thread-1"
        assert captured["orchestrator"] is orchestrator
        assert captured["orchestrators"] is None
        assert captured["tool_registry"] is tool_registry
        assert captured["config"] is config


class TestIntegrationWithYAMLLoader:
    """Integration tests with YAML loader."""

    @pytest.mark.asyncio
    async def test_load_and_compile_yaml(self):
        """Test loading YAML and compiling to StateGraph."""
        from victor.workflows.yaml_loader import load_workflow_from_yaml

        yaml_content = """
workflows:
  simple_workflow:
    description: "Simple test workflow"
    nodes:
      - id: transform
        type: transform
        transform: "result = 'transformed'"
        next: []
"""

        workflow = load_workflow_from_yaml(yaml_content, "simple_workflow")

        compiler = YAMLToStateGraphCompiler()
        compiled = compiler.compile(workflow)

        assert compiled is not None

    @pytest.mark.asyncio
    async def test_compile_eda_style_workflow(self):
        """Test compiling an EDA-style workflow with multiple node types."""
        workflow = WorkflowDefinition(
            name="eda_test",
            description="EDA-style workflow",
            nodes={
                "load": ComputeNode(
                    id="load",
                    name="Load Data",
                    tools=["read_file"],
                    output_key="raw_data",
                    constraints=TaskConstraints(llm_allowed=False),
                    next_nodes=["validate"],
                ),
                "validate": ComputeNode(
                    id="validate",
                    name="Validate",
                    tools=["shell"],
                    input_mapping={"data": "raw_data"},
                    output_key="validation_result",
                    next_nodes=["check_quality"],
                ),
                "check_quality": ConditionNode(
                    id="check_quality",
                    name="Check Quality",
                    condition=lambda ctx: ("good" if ctx.get("quality_score", 0) > 0.8 else "bad"),
                    branches={"good": "analyze", "bad": "cleanup"},
                ),
                "cleanup": AgentNode(
                    id="cleanup",
                    name="Cleanup",
                    role="executor",
                    goal="Clean the data",
                    next_nodes=["analyze"],
                ),
                "analyze": AgentNode(
                    id="analyze",
                    name="Analyze",
                    role="analyst",
                    goal="Analyze patterns",
                    tool_budget=20,
                    next_nodes=[],
                ),
            },
            start_node="load",
        )

        compiler = YAMLToStateGraphCompiler()
        compiled = compiler.compile(workflow)

        assert compiled is not None

        # Execute with mock state
        result = await compiled.invoke(
            {
                "_workflow_id": "test",
                "_node_results": {},
                "_iteration": 0,
                "quality_score": 0.9,  # Will take "good" branch
            }
        )

        assert result.success is True
        # Should have executed: load -> validate -> check_quality -> analyze
        assert "load" in result.node_history
        assert "check_quality" in result.node_history
        assert "analyze" in result.node_history
        # Should NOT have executed cleanup (quality was good)
        assert "cleanup" not in result.node_history
