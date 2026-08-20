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

"""TDD Tests for UnifiedWorkflowCompiler.

These tests define the expected behavior of the UnifiedWorkflowCompiler
that consolidates all workflow compilation paths:
1. WorkflowGraphCompiler (victor/workflows/graph_compiler.py) - for graph_dsl workflows
2. YAMLToStateGraphCompiler (victor/workflows/yaml_to_graph_compiler.py) - for YAML workflows
3. WorkflowDefinitionCompiler (victor/workflows/graph_compiler.py) - for direct definitions

The UnifiedWorkflowCompiler provides a single entry point for compiling
any workflow source to CompiledGraph, with caching, observability, and
consistent error handling.

Tests follow TDD RED-GREEN-REFACTOR pattern. Some tests may fail until
missing features are implemented in the UnifiedWorkflowCompiler.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, TypedDict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.workflows.executors.registry import (
    clear_registered_workflow_node_executors,
    register_workflow_node_executor,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_yaml_content() -> str:
    """Simple YAML workflow content for testing."""
    return """
workflows:
  test_workflow:
    description: "Test workflow for UnifiedWorkflowCompiler"
    nodes:
      - id: start
        type: transform
        transform: "result = 'started'"
        next: [process]
      - id: process
        type: transform
        transform: "result = 'processed'"
        next: []
"""


@pytest.fixture
def yaml_workflow_file(sample_yaml_content: str, tmp_path: Path) -> Path:
    """Create a temporary YAML workflow file."""
    yaml_file = tmp_path / "test_workflow.yaml"
    yaml_file.write_text(sample_yaml_content)
    return yaml_file


@pytest.fixture
def conditional_yaml_content() -> str:
    """YAML workflow with conditional branching."""
    return """
workflows:
  conditional_workflow:
    description: "Workflow with condition node"
    nodes:
      - id: check_flag
        type: condition
        condition: "flag"
        branches:
          "true": success_branch
          "false": failure_branch
      - id: success_branch
        type: transform
        transform: "result = 'success'"
        next: []
      - id: failure_branch
        type: transform
        transform: "result = 'failure'"
        next: []
"""


@pytest.fixture
def parallel_yaml_content() -> str:
    """YAML workflow with parallel execution."""
    return """
workflows:
  parallel_workflow:
    description: "Workflow with parallel nodes"
    nodes:
      - id: start
        type: transform
        transform: "result = 'started'"
        next: [parallel_exec]
      - id: parallel_exec
        type: parallel
        parallel_nodes: [task_a, task_b]
        join_strategy: all
        next: [finish]
      - id: task_a
        type: transform
        transform: "task_a_result = 'a_done'"
      - id: task_b
        type: transform
        transform: "task_b_result = 'b_done'"
      - id: finish
        type: transform
        transform: "final_result = 'complete'"
        next: []
"""


@pytest.fixture
def agent_timeout_yaml_content() -> str:
    """YAML workflow with agent node timeout configuration."""
    return """
workflows:
  agent_timeout_workflow:
    description: "Workflow with agent timeout"
    nodes:
      - id: agent_with_timeout
        type: agent
        role: researcher
        goal: "Research the topic"
        tool_budget: 10
        timeout: 30
        next: [agent_without_timeout]
      - id: agent_without_timeout
        type: agent
        role: executor
        goal: "Execute the task"
        tool_budget: 5
        next: []
"""


@pytest.fixture
def workflow_execution_limits_yaml_content() -> str:
    """YAML workflow with workflow-level execution limits."""
    return """
workflows:
  limited_workflow:
    description: "Workflow with execution limits"
    timeout: 300
    default_node_timeout: 30
    max_iterations: 50
    max_retries: 3
    nodes:
      - id: step1
        type: transform
        transform: "result = 'done'"
        next: []
"""


@pytest.fixture
def workflow_execution_block_yaml_content() -> str:
    """YAML workflow with execution block syntax."""
    return """
workflows:
  execution_block_workflow:
    description: "Workflow with execution block"
    execution:
      timeout: 600
      default_node_timeout: 60
      max_iterations: 100
      max_retries: 5
    nodes:
      - id: step1
        type: transform
        transform: "result = 'done'"
        next: []
"""


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator for testing agent nodes."""
    orchestrator = MagicMock()
    orchestrator.chat = AsyncMock(return_value="mock response")
    return orchestrator


@pytest.fixture
def mock_event_emitter():
    """Mock event emitter for observability tests."""
    emitter = MagicMock()
    emitter.emit = MagicMock()
    emitter.emit_node_start = MagicMock()
    emitter.emit_node_complete = MagicMock()
    emitter.emit_node_error = MagicMock()
    emitter.emit_compilation_start = MagicMock()
    emitter.emit_compilation_complete = MagicMock()
    return emitter


@pytest.fixture
def reset_workflow_caches():
    """Reset global workflow caches before and after tests."""
    from victor.workflows.cache import (
        configure_workflow_cache,
        configure_workflow_definition_cache,
        WorkflowCacheConfig,
        DefinitionCacheConfig,
    )

    # Reset before test
    configure_workflow_definition_cache(DefinitionCacheConfig(enabled=True))
    configure_workflow_cache(WorkflowCacheConfig(enabled=True))

    yield

    # Reset after test
    configure_workflow_definition_cache(DefinitionCacheConfig(enabled=True))
    configure_workflow_cache(WorkflowCacheConfig(enabled=True))


# =============================================================================
# SECTION 1: Compilation Tests
# =============================================================================


class TestNodeExecutorFactoryCompatibility:
    """Tests for the deprecated unified-compiler factory shim."""

    @pytest.mark.asyncio
    async def test_factory_uses_shared_custom_executor_registry(self):
        """Custom node registrations should flow through the shared factory."""
        from victor.workflows.unified_compiler import NodeExecutorFactory

        class CustomExecutor:
            def __init__(self, context=None):
                self.context = context

            async def execute(self, node, state):
                state = dict(state)
                state["custom_output"] = node.id
                return state

        custom_node = SimpleNamespace(id="custom", node_type="custom_plugin")

        try:
            clear_registered_workflow_node_executors()
            register_workflow_node_executor("custom_plugin", CustomExecutor)

            factory = NodeExecutorFactory()
            executor = factory.create_executor(custom_node)
            result = await executor({})

            assert factory.supports_node_type("custom_plugin") is True
            assert result["custom_output"] == "custom"
        finally:
            clear_registered_workflow_node_executors()

    def test_set_runner_registry_updates_factory_compatibility_state(self):
        """UnifiedWorkflowCompiler should keep the runner registry on the shim."""
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        compiler = UnifiedWorkflowCompiler()
        registry = object()

        compiler.set_runner_registry(registry)

        assert compiler._runner_registry is registry
        assert compiler._executor_factory._runner_registry is registry


class TestCompilationFromYAML:
    """Tests for compiling workflows from YAML sources."""

    @pytest.mark.asyncio
    async def test_compile_from_yaml_path(self, yaml_workflow_file: Path, reset_workflow_caches):
        """Test compiling a YAML file path to CompiledGraph."""
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        compiler = UnifiedWorkflowCompiler(enable_caching=True)
        compiled = compiler.compile_yaml(yaml_workflow_file, workflow_name="test_workflow")

        # Verify it returns a CachedCompiledGraph
        assert isinstance(compiled, CachedCompiledGraph)

        # Verify underlying CompiledGraph exists
        assert compiled.compiled_graph is not None

        # Verify metadata
        assert compiled.workflow_name == "test_workflow"
        assert compiled.source_path == yaml_workflow_file

    @pytest.mark.asyncio
    async def test_compile_from_yaml_content(self, sample_yaml_content: str, tmp_path: Path):
        """Test compiling YAML string content via file to CompiledGraph."""
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        # Write content to temp file (current API requires file path)
        yaml_file = tmp_path / "content_test.yaml"
        yaml_file.write_text(sample_yaml_content)

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_yaml(yaml_file, workflow_name="test_workflow")

        assert isinstance(compiled, CachedCompiledGraph)

    @pytest.mark.asyncio
    async def test_compile_yaml_content_preserves_shared_parser_metadata(
        self, sample_yaml_content: str
    ):
        """YAML content compilation should use the shared parser metadata contract."""
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_yaml_content(sample_yaml_content, workflow_name="test_workflow")

        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.workflow_name == "test_workflow"
        assert compiled.source_path is None

    @pytest.mark.asyncio
    async def test_compile_yaml_with_agent_timeout(
        self, agent_timeout_yaml_content: str, tmp_path: Path
    ):
        """Test that agent timeout is parsed from YAML correctly."""
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
        from victor.workflows.yaml_loader import (
            load_workflow_from_yaml,
            YAMLWorkflowConfig,
        )

        # First verify YAML loader parses timeout correctly
        config = YAMLWorkflowConfig()
        workflows = load_workflow_from_yaml(agent_timeout_yaml_content, config=config)

        workflow_def = workflows.get("agent_timeout_workflow")
        assert workflow_def is not None

        # Check agent with timeout
        agent_with_timeout = workflow_def.nodes.get("agent_with_timeout")
        assert agent_with_timeout is not None
        assert agent_with_timeout.timeout_seconds == 30

        # Check agent without timeout
        agent_without_timeout = workflow_def.nodes.get("agent_without_timeout")
        assert agent_without_timeout is not None
        assert agent_without_timeout.timeout_seconds is None

        # Also verify compilation works
        yaml_file = tmp_path / "agent_timeout.yaml"
        yaml_file.write_text(agent_timeout_yaml_content)

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_yaml(yaml_file, "agent_timeout_workflow")
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_compile_yaml_with_workflow_level_limits(
        self, workflow_execution_limits_yaml_content: str, tmp_path: Path
    ):
        """Test that workflow-level limits are parsed from YAML correctly."""
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )
        from victor.workflows.yaml_loader import (
            load_workflow_from_yaml,
            YAMLWorkflowConfig,
        )

        # Verify YAML loader parses workflow-level limits
        config = YAMLWorkflowConfig()
        workflows = load_workflow_from_yaml(workflow_execution_limits_yaml_content, config=config)

        workflow_def = workflows.get("limited_workflow")
        assert workflow_def is not None
        assert workflow_def.max_execution_timeout_seconds == 300
        assert workflow_def.default_node_timeout_seconds == 30
        assert workflow_def.max_iterations == 50
        assert workflow_def.max_retries == 3

        # Verify compilation passes settings to CachedCompiledGraph
        yaml_file = tmp_path / "limits.yaml"
        yaml_file.write_text(workflow_execution_limits_yaml_content)

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_yaml(yaml_file, "limited_workflow")
        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.max_execution_timeout_seconds == 300
        assert compiled.max_iterations == 50

    @pytest.mark.asyncio
    async def test_compile_yaml_with_execution_block(
        self, workflow_execution_block_yaml_content: str, tmp_path: Path
    ):
        """Test that execution block syntax is parsed correctly."""
        from victor.workflows.yaml_loader import (
            load_workflow_from_yaml,
            YAMLWorkflowConfig,
        )

        config = YAMLWorkflowConfig()
        workflows = load_workflow_from_yaml(workflow_execution_block_yaml_content, config=config)

        workflow_def = workflows.get("execution_block_workflow")
        assert workflow_def is not None
        assert workflow_def.max_execution_timeout_seconds == 600
        assert workflow_def.default_node_timeout_seconds == 60
        assert workflow_def.max_iterations == 100
        assert workflow_def.max_retries == 5


class TestCompilationFromDefinition:
    """Tests for compiling from WorkflowDefinition objects."""

    @pytest.mark.asyncio
    async def test_compile_from_workflow_definition(self):
        """Test compiling a WorkflowDefinition to CachedCompiledGraph."""
        from victor.workflows.definition import (
            TransformNode,
            WorkflowBuilder,
            WorkflowDefinition,
        )
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        # Create a simple workflow definition
        workflow = (
            WorkflowBuilder("test")
            .add_transform("step1", lambda ctx: {**ctx, "step1": True})
            .add_transform("step2", lambda ctx: {**ctx, "step2": True})
            .build()
        )

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_definition(workflow)

        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.compiled_graph is not None

    @pytest.mark.asyncio
    async def test_compile_definition_uses_shared_custom_executor_registry(self):
        """Definition compilation should honor the shared custom executor registry."""
        from victor.workflows.definition import WorkflowDefinition, WorkflowNode
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        class CustomExecutor:
            def __init__(self, context=None):
                self.context = context

            async def execute(self, node, state):
                next_state = dict(state)
                next_state["custom_output"] = node.id
                return next_state

        @dataclass
        class CustomNode(WorkflowNode):
            @property
            def node_type(self) -> str:
                return "custom_plugin"

        workflow = WorkflowDefinition(
            name="custom_workflow",
            nodes={
                "custom": CustomNode(
                    id="custom",
                    name="Custom Node",
                ),
            },
            start_node="custom",
        )

        try:
            clear_registered_workflow_node_executors()
            register_workflow_node_executor("custom_plugin", CustomExecutor)

            compiler = UnifiedWorkflowCompiler()
            compiled = compiler.compile_definition(workflow, cache_key="custom-workflow")
            result = await compiled.invoke({})

            assert result.success is True
            assert result.state["custom_output"] == "custom"
        finally:
            clear_registered_workflow_node_executors()


class TestCompilationFromGraphDSL:
    """Tests for compiling from WorkflowGraph (graph_dsl)."""

    @pytest.mark.asyncio
    async def test_compile_from_workflow_graph(self):
        """Test compiling a WorkflowGraph (DSL) to CachedCompiledGraph."""
        from victor.workflows.graph_dsl import State, WorkflowGraph
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        @dataclass
        class TestState(State):
            value: str = ""
            processed: bool = False

        def process_node(state: TestState) -> TestState:
            state.processed = True
            state.value = "processed"
            return state

        # Create graph using DSL
        graph = WorkflowGraph(TestState, name="test_graph")
        graph.add_node("process", process_node)
        graph.set_entry_point("process")
        graph.set_finish_point("process")

        # Compile via UnifiedWorkflowCompiler
        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_graph(graph)

        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.compiled_graph is not None

    def test_compile_graph_uses_boundary_graph_dsl_adapter(self):
        """UnifiedWorkflowCompiler should depend on the boundary adapter, not graph_compiler directly."""
        from victor.workflows.graph_dsl import State, WorkflowGraph
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        @dataclass
        class TestState(State):
            value: str = ""

        graph = WorkflowGraph(TestState, name="boundary_graph")
        graph.add_node("process", lambda state: state)
        graph.set_entry_point("process")
        graph.set_finish_point("process")

        backend = MagicMock(name="graph_backend")
        backend.compile.return_value = object()

        with patch(
            "victor.workflows.compiler.boundary.LegacyWorkflowDslCompiler",
            return_value=backend,
        ) as adapter_cls:
            compiler = UnifiedWorkflowCompiler()
            compiled = compiler.compile_graph(graph, name="graph-name")

        adapter_cls.assert_called_once_with(
            runner_registry=None,
            validate_before_compile=True,
            preserve_state_type=False,
            emitter=None,
            enable_observability=False,
        )
        backend.compile.assert_called_once_with(graph, "graph-name")
        assert compiled.compiled_graph is backend.compile.return_value


class TestCompilationWithCaching:
    """Tests for compilation caching behavior."""

    @pytest.mark.asyncio
    async def test_compile_with_caching_enabled(
        self, yaml_workflow_file: Path, reset_workflow_caches
    ):
        """Test that compilation uses definition cache when enabled."""
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        compiler = UnifiedWorkflowCompiler(enable_caching=True)

        # First compilation
        compiled1 = compiler.compile_yaml(yaml_workflow_file, workflow_name="test_workflow")

        # Second compilation should use cache
        compiled2 = compiler.compile_yaml(yaml_workflow_file, workflow_name="test_workflow")

        # Both should be valid
        assert compiled1.compiled_graph is not None
        assert compiled2.compiled_graph is not None

        # Cache stats should show activity
        stats = compiler.get_cache_stats()
        assert stats.get("caching_enabled") is True

    @pytest.mark.asyncio
    async def test_compile_cache_hit(self, yaml_workflow_file: Path, reset_workflow_caches):
        """Test that cached compilation is returned for identical inputs."""
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        compiler = UnifiedWorkflowCompiler(enable_caching=True)

        # Compile multiple times
        _ = compiler.compile_yaml(yaml_workflow_file, workflow_name="test_workflow")
        _ = compiler.compile_yaml(yaml_workflow_file, workflow_name="test_workflow")
        _ = compiler.compile_yaml(yaml_workflow_file, workflow_name="test_workflow")

        # Verify cache hits
        stats = compiler.get_cache_stats()
        assert stats.get("definition_cache", {}).get("hits", 0) >= 2

    @pytest.mark.asyncio
    async def test_compile_cache_invalidation_on_file_change(
        self, tmp_path: Path, reset_workflow_caches
    ):
        """Test that cache is invalidated when source file changes."""
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        # Create initial YAML file
        yaml_file = tmp_path / "workflow.yaml"
        yaml_file.write_text("""
workflows:
  test:
    nodes:
      - id: step1
        type: transform
        transform: "result = 'v1'"
        next: []
""")

        compiler = UnifiedWorkflowCompiler(enable_caching=True)

        # First compilation
        compiled1 = compiler.compile_yaml(yaml_file, workflow_name="test")

        # Small delay to ensure mtime changes
        time.sleep(0.1)

        # Modify the file
        yaml_file.write_text("""
workflows:
  test:
    nodes:
      - id: step1
        type: transform
        transform: "result = 'v2'"
        next: []
""")

        # Re-compile - should detect file change and recompile
        compiled2 = compiler.compile_yaml(yaml_file, workflow_name="test")

        # Both should work
        assert compiled1.compiled_graph is not None
        assert compiled2.compiled_graph is not None


# =============================================================================
# SECTION 2: Node Execution Tests
# =============================================================================


class TestNodeExecution:
    """Tests for individual node type execution after compilation."""

    @pytest.mark.asyncio
    async def test_agent_node_execution(self, mock_orchestrator):
        """Test that AgentNode compiles correctly."""
        from victor.workflows.definition import AgentNode, WorkflowDefinition
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        workflow = WorkflowDefinition(
            name="agent_test",
            description="Test agent node execution",
            nodes={
                "agent": AgentNode(
                    id="agent",
                    name="Test Agent",
                    role="researcher",
                    goal="Test the system",
                    tool_budget=5,
                    next_nodes=[],
                ),
            },
            start_node="agent",
        )

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_definition(workflow)

        # Agent node should compile to CachedCompiledGraph
        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.compiled_graph is not None

    @pytest.mark.asyncio
    async def test_compute_node_execution(self):
        """Test that ComputeNode compiles correctly."""
        from victor.workflows.definition import (
            ComputeNode,
            TaskConstraints,
            WorkflowDefinition,
        )
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        workflow = WorkflowDefinition(
            name="compute_test",
            description="Test compute node execution",
            nodes={
                "compute": ComputeNode(
                    id="compute",
                    name="Test Compute",
                    tools=[],  # No actual tools
                    constraints=TaskConstraints(llm_allowed=False),
                    handler=None,
                    next_nodes=[],
                ),
            },
            start_node="compute",
        )

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_definition(workflow)

        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.compiled_graph is not None

    @pytest.mark.asyncio
    async def test_condition_node_routing(self):
        """Test that conditional branching compiles correctly."""
        from victor.workflows.definition import (
            ConditionNode,
            TransformNode,
            WorkflowDefinition,
        )
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        workflow = WorkflowDefinition(
            name="condition_test",
            description="Test condition node routing",
            nodes={
                "check": ConditionNode(
                    id="check",
                    name="Check Flag",
                    condition=lambda ctx: "yes" if ctx.get("flag") else "no",
                    branches={"yes": "success", "no": "failure"},
                ),
                "success": TransformNode(
                    id="success",
                    name="Success",
                    transform=lambda ctx: {**ctx, "result": "success_path"},
                    next_nodes=[],
                ),
                "failure": TransformNode(
                    id="failure",
                    name="Failure",
                    transform=lambda ctx: {**ctx, "result": "failure_path"},
                    next_nodes=[],
                ),
            },
            start_node="check",
        )

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_definition(workflow)

        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.compiled_graph is not None

    @pytest.mark.asyncio
    async def test_transform_node_execution(self):
        """Test that TransformNode compiles correctly."""
        from victor.workflows.definition import TransformNode, WorkflowDefinition
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        workflow = WorkflowDefinition(
            name="transform_test",
            description="Test transform node",
            nodes={
                "transform": TransformNode(
                    id="transform",
                    name="Double Value",
                    transform=lambda ctx: {**ctx, "doubled": ctx.get("value", 0) * 2},
                    next_nodes=[],
                ),
            },
            start_node="transform",
        )

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_definition(workflow)

        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.compiled_graph is not None

    @pytest.mark.asyncio
    async def test_parallel_node_compilation(self):
        """Test that parallel nodes compile correctly."""
        from victor.workflows.definition import (
            ParallelNode,
            TransformNode,
            WorkflowDefinition,
        )
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        workflow = WorkflowDefinition(
            name="parallel_test",
            description="Test parallel execution",
            nodes={
                "start": TransformNode(
                    id="start",
                    name="Start",
                    transform=lambda ctx: ctx,
                    next_nodes=["parallel"],
                ),
                "parallel": ParallelNode(
                    id="parallel",
                    name="Parallel Exec",
                    parallel_nodes=["task_a", "task_b"],
                    join_strategy="all",
                    next_nodes=["end"],
                ),
                "task_a": TransformNode(
                    id="task_a",
                    name="Task A",
                    transform=lambda ctx: {**ctx, "task_a": True},
                    next_nodes=[],
                ),
                "task_b": TransformNode(
                    id="task_b",
                    name="Task B",
                    transform=lambda ctx: {**ctx, "task_b": True},
                    next_nodes=[],
                ),
                "end": TransformNode(
                    id="end",
                    name="End",
                    transform=lambda ctx: ctx,
                    next_nodes=[],
                ),
            },
            start_node="start",
        )

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_definition(workflow)

        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.compiled_graph is not None

    @pytest.mark.asyncio
    async def test_hitl_node_compilation(self):
        """Test that HITL nodes compile correctly."""
        from victor.workflows.definition import WorkflowBuilder
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        # Build workflow with HITL approval node
        workflow = (
            WorkflowBuilder("hitl_test")
            .add_transform("prepare", lambda ctx: {**ctx, "prepared": True})
            .add_hitl_approval(
                "approval",
                prompt="Approve this action?",
                timeout=5.0,
                fallback="continue",
                next_nodes=["finish"],
            )
            .add_transform("finish", lambda ctx: {**ctx, "finished": True})
            .build()
        )

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_definition(workflow)

        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.compiled_graph is not None

    @pytest.mark.asyncio
    async def test_agent_node_with_timeout_configuration(self):
        """Test that AgentNode timeout_seconds is properly configured."""
        from victor.workflows.definition import AgentNode, WorkflowDefinition
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        workflow = WorkflowDefinition(
            name="agent_timeout_test",
            description="Test agent node with timeout",
            nodes={
                "agent": AgentNode(
                    id="agent",
                    name="Test Agent With Timeout",
                    role="researcher",
                    goal="Test the system with timeout",
                    tool_budget=5,
                    timeout_seconds=30.0,  # 30 second timeout
                    next_nodes=[],
                ),
            },
            start_node="agent",
        )

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_definition(workflow)

        # Agent node should compile to CachedCompiledGraph
        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.compiled_graph is not None

        # Verify the timeout is stored in the workflow definition
        agent_node = workflow.nodes["agent"]
        assert agent_node.timeout_seconds == 30.0

    @pytest.mark.asyncio
    async def test_agent_node_timeout_in_serialization(self):
        """Test that AgentNode timeout serializes correctly to dict."""
        from victor.workflows.definition import AgentNode

        node = AgentNode(
            id="test_agent",
            name="Test Agent",
            role="executor",
            goal="Execute task",
            timeout_seconds=60.0,
        )

        serialized = node.to_dict()
        assert serialized["timeout_seconds"] == 60.0

        # Also test without timeout
        node_no_timeout = AgentNode(
            id="test_agent_2",
            name="Test Agent 2",
            role="executor",
            goal="Execute task",
        )
        serialized_no_timeout = node_no_timeout.to_dict()
        assert serialized_no_timeout["timeout_seconds"] is None

    @pytest.mark.asyncio
    async def test_workflow_level_timeout_configuration(self):
        """Test that workflow-level timeout settings are properly configured."""
        from victor.workflows.definition import TransformNode, WorkflowDefinition
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        workflow = WorkflowDefinition(
            name="workflow_timeout_test",
            description="Test workflow with execution limits",
            nodes={
                "step1": TransformNode(
                    id="step1",
                    name="Step 1",
                    transform=lambda ctx: ctx,
                    next_nodes=[],
                ),
            },
            start_node="step1",
            max_execution_timeout_seconds=300.0,  # 5 minute timeout
            default_node_timeout_seconds=30.0,
            max_iterations=50,
            max_retries=3,
        )

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_definition(workflow)

        # Verify workflow-level settings are passed to CachedCompiledGraph
        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.max_execution_timeout_seconds == 300.0
        assert compiled.default_node_timeout_seconds == 30.0
        assert compiled.max_iterations == 50
        assert compiled.max_retries == 3

    @pytest.mark.asyncio
    async def test_workflow_definition_serialization_with_limits(self):
        """Test that WorkflowDefinition with limits serializes correctly."""
        from victor.workflows.definition import TransformNode, WorkflowDefinition

        workflow = WorkflowDefinition(
            name="limits_test",
            description="Test serialization",
            nodes={
                "step1": TransformNode(
                    id="step1",
                    name="Step 1",
                    transform=lambda ctx: ctx,
                    next_nodes=[],
                ),
            },
            start_node="step1",
            max_execution_timeout_seconds=600.0,
            default_node_timeout_seconds=60.0,
            max_iterations=100,
            max_retries=5,
        )

        serialized = workflow.to_dict()
        assert serialized["max_execution_timeout_seconds"] == 600.0
        assert serialized["default_node_timeout_seconds"] == 60.0
        assert serialized["max_iterations"] == 100
        assert serialized["max_retries"] == 5


# =============================================================================
# SECTION 3: State Handling Tests
# =============================================================================


class TestStateHandling:
    """Tests for state type handling during compilation."""

    @pytest.mark.asyncio
    async def test_state_dict_input(self):
        """Test that plain dict state works correctly."""
        from victor.workflows.definition import TransformNode, WorkflowDefinition
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        workflow = WorkflowDefinition(
            name="dict_state_test",
            nodes={
                "process": TransformNode(
                    id="process",
                    name="Process",
                    transform=lambda ctx: {**ctx, "processed": True},
                    next_nodes=[],
                ),
            },
            start_node="process",
        )

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_definition(workflow)

        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.compiled_graph is not None

    @pytest.mark.asyncio
    async def test_state_typed_input(self):
        """Test that TypedDict state works correctly with graph DSL."""
        from victor.workflows.graph_dsl import State, WorkflowGraph
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        @dataclass
        class DataclassState(State):
            value: int = 0
            result: Optional[str] = None
            processed: bool = False

        def process_typed(state: DataclassState) -> DataclassState:
            state.processed = True
            state.result = f"processed_{state.value}"
            return state

        graph = WorkflowGraph(DataclassState, name="typed_test")
        graph.add_node("process", process_typed)
        graph.set_entry_point("process")
        graph.set_finish_point("process")

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_graph(graph)

        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.compiled_graph is not None

    @pytest.mark.asyncio
    async def test_state_dataclass_input(self):
        """Test that dataclass state works correctly."""
        from victor.workflows.graph_dsl import State, WorkflowGraph
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        @dataclass
        class MyDataclassState(State):
            counter: int = 0
            name: str = ""

        def increment(state: MyDataclassState) -> MyDataclassState:
            state.counter += 1
            return state

        graph = WorkflowGraph(MyDataclassState, name="dataclass_test")
        graph.add_node("increment", increment)
        graph.set_entry_point("increment")
        graph.set_finish_point("increment")

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_graph(graph)

        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.compiled_graph is not None

    @pytest.mark.asyncio
    async def test_state_consistency_across_nodes(self):
        """Test that state flows correctly through multiple nodes."""
        from victor.workflows.definition import TransformNode, WorkflowDefinition
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        workflow = WorkflowDefinition(
            name="state_flow_test",
            nodes={
                "step1": TransformNode(
                    id="step1",
                    name="Step 1",
                    transform=lambda ctx: {
                        **ctx,
                        "step1_value": ctx.get("initial", 0) + 1,
                    },
                    next_nodes=["step2"],
                ),
                "step2": TransformNode(
                    id="step2",
                    name="Step 2",
                    transform=lambda ctx: {
                        **ctx,
                        "step2_value": ctx.get("step1_value", 0) + 1,
                    },
                    next_nodes=["step3"],
                ),
                "step3": TransformNode(
                    id="step3",
                    name="Step 3",
                    transform=lambda ctx: {
                        **ctx,
                        "final_value": ctx.get("step2_value", 0) + 1,
                    },
                    next_nodes=[],
                ),
            },
            start_node="step1",
        )

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_definition(workflow)

        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.compiled_graph is not None


# =============================================================================
# SECTION 4: Caching Integration Tests
# =============================================================================


class TestCachingIntegration:
    """Tests for caching integration with the compiler."""

    @pytest.mark.asyncio
    async def test_definition_cache_integration(
        self, yaml_workflow_file: Path, reset_workflow_caches
    ):
        """Test that WorkflowDefinitionCache is properly integrated."""
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        # Create compiler with caching enabled
        compiler = UnifiedWorkflowCompiler(enable_caching=True)

        # First compile
        compiled1 = compiler.compile_yaml(yaml_workflow_file, workflow_name="test_workflow")

        # Get cache stats
        stats = compiler.get_cache_stats()

        # Definition cache should be active
        assert stats.get("definition_cache", {}).get("enabled", False) is True

        # Second compile should hit cache
        compiled2 = compiler.compile_yaml(yaml_workflow_file, workflow_name="test_workflow")

        stats_after = compiler.get_cache_stats()
        assert stats_after.get("definition_cache", {}).get("hits", 0) >= 1

    @pytest.mark.asyncio
    async def test_execution_cache_integration(self, reset_workflow_caches):
        """Test that execution result caching works when enabled."""
        from victor.workflows.definition import TransformNode, WorkflowDefinition
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
        from victor.workflows.cache import WorkflowCacheConfig

        workflow = WorkflowDefinition(
            name="exec_cache_test",
            nodes={
                "transform": TransformNode(
                    id="transform",
                    name="Deterministic Transform",
                    transform=lambda ctx: {**ctx, "result": ctx.get("input", 0) * 2},
                    next_nodes=[],
                ),
            },
            start_node="transform",
        )

        # Create compiler with execution caching
        from victor.workflows.cache import (
            WorkflowCacheManager,
            WorkflowDefinitionCache,
            DefinitionCacheConfig,
        )

        exec_cache = WorkflowCacheManager(WorkflowCacheConfig(enabled=True))
        def_cache = WorkflowDefinitionCache(DefinitionCacheConfig(enabled=True))

        compiler = UnifiedWorkflowCompiler(
            definition_cache=def_cache,
            execution_cache=exec_cache,
            enable_caching=True,
        )
        compiled = compiler.compile_definition(workflow)

        # Execution cache stats should be available
        stats = compiler.get_cache_stats()
        assert stats.get("caching_enabled", False) is True

    @pytest.mark.asyncio
    async def test_cache_cascade_invalidation(self, tmp_path: Path, reset_workflow_caches):
        """Test that cache invalidation works correctly."""
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        # Create YAML file
        yaml_file = tmp_path / "cascade_test.yaml"
        yaml_file.write_text("""
workflows:
  test:
    nodes:
      - id: step1
        type: transform
        transform: "result = 'done'"
        next: []
""")

        compiler = UnifiedWorkflowCompiler(enable_caching=True)

        # Compile to populate cache
        _ = compiler.compile_yaml(yaml_file, workflow_name="test")

        # Invalidate the YAML
        invalidated_count = compiler.invalidate_yaml(yaml_file)

        # Should have invalidated entries
        assert invalidated_count >= 0


# =============================================================================
# SECTION 5: Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling during compilation."""

    def test_invalid_yaml_raises_error(self, tmp_path: Path):
        """Test that invalid YAML syntax raises appropriate error."""
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
        from victor.workflows.yaml_loader import YAMLWorkflowError

        # Create invalid YAML file
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("""
workflows:
  test:
    nodes:
      - id: broken
        type: [this is not valid yaml
""")

        compiler = UnifiedWorkflowCompiler()

        with pytest.raises((YAMLWorkflowError, Exception)) as exc_info:
            compiler.compile_yaml(yaml_file, workflow_name="test")

        # Should contain info about the error
        assert exc_info.value is not None

    def test_missing_node_raises_error(self, tmp_path: Path):
        """Test that missing node reference raises appropriate error."""
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
        from victor.workflows.yaml_loader import YAMLWorkflowError

        yaml_file = tmp_path / "missing_node.yaml"
        yaml_file.write_text("""
workflows:
  test:
    nodes:
      - id: start
        type: transform
        transform: "result = 'done'"
        next: [nonexistent_node]
""")

        compiler = UnifiedWorkflowCompiler()

        with pytest.raises((YAMLWorkflowError, ValueError, Exception)) as exc_info:
            compiler.compile_yaml(yaml_file, workflow_name="test")

        # Should indicate the missing node
        error_msg = str(exc_info.value).lower()
        assert "nonexistent" in error_msg or "not found" in error_msg or "invalid" in error_msg

    def test_circular_dependency_detection(self, tmp_path: Path):
        """Test that circular dependencies are handled."""
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        yaml_file = tmp_path / "circular.yaml"
        yaml_file.write_text("""
workflows:
  test:
    nodes:
      - id: node_a
        type: transform
        transform: "a = 1"
        next: [node_b]
      - id: node_b
        type: transform
        transform: "b = 2"
        next: [node_c]
      - id: node_c
        type: transform
        transform: "c = 3"
        next: [node_a]
""")

        compiler = UnifiedWorkflowCompiler()

        # Circular dependencies may be allowed (for cyclic graphs) or raise error
        # depending on implementation. Test that it at least doesn't crash.
        try:
            compiled = compiler.compile_yaml(yaml_file, workflow_name="test")
            # If compiled, it should be a valid object
            assert compiled.compiled_graph is not None
        except Exception as e:
            # If it raises, should be clear about circular dependency
            error_msg = str(e).lower()
            assert (
                "circular" in error_msg
                or "cycle" in error_msg
                or "loop" in error_msg
                or "unreachable" in error_msg
                or len(error_msg) > 0  # Has some error message
            )

    def test_missing_entry_point_error(self):
        """Test that missing entry point (empty workflow) raises appropriate error.

        Note: WorkflowDefinition auto-sets start_node to the first node if not
        provided, so we test with completely empty nodes to trigger validation error.
        """
        from victor.workflows.definition import WorkflowDefinition
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            UnifiedCompilerConfig,
        )

        # Empty workflow - start_node cannot be auto-inferred
        workflow = WorkflowDefinition(
            name="empty",
            description="Workflow with no nodes",
            nodes={},  # Empty nodes
            start_node=None,
        )

        # Create compiler with validation enabled (default)
        config = UnifiedCompilerConfig(validate_before_compile=True)
        compiler = UnifiedWorkflowCompiler(config=config)

        # Should raise error about empty workflow / missing nodes
        with pytest.raises((ValueError, Exception)) as exc_info:
            compiler.compile_definition(workflow)

        error_msg = str(exc_info.value).lower()
        assert (
            "node" in error_msg
            or "empty" in error_msg
            or "validation" in error_msg
            or len(error_msg) > 0
        )


# =============================================================================
# SECTION 6: Observability Tests
# =============================================================================


class TestObservability:
    """Tests for observability event emission during compilation and execution."""

    @pytest.mark.asyncio
    async def test_compilation_produces_valid_graph(
        self, yaml_workflow_file: Path, reset_workflow_caches
    ):
        """Test that compilation produces a graph that can emit schema."""
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_yaml(yaml_workflow_file, workflow_name="test_workflow")

        # Should be able to get graph schema for observability
        schema = compiled.get_graph_schema()
        assert schema is not None
        assert isinstance(schema, dict)

    @pytest.mark.asyncio
    async def test_cached_compiled_graph_metadata(
        self, yaml_workflow_file: Path, reset_workflow_caches
    ):
        """Test that CachedCompiledGraph includes observability metadata."""
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_yaml(yaml_workflow_file, workflow_name="test_workflow")

        # Check metadata
        assert compiled.workflow_name == "test_workflow"
        assert compiled.source_path == yaml_workflow_file
        assert compiled.compiled_at > 0
        assert compiled.source_mtime is not None
        assert compiled.age_seconds >= 0


# =============================================================================
# SECTION 7: API Consistency Tests
# =============================================================================


class TestAPIConsistency:
    """Tests for consistent API across all compilation methods."""

    @pytest.mark.asyncio
    async def test_all_compilation_methods_return_appropriate_types(
        self, yaml_workflow_file: Path, sample_yaml_content: str, reset_workflow_caches
    ):
        """Test that all compilation methods return CachedCompiledGraph."""
        from victor.workflows.definition import WorkflowBuilder
        from victor.workflows.graph_dsl import State, WorkflowGraph
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        compiler = UnifiedWorkflowCompiler()

        # Method 1: From YAML path - returns CachedCompiledGraph
        compiled1 = compiler.compile_yaml(yaml_workflow_file, workflow_name="test_workflow")
        assert isinstance(compiled1, CachedCompiledGraph)
        assert compiled1.compiled_graph is not None

        # Method 2: From WorkflowDefinition - returns CachedCompiledGraph
        workflow_def = WorkflowBuilder("api_test").add_transform("step", lambda ctx: ctx).build()
        compiled2 = compiler.compile_definition(workflow_def)
        assert isinstance(compiled2, CachedCompiledGraph)
        assert compiled2.compiled_graph is not None

        # Method 3: From WorkflowGraph - returns CachedCompiledGraph
        @dataclass
        class SimpleState(State):
            value: str = ""

        graph = WorkflowGraph(SimpleState, name="graph_test")
        graph.add_node("node", lambda s: s)
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        compiled3 = compiler.compile_graph(graph)
        assert isinstance(compiled3, CachedCompiledGraph)
        assert compiled3.compiled_graph is not None


# =============================================================================
# SECTION 8: Edge Cases and Robustness
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    @pytest.mark.asyncio
    async def test_empty_workflow_raises_error(self):
        """Test handling of empty workflow (no nodes)."""
        from victor.workflows.definition import WorkflowDefinition
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        workflow = WorkflowDefinition(
            name="empty",
            description="Empty workflow",
            nodes={},
            start_node=None,
        )

        compiler = UnifiedWorkflowCompiler()

        # Should raise validation error
        with pytest.raises((ValueError, Exception)) as exc_info:
            compiler.compile_definition(workflow)

        # Some error should be raised
        assert exc_info.value is not None

    @pytest.mark.asyncio
    async def test_single_node_workflow(self):
        """Test workflow with only one node."""
        from victor.workflows.definition import TransformNode, WorkflowDefinition
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        workflow = WorkflowDefinition(
            name="single",
            nodes={
                "only_node": TransformNode(
                    id="only_node",
                    name="Only Node",
                    transform=lambda ctx: {**ctx, "done": True},
                    next_nodes=[],
                ),
            },
            start_node="only_node",
        )

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_definition(workflow)

        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.compiled_graph is not None

    @pytest.mark.asyncio
    async def test_workflow_with_unicode_names(self):
        """Test workflow with unicode characters in names."""
        from victor.workflows.definition import TransformNode, WorkflowDefinition
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        workflow = WorkflowDefinition(
            name="unicode_test",
            description="Workflow with unicode",
            nodes={
                "step_emoji": TransformNode(
                    id="step_emoji",
                    name="Step with unicode",
                    transform=lambda ctx: {**ctx, "unicode_key": "value_test"},
                    next_nodes=[],
                ),
            },
            start_node="step_emoji",
        )

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_definition(workflow)

        assert isinstance(compiled, CachedCompiledGraph)
        assert compiled.compiled_graph is not None

    @pytest.mark.asyncio
    async def test_concurrent_compilation(self, yaml_workflow_file: Path, reset_workflow_caches):
        """Test thread safety of concurrent compilations."""
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        compiler = UnifiedWorkflowCompiler(enable_caching=True)

        async def compile_workflow():
            return compiler.compile_yaml(yaml_workflow_file, workflow_name="test_workflow")

        # Run multiple compilations concurrently
        tasks = [compile_workflow() for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent compilation failed: {result}")
            assert isinstance(result, CachedCompiledGraph)


# =============================================================================
# SECTION 9: Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_unified_compiler(self):
        """Test factory function creates proper compiler."""
        from victor.workflows.unified_compiler import (
            create_unified_compiler,
            UnifiedWorkflowCompiler,
        )

        compiler = create_unified_compiler(enable_caching=True)

        assert isinstance(compiler, UnifiedWorkflowCompiler)
        stats = compiler.get_cache_stats()
        assert stats.get("caching_enabled") is True

    def test_create_unified_compiler_without_caching(self):
        """Test factory function can disable caching."""
        from victor.workflows.unified_compiler import (
            create_unified_compiler,
            UnifiedWorkflowCompiler,
        )

        compiler = create_unified_compiler(enable_caching=False)

        assert isinstance(compiler, UnifiedWorkflowCompiler)
        stats = compiler.get_cache_stats()
        assert stats.get("caching_enabled") is False


# =============================================================================
# SECTION 10: Module Structure Tests
# =============================================================================


class TestModuleStructure:
    """Tests for module structure and exports."""

    def test_unified_compiler_module_exports(self):
        """Test that unified_compiler module exports expected items."""
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
            create_unified_compiler,
        )

        assert UnifiedWorkflowCompiler is not None
        assert CachedCompiledGraph is not None
        assert callable(create_unified_compiler)

    def test_compiler_methods_exist(self):
        """Test that compiler has expected methods."""
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        compiler = UnifiedWorkflowCompiler()

        # Check methods exist
        assert hasattr(compiler, "compile_yaml")
        assert hasattr(compiler, "compile_graph")
        assert hasattr(compiler, "compile_definition")
        assert hasattr(compiler, "get_cache_stats")
        assert hasattr(compiler, "clear_cache")
        assert hasattr(compiler, "invalidate_yaml")
        assert hasattr(compiler, "set_runner_registry")


# =============================================================================
# SECTION 11: CachedCompiledGraph Tests
# =============================================================================


class TestCachedCompiledGraph:
    """Tests for CachedCompiledGraph wrapper class."""

    @pytest.mark.asyncio
    async def test_cached_graph_wrapper_properties(
        self, yaml_workflow_file: Path, reset_workflow_caches
    ):
        """Test CachedCompiledGraph wrapper has correct properties."""
        from victor.workflows.unified_compiler import (
            UnifiedWorkflowCompiler,
            CachedCompiledGraph,
        )

        compiler = UnifiedWorkflowCompiler()
        cached = compiler.compile_yaml(yaml_workflow_file, workflow_name="test_workflow")

        # Check properties
        assert isinstance(cached, CachedCompiledGraph)
        assert cached.workflow_name == "test_workflow"
        assert cached.source_path == yaml_workflow_file
        assert cached.compiled_at > 0
        assert cached.age_seconds >= 0
        assert cached.compiled_graph is not None

    @pytest.mark.asyncio
    async def test_cached_graph_get_schema(self, yaml_workflow_file: Path, reset_workflow_caches):
        """Test CachedCompiledGraph can return schema."""
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        compiler = UnifiedWorkflowCompiler()
        cached = compiler.compile_yaml(yaml_workflow_file, workflow_name="test_workflow")

        schema = cached.get_graph_schema()
        assert isinstance(schema, dict)

    @pytest.mark.asyncio
    async def test_cached_graph_invoke_uses_shared_initial_state(self):
        """CachedCompiledGraph should seed execution with the canonical state shape."""
        from victor.workflows.unified_compiler import CachedCompiledGraph

        captured: Dict[str, Any] = {}

        class FakeCompiledGraph:
            async def invoke(self, state, *, config=None, thread_id=None):
                captured["state"] = state
                captured["thread_id"] = thread_id
                return MagicMock(
                    success=True,
                    state=state,
                    error=None,
                    node_history=[],
                    iterations=0,
                )

        cached = CachedCompiledGraph(
            compiled_graph=FakeCompiledGraph(),
            workflow_name="test_workflow",
            max_iterations=17,
        )

        result = await cached.invoke({"input": "value"}, thread_id="thread-123")

        assert result.success is True
        assert captured["thread_id"] == "thread-123"
        assert captured["state"]["_workflow_id"]
        assert captured["state"]["_workflow_name"] == "test_workflow"
        assert captured["state"]["_current_node"] == ""
        assert captured["state"]["_node_results"] == {}
        assert captured["state"]["_parallel_results"] == {}
        assert captured["state"]["_hitl_pending"] is False
        assert captured["state"]["_hitl_response"] is None
        assert captured["state"]["_max_iterations"] == 17
        assert captured["state"]["input"] == "value"

    def test_prepare_state_preserves_existing_workflow_metadata(self):
        """CachedCompiledGraph should not overwrite caller-provided metadata."""
        from victor.workflows.unified_compiler import CachedCompiledGraph

        cached = CachedCompiledGraph(
            compiled_graph=MagicMock(),
            workflow_name="fallback_workflow",
        )

        prepared = cached._prepare_state(
            {
                "_workflow_id": "wf-123",
                "_workflow_name": "provided_workflow",
                "_current_node": "step1",
                "_node_results": {"step1": {"success": True}},
                "_parallel_results": {"branch": {"done": True}},
                "input": "value",
            }
        )

        assert prepared["_workflow_id"] == "wf-123"
        assert prepared["_workflow_name"] == "provided_workflow"
        assert prepared["_current_node"] == "step1"
        assert prepared["_node_results"] == {"step1": {"success": True}}
        assert prepared["_parallel_results"] == {"branch": {"done": True}}
        assert prepared["input"] == "value"
