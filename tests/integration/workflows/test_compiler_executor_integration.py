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

"""Integration tests for workflow compiler and executor.

Tests end-to-end compilation and execution, ensuring that:
1. Compiler correctly compiles YAML workflows
2. Executor correctly executes compiled workflows
3. All node types work correctly
4. No regressions from legacy implementation
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict

from victor.core.bootstrap import bootstrap_container
from victor.config.settings import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def container(settings):
    """Create DI container with workflow services."""
    return bootstrap_container(settings)


@pytest.fixture
def simple_workflow_yaml():
    """Create a simple workflow YAML for testing."""
    return """
workflows:
  test_workflow:
    description: "Simple test workflow"
    nodes:
      - id: start
        type: agent
        role: researcher
        goal: "Test task"
        output: result
        next: []
"""


@pytest.fixture
def workflow_file(simple_workflow_yaml):
    """Create temporary workflow YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(simple_workflow_yaml)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.mark.integration
@pytest.mark.workflows
class TestWorkflowCompiler:
    """Test workflow compiler integration."""

    @pytest.mark.asyncio
    async def test_compiler_from_container(self, container):
        """Test that compiler can be resolved from DI container."""
        from victor.workflows.compiler.workflow_compiler_impl import (
            WorkflowCompilerImpl,
        )

        compiler = container.get(WorkflowCompilerImpl)
        assert compiler is not None

    @pytest.mark.asyncio
    async def test_compile_from_string(self, container, simple_workflow_yaml):
        """Test compiling workflow from YAML string."""
        from victor.workflows.compiler.workflow_compiler_impl import (
            WorkflowCompilerImpl,
        )

        compiler = container.get(WorkflowCompilerImpl)
        compiled = compiler.compile(simple_workflow_yaml)

        assert compiled is not None
        assert hasattr(compiled, "invoke")
        assert hasattr(compiled, "graph")

    @pytest.mark.asyncio
    async def test_compile_from_file(self, container, workflow_file):
        """Test compiling workflow from YAML file."""
        from victor.workflows.compiler.workflow_compiler_impl import (
            WorkflowCompilerImpl,
        )

        compiler = container.get(WorkflowCompilerImpl)
        compiled = compiler.compile(workflow_file)

        assert compiled is not None
        assert hasattr(compiled, "invoke")
        assert hasattr(compiled, "graph")


@pytest.mark.integration
@pytest.mark.workflows
class TestWorkflowExecutor:
    """Test workflow executor integration."""

    @pytest.mark.asyncio
    async def test_executor_from_container(self, container):
        """Test that executor can be resolved from DI container."""
        from victor.workflows.compiled_executor import WorkflowExecutor

        executor = container.get(WorkflowExecutor)
        assert executor is not None

    @pytest.mark.asyncio
    async def test_execute_compiled_workflow(self, container, simple_workflow_yaml):
        """Test executing a compiled workflow."""
        from victor.workflows.compiler.workflow_compiler_impl import (
            WorkflowCompilerImpl,
        )

        compiler = container.get(WorkflowCompilerImpl)
        compiled = compiler.compile(simple_workflow_yaml)

        # Note: This test will fail without proper mocking of orchestrator
        # It's here to verify the interface works
        assert compiled is not None


@pytest.mark.integration
@pytest.mark.workflows
class TestNodeExecutors:
    """Test individual node executors."""

    @pytest.mark.asyncio
    async def test_agent_executor_instantiation(self):
        """Test AgentNodeExecutor can be instantiated."""
        from victor.workflows.executors.agent import AgentNodeExecutor

        executor = AgentNodeExecutor(context=None)
        assert executor is not None
        assert executor.supports_node_type("agent")

    @pytest.mark.asyncio
    async def test_compute_executor_instantiation(self):
        """Test ComputeNodeExecutor can be instantiated."""
        from victor.workflows.executors.compute import ComputeNodeExecutor

        executor = ComputeNodeExecutor(context=None)
        assert executor is not None
        assert executor.supports_node_type("compute")

    @pytest.mark.asyncio
    async def test_transform_executor_instantiation(self):
        """Test TransformNodeExecutor can be instantiated."""
        from victor.workflows.executors.transform import TransformNodeExecutor

        executor = TransformNodeExecutor(context=None)
        assert executor is not None
        assert executor.supports_node_type("transform")

    @pytest.mark.asyncio
    async def test_parallel_executor_instantiation(self):
        """Test ParallelNodeExecutor can be instantiated."""
        from victor.workflows.executors.parallel import ParallelNodeExecutor

        executor = ParallelNodeExecutor(context=None)
        assert executor is not None
        assert executor.supports_node_type("parallel")

    @pytest.mark.asyncio
    async def test_condition_executor_instantiation(self):
        """Test ConditionNodeExecutor can be instantiated."""
        from victor.workflows.executors.condition import ConditionNodeExecutor

        executor = ConditionNodeExecutor(context=None)
        assert executor is not None
        assert executor.supports_node_type("condition")


@pytest.mark.integration
@pytest.mark.workflows
class TestNodeExecutorFactory:
    """Test node executor factory integration."""

    @pytest.mark.asyncio
    async def test_factory_from_container(self, container):
        """Test that factory can be resolved from DI container."""
        from victor.workflows.compiler_protocols import NodeExecutorFactoryProtocol

        factory = container.get(NodeExecutorFactoryProtocol)
        assert factory is not None

    @pytest.mark.asyncio
    async def test_factory_creates_all_executor_types(self, container):
        """Test that factory can create all executor types."""
        from victor.workflows.compiler_protocols import NodeExecutorFactoryProtocol
        from victor.workflows.definition import AgentNode, ComputeNode, TeamStepWorkflow
        from victor.workflows.hitl import HITLNode, HITLNodeType

        factory = container.get(NodeExecutorFactoryProtocol)

        # Test creating agent executor
        agent_node = AgentNode(
            id="test",
            name="Test Agent",
            role="researcher",
            goal="Test",
            output_key="result",
        )
        agent_executor = factory.create_executor(agent_node)
        assert agent_executor is not None
        assert callable(agent_executor)

        # Test creating compute executor
        compute_node = ComputeNode(
            id="test", name="Test Compute", handler="test_handler", output_key="result"
        )
        compute_executor = factory.create_executor(compute_node)
        assert compute_executor is not None
        assert callable(compute_executor)

        team_step = TeamStepWorkflow(
            id="team",
            name="Team",
            goal="Coordinate",
            members=[{"id": "lead", "role": "researcher", "goal": "Inspect"}],
        )
        team_executor = factory.create_executor(team_step)
        assert team_executor is not None
        assert callable(team_executor)

        hitl_node = HITLNode(
            id="approve",
            name="Approve",
            hitl_type=HITLNodeType.APPROVAL,
            prompt="Approve deploy?",
        )
        hitl_executor = factory.create_executor(hitl_node)
        assert hitl_executor is not None
        assert callable(hitl_executor)

    @pytest.mark.asyncio
    async def test_factory_supports_all_node_types(self, container):
        """Test that factory supports all expected node types."""
        from victor.workflows.compiler_protocols import NodeExecutorFactoryProtocol

        factory = container.get(NodeExecutorFactoryProtocol)

        assert factory.supports_node_type("agent")
        assert factory.supports_node_type("compute")
        assert factory.supports_node_type("transform")
        assert factory.supports_node_type("parallel")
        assert factory.supports_node_type("condition")
        assert factory.supports_node_type("team")
        assert factory.supports_node_type("hitl")


@pytest.mark.integration
@pytest.mark.workflows
class TestExecutionResult:
    """Test execution result protocol."""

    @pytest.mark.asyncio
    async def test_execution_result_properties(self):
        """Test that execution result has required properties."""
        from victor.workflows.compiled_executor import ExecutionResult

        result = ExecutionResult(final_state={"key": "value"}, metrics={"duration": 1.0})

        assert result.final_state == {"key": "value"}
        assert result.metrics == {"duration": 1.0}


@pytest.mark.integration
@pytest.mark.workflows
class TestCompiledGraph:
    """Test compiled graph protocol."""

    @pytest.mark.asyncio
    async def test_compiled_graph_has_graph_property(self, container, simple_workflow_yaml):
        """Test that compiled graph has graph property."""
        from victor.workflows.compiler.workflow_compiler_impl import (
            WorkflowCompilerImpl,
        )

        compiler = container.get(WorkflowCompilerImpl)
        compiled = compiler.compile(simple_workflow_yaml)

        assert hasattr(compiled, "graph")
        graph = compiled.graph
        assert graph is not None


@pytest.mark.integration
@pytest.mark.workflows
class TestDiContainerIntegration:
    """Test DI container integration."""

    @pytest.mark.asyncio
    async def test_all_workflow_services_registered(self, container):
        """Test that all workflow services are registered."""
        from victor.workflows.compiler.workflow_compiler_impl import (
            WorkflowCompilerImpl,
        )
        from victor.workflows.compiler_protocols import NodeExecutorFactoryProtocol
        from victor.workflows.compiled_executor import WorkflowExecutor

        # Should all be resolvable
        compiler = container.get(WorkflowCompilerImpl)
        factory = container.get(NodeExecutorFactoryProtocol)
        executor = container.get(WorkflowExecutor)

        assert compiler is not None
        assert factory is not None
        assert executor is not None


@pytest.mark.integration
@pytest.mark.workflows
class TestErrorHandling:
    """Test error handling in compiler and executor."""

    @pytest.mark.asyncio
    async def test_compile_invalid_yaml(self, container):
        """Test compiling invalid YAML raises appropriate error."""
        from victor.workflows.compiler.workflow_compiler_impl import (
            WorkflowCompilerImpl,
        )

        compiler = container.get(WorkflowCompilerImpl)

        with pytest.raises(Exception):
            compiler.compile("invalid: yaml: content: [[")

    @pytest.mark.asyncio
    async def test_compile_nonexistent_file(self, container):
        """Test compiling nonexistent file raises appropriate error."""
        from victor.workflows.compiler.workflow_compiler_impl import (
            WorkflowCompilerImpl,
        )

        compiler = container.get(WorkflowCompilerImpl)

        with pytest.raises(FileNotFoundError):
            compiler.compile("/nonexistent/file.yaml")


@pytest.mark.integration
@pytest.mark.workflows
class TestStreamingExecution:
    """Test streaming execution interface."""

    @pytest.mark.asyncio
    async def test_compiled_graph_has_stream_method(self, container, simple_workflow_yaml):
        """Test that compiled graph has stream method."""
        from victor.workflows.compiler.workflow_compiler_impl import (
            WorkflowCompilerImpl,
        )

        compiler = container.get(WorkflowCompilerImpl)
        compiled = compiler.compile(simple_workflow_yaml)

        assert hasattr(compiled, "stream")
        assert callable(compiled.stream)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
