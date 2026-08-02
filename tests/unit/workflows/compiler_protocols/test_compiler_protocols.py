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

"""Tests for workflow compiler protocols.

Verifies that all protocol definitions are ISP-compliant and that
concrete implementations satisfy the protocols.
"""

import pytest
from typing import Any, Dict
from victor.workflows.compiler_protocols import (
    WorkflowCompilerProtocol,
    CompiledGraphProtocol,
    ExecutionResultProtocol,
    ExecutionEventProtocol,
    NodeExecutorFactoryProtocol,
    NodeExecutorProtocol,
    ExecutionContextProtocol,
)


class TestWorkflowCompilerProtocol:
    """Test WorkflowCompilerProtocol ISP compliance."""

    def test_protocol_has_compile_method(self):
        """Verify protocol has compile method with correct signature."""
        assert hasattr(WorkflowCompilerProtocol, "compile")
        # Protocol methods are callable descriptors
        assert callable(WorkflowCompilerProtocol.compile)

    def test_protocol_is_minimal(self):
        """Verify protocol has only necessary methods (ISP compliance)."""
        # Count protocol methods
        protocol_methods = [
            name
            for name in dir(WorkflowCompilerProtocol)
            if not name.startswith("_") and callable(getattr(WorkflowCompilerProtocol, name))
        ]
        # Should only have 1 method: compile
        assert len(protocol_methods) <= 2, f"Too many methods: {protocol_methods}"


class TestCompiledGraphProtocol:
    """Test CompiledGraphProtocol ISP compliance."""

    def test_protocol_has_invoke_method(self):
        """Verify protocol has invoke method."""
        assert hasattr(CompiledGraphProtocol, "invoke")
        assert callable(CompiledGraphProtocol.invoke)

    def test_protocol_has_stream_method(self):
        """Verify protocol has stream method."""
        assert hasattr(CompiledGraphProtocol, "stream")
        assert callable(CompiledGraphProtocol.stream)

    def test_protocol_has_graph_property(self):
        """Verify protocol has graph property."""
        assert hasattr(CompiledGraphProtocol, "graph")

    def test_protocol_is_minimal(self):
        """Verify protocol has only necessary methods (ISP compliance)."""
        protocol_methods = [
            name
            for name in dir(CompiledGraphProtocol)
            if not name.startswith("_") and callable(getattr(CompiledGraphProtocol, name))
        ]
        # Should have 2 methods: invoke, stream
        assert len(protocol_methods) <= 3, f"Too many methods: {protocol_methods}"


class TestExecutionResultProtocol:
    """Test ExecutionResultProtocol ISP compliance."""

    def test_protocol_has_final_state_property(self):
        """Verify protocol has final_state property."""
        assert hasattr(ExecutionResultProtocol, "final_state")

    def test_protocol_has_metrics_property(self):
        """Verify protocol has metrics property."""
        assert hasattr(ExecutionResultProtocol, "metrics")

    def test_protocol_is_minimal(self):
        """Verify protocol has only necessary properties (ISP compliance)."""
        # Should only have 2 properties
        protocol_attrs = [name for name in dir(ExecutionResultProtocol) if not name.startswith("_")]
        # Should be minimal (properties + __protocol_attrs__ etc)
        assert len(protocol_attrs) <= 5, f"Too many attributes: {protocol_attrs}"


class TestExecutionEventProtocol:
    """Test ExecutionEventProtocol ISP compliance."""

    def test_protocol_has_node_id_property(self):
        """Verify protocol has node_id property."""
        assert hasattr(ExecutionEventProtocol, "node_id")

    def test_protocol_has_event_type_property(self):
        """Verify protocol has event_type property."""
        assert hasattr(ExecutionEventProtocol, "event_type")

    def test_protocol_has_data_property(self):
        """Verify protocol has data property."""
        assert hasattr(ExecutionEventProtocol, "data")


class TestNodeExecutorFactoryProtocol:
    """Test NodeExecutorFactoryProtocol ISP compliance."""

    def test_protocol_has_create_executor_method(self):
        """Verify protocol has create_executor method."""
        assert hasattr(NodeExecutorFactoryProtocol, "create_executor")
        assert callable(NodeExecutorFactoryProtocol.create_executor)

    def test_protocol_has_register_executor_type_method(self):
        """Verify protocol has register_executor_type method."""
        assert hasattr(NodeExecutorFactoryProtocol, "register_executor_type")
        assert callable(NodeExecutorFactoryProtocol.register_executor_type)

    def test_protocol_has_supports_node_type_method(self):
        """Verify protocol has supports_node_type method."""
        assert hasattr(NodeExecutorFactoryProtocol, "supports_node_type")
        assert callable(NodeExecutorFactoryProtocol.supports_node_type)


class TestNodeExecutorProtocol:
    """Test NodeExecutorProtocol ISP compliance."""

    def test_protocol_has_execute_method(self):
        """Verify protocol has execute method."""
        assert hasattr(NodeExecutorProtocol, "execute")
        assert callable(NodeExecutorProtocol.execute)

    def test_protocol_has_supports_node_type_method(self):
        """Verify protocol has supports_node_type method."""
        assert hasattr(NodeExecutorProtocol, "supports_node_type")
        assert callable(NodeExecutorProtocol.supports_node_type)


class TestExecutionContextProtocol:
    """Test ExecutionContextProtocol ISP compliance."""

    def test_protocol_has_orchestrator_property(self):
        """Verify protocol has orchestrator property."""
        assert hasattr(ExecutionContextProtocol, "orchestrator")

    def test_protocol_has_services_property(self):
        """Verify protocol has services property."""
        assert hasattr(ExecutionContextProtocol, "services")

    def test_protocol_has_settings_property(self):
        """Verify protocol has settings property."""
        assert hasattr(ExecutionContextProtocol, "settings")


class TestConcreteImplementations:
    """Test that concrete implementations satisfy protocols."""

    def test_node_executor_factory_satisfies_protocol(self):
        """Verify NodeExecutorFactory satisfies protocol."""
        from victor.workflows.executors.factory import NodeExecutorFactory

        factory = NodeExecutorFactory()

        # Test create_executor method
        assert hasattr(factory, "create_executor")
        assert callable(factory.create_executor)

        # Test register_executor_type method
        assert hasattr(factory, "register_executor_type")
        assert callable(factory.register_executor_type)

        # Test supports_node_type method
        assert hasattr(factory, "supports_node_type")
        assert callable(factory.supports_node_type)

        # Verify it supports all expected node types
        assert factory.supports_node_type("agent")
        assert factory.supports_node_type("compute")
        assert factory.supports_node_type("transform")
        assert factory.supports_node_type("parallel")
        assert factory.supports_node_type("condition")
        assert factory.supports_node_type("team")
        assert factory.supports_node_type("hitl")

    def test_agent_executor_satisfies_protocol(self):
        """Verify AgentNodeExecutor satisfies protocol."""
        from victor.workflows.executors.agent import AgentNodeExecutor

        executor = AgentNodeExecutor(context=None)

        # Test execute method
        assert hasattr(executor, "execute")
        assert callable(executor.execute)

        # Test supports_node_type method
        assert hasattr(executor, "supports_node_type")
        assert callable(executor.supports_node_type)

        # Verify it supports agent nodes
        assert executor.supports_node_type("agent")
        assert not executor.supports_node_type("compute")

    def test_compute_executor_satisfies_protocol(self):
        """Verify ComputeNodeExecutor satisfies protocol."""
        from victor.workflows.executors.compute import ComputeNodeExecutor

        executor = ComputeNodeExecutor(context=None)

        # Test execute method
        assert hasattr(executor, "execute")
        assert callable(executor.execute)

        # Test supports_node_type method
        assert hasattr(executor, "supports_node_type")
        assert callable(executor.supports_node_type)

        # Verify it supports compute nodes
        assert executor.supports_node_type("compute")
        assert not executor.supports_node_type("agent")

    def test_transform_executor_satisfies_protocol(self):
        """Verify TransformNodeExecutor satisfies protocol."""
        from victor.workflows.executors.transform import TransformNodeExecutor

        executor = TransformNodeExecutor(context=None)

        # Test execute method
        assert hasattr(executor, "execute")
        assert callable(executor.execute)

        # Test supports_node_type method
        assert hasattr(executor, "supports_node_type")
        assert callable(executor.supports_node_type)

        # Verify it supports transform nodes
        assert executor.supports_node_type("transform")
        assert not executor.supports_node_type("agent")

    def test_parallel_executor_satisfies_protocol(self):
        """Verify ParallelNodeExecutor satisfies protocol."""
        from victor.workflows.executors.parallel import ParallelNodeExecutor

        executor = ParallelNodeExecutor(context=None)

        # Test execute method
        assert hasattr(executor, "execute")
        assert callable(executor.execute)

        # Test supports_node_type method
        assert hasattr(executor, "supports_node_type")
        assert callable(executor.supports_node_type)

        # Verify it supports parallel nodes
        assert executor.supports_node_type("parallel")
        assert not executor.supports_node_type("agent")

    def test_condition_executor_satisfies_protocol(self):
        """Verify ConditionNodeExecutor satisfies protocol."""
        from victor.workflows.executors.condition import ConditionNodeExecutor

        executor = ConditionNodeExecutor(context=None)

        # Test execute method
        assert hasattr(executor, "execute")
        assert callable(executor.execute)

        # Test supports_node_type method
        assert hasattr(executor, "supports_node_type")
        assert callable(executor.supports_node_type)

        # Verify it supports condition nodes
        assert executor.supports_node_type("condition")
        assert not executor.supports_node_type("agent")

    def test_team_executor_satisfies_protocol(self):
        """Verify TeamStepExecutor satisfies protocol."""
        from victor.workflows.executors.team import TeamStepExecutor

        executor = TeamStepExecutor(context=None)

        assert hasattr(executor, "execute")
        assert callable(executor.execute)
        assert hasattr(executor, "supports_node_type")
        assert callable(executor.supports_node_type)
        assert executor.supports_node_type("team")
        assert not executor.supports_node_type("agent")

    def test_hitl_executor_satisfies_protocol(self):
        """Verify HITLNodeExecutor satisfies protocol."""
        from victor.workflows.executors.hitl import HITLNodeExecutor

        executor = HITLNodeExecutor(context=None)

        assert hasattr(executor, "execute")
        assert callable(executor.execute)
        assert hasattr(executor, "supports_node_type")
        assert callable(executor.supports_node_type)
        assert executor.supports_node_type("hitl")
        assert not executor.supports_node_type("agent")


class TestProtocolTypeHints:
    """Test that protocols have proper type hints."""

    def test_workflow_compiler_has_type_hints(self):
        """Verify WorkflowCompilerProtocol has type hints."""
        # Protocol should have __annotations__ or be a Protocol
        assert hasattr(WorkflowCompilerProtocol, "__annotations__") or hasattr(
            WorkflowCompilerProtocol, "__protocol_attrs__"
        )

    def test_compiled_graph_has_type_hints(self):
        """Verify CompiledGraphProtocol has type hints."""
        # Protocol should have __annotations__ or be a Protocol
        assert hasattr(CompiledGraphProtocol, "__annotations__") or hasattr(
            CompiledGraphProtocol, "__protocol_attrs__"
        )


@pytest.mark.unit
@pytest.mark.workflows
class TestProtocolDocumentation:
    """Test that protocols have comprehensive documentation."""

    def test_workflow_compiler_has_docstring(self):
        """Verify WorkflowCompilerProtocol has docstring."""
        assert WorkflowCompilerProtocol.__doc__ is not None
        assert len(WorkflowCompilerProtocol.__doc__) > 50

    def test_compiled_graph_has_docstring(self):
        """Verify CompiledGraphProtocol has docstring."""
        assert CompiledGraphProtocol.__doc__ is not None
        assert len(CompiledGraphProtocol.__doc__) > 50

    def test_execution_result_has_docstring(self):
        """Verify ExecutionResultProtocol has docstring."""
        assert ExecutionResultProtocol.__doc__ is not None
        assert len(ExecutionResultProtocol.__doc__) > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
