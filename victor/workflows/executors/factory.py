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

"""Unified node executor factory for workflow nodes.

This factory consolidates the duplicate NodeExecutorFactory implementations
from:
- victor/workflows/unified_compiler.py (lines 171-600)
- victor/workflows/yaml_to_graph_compiler.py (lines 168-600)

Built-in and explicitly registered custom executor classes are supported.
Unregistered node types fail fast instead of silently delegating to the legacy
YAML compiler path.

Design Pattern: Factory
- Maps node types to executor functions
- Supports registration of custom node types (OCP compliance)
- Protocol-based dependency injection (DIP compliance)
"""

from __future__ import annotations

import logging
import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from victor.core.container import ServiceContainer
    from victor.workflows.definition import WorkflowNode
    from victor.workflows.compiler_protocols import NodeExecutorFactoryProtocol
    from victor.workflows.runtime_types import WorkflowState

logger = logging.getLogger(__name__)

BUILTIN_NODE_EXECUTOR_TYPES = (
    "agent",
    "compute",
    "transform",
    "parallel",
    "condition",
    "team",
    "hitl",
)


class NodeExecutorFactory:
    """Factory for creating StateGraph node executor functions.

    This factory converts YAML workflow node definitions into async functions
    that can be used as StateGraph nodes.

    Design Principles:
    - SRP: ONLY creates executor functions (doesn't execute them)
    - OCP: Open for extension via register_executor_type()
    - DIP: Depends on ExecutionContextProtocol, not concrete classes

    Attributes:
        _container: DI container for resolving dependencies
        _executor_types: Registered node executor classes

    Example:
        factory = NodeExecutorFactory(container=container)
        executor_fn = factory.create_executor(agent_node)
        result_state = await executor_fn(initial_state)
    """

    def __init__(self, container: Optional["ServiceContainer"] = None):
        """Initialize the factory.

        Args:
            container: DI container for resolving dependencies
        """
        from victor.core.container import ServiceContainer

        self._container: ServiceContainer = container or ServiceContainer()
        self._executor_types: Dict[str, Any] = {}
        self._register_builtin_executor_types()
        self._register_extension_executor_types()

    def _register_builtin_executor_types(self) -> None:
        """Register built-in workflow node executors."""
        from victor.workflows.executors.agent import AgentNodeExecutor
        from victor.workflows.executors.compute import ComputeNodeExecutor
        from victor.workflows.executors.condition import ConditionNodeExecutor
        from victor.workflows.executors.hitl import HITLNodeExecutor
        from victor.workflows.executors.parallel import ParallelNodeExecutor
        from victor.workflows.executors.team import TeamStepExecutor
        from victor.workflows.executors.transform import TransformNodeExecutor

        self.register_executor_type("agent", AgentNodeExecutor, replace=True)
        self.register_executor_type("compute", ComputeNodeExecutor, replace=True)
        self.register_executor_type("transform", TransformNodeExecutor, replace=True)
        self.register_executor_type("parallel", ParallelNodeExecutor, replace=True)
        self.register_executor_type("condition", ConditionNodeExecutor, replace=True)
        self.register_executor_type("team", TeamStepExecutor, replace=True)
        self.register_executor_type("hitl", HITLNodeExecutor, replace=True)

    def _register_extension_executor_types(self) -> None:
        """Register plugin- or application-provided workflow node executors."""
        from victor.workflows.executors.registry import (
            get_workflow_node_executor_registry,
        )

        registry = get_workflow_node_executor_registry()
        for registration in registry.get_registrations().values():
            self.register_executor_type(
                registration.node_type,
                registration.executor_factory,
                replace=registration.replace,
            )

    def register_executor_type(
        self,
        node_type: str,
        executor_class: Any,
        *,
        replace: bool = False,
    ) -> None:
        """Register a custom node executor type.

        Args:
            node_type: Node type identifier (e.g., "custom_compute")
            executor_class: Executor class (not instance) that implements
                          NodeExecutorProtocol
            replace: Whether to replace existing registration

        Raises:
            ValueError: If node_type already exists and replace=False

        Example:
            from victor.workflows.executors.agent import AgentNodeExecutor

            factory.register_executor_type("agent", AgentNodeExecutor)
        """
        if node_type in self._executor_types and not replace:
            raise ValueError(
                f"Node type '{node_type}' already registered. " f"Use replace=True to override."
            )

        self._executor_types[node_type] = executor_class
        logger.debug(f"Registered executor type: {node_type} -> {executor_class.__name__}")

    def create_executor(self, node: "WorkflowNode") -> Callable[["WorkflowState"], "WorkflowState"]:
        """Create an executor function for a workflow node.

        This is the main factory method. It dispatches to the appropriate
        executor class based on the node type.

        Args:
            node: Workflow node definition

        Returns:
            Callable: Async executor function that takes WorkflowState and
                     returns WorkflowState

        Raises:
            ValueError: If node type is not supported

        Example:
            factory = NodeExecutorFactory()
            executor_fn = factory.create_executor(agent_node)
            result_state = await executor_fn(initial_state)
        """
        node_type = getattr(getattr(node, "node_type", None), "value", None) or getattr(
            node, "node_type", "unknown"
        )
        registered = self._executor_types.get(node_type)
        if registered is not None:
            return self._create_registered_executor(node, registered)
        raise self._unsupported_node_type_error(node_type)

    def _create_registered_executor(
        self,
        node: "WorkflowNode",
        registered: Any,
    ) -> Callable[["WorkflowState"], "WorkflowState"]:
        """Create executor from a registered implementation.

        Supports both executor classes and direct factory callables.
        """
        if inspect.isclass(registered):
            executor = registered(context=self._resolve_execution_context())

            async def execute(state: "WorkflowState") -> "WorkflowState":
                return await executor.execute(node, state)

            return execute

        return registered(node)

    def _unsupported_node_type_error(self, node_type: Any) -> ValueError:
        """Build a consistent error for unsupported workflow node types."""
        supported = ", ".join(sorted(self._executor_types))
        return ValueError(
            f"Unsupported workflow node type '{node_type}'. "
            f"Register a custom executor with register_executor_type(). "
            f"Supported node types: {supported}"
        )

    def _resolve_execution_context(self) -> Any:
        """Resolve execution context for registered executors.

        SEAM BOUNDARY: Workflow node-level execution context (not agent or tool-level)
        - Use WorkflowNodeContext for: Workflow node executors, per-node execution state
        - Do NOT use for: Agent orchestrator (use RuntimeExecutionContext), tools (use ToolExecutionContext)
        """
        from victor.workflows.compiler_protocols import ExecutionContextProtocol
        from victor.workflows.execution_context import WorkflowNodeContext

        if hasattr(self._container, "get_optional"):
            context = self._container.get_optional(ExecutionContextProtocol)
            if context is not None:
                return context

        return WorkflowNodeContext(services=self._container)

    def supports_node_type(self, node_type: str) -> bool:
        """Check if a node type is supported.

        Args:
            node_type: Node type identifier

        Returns:
            bool: True if node type is supported

        Example:
            if factory.supports_node_type("agent"):
                executor = factory.create_executor(agent_node)
        """
        return node_type in self._executor_types


__all__ = [
    "BUILTIN_NODE_EXECUTOR_TYPES",
    "NodeExecutorFactory",
]
