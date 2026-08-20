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

"""Workflow compiler and executor protocols.

This module defines ISP-compliant minimal protocols (interfaces) for the workflow
system. These protocols enable:

- **Dependency Inversion Principle (DIP)**: High-level modules depend on protocols,
  not concrete implementations
- **Interface Segregation Principle (ISP)**: Focused protocols with 1-3 methods each
- **Open/Closed Principle (OCP)**: New implementations can be added without modifying
  existing code
- **Liskov Substitution Principle (LSP)**: All protocol implementations are
  interchangeable

Design Principles:
- Minimal interfaces: Each protocol has ONLY the methods it needs
- No unused dependencies: Clients only depend on methods they actually use
- Protocol-based typing: Use Protocol for structural subtyping (duck typing with
  type hints)

Example:
    from victor.workflows.protocols import WorkflowCompilerProtocol

    def execute_workflow(compiler: WorkflowCompilerProtocol, yaml_path: str):
        compiled = compiler.compile(yaml_path)
        # compiler is guaranteed to have compile() method
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
)

if TYPE_CHECKING:
    from victor.framework.graph import StateGraph
    from victor.workflows.definition import WorkflowDefinition, WorkflowNode
    from victor.workflows.runtime_types import WorkflowState


class WorkflowCompilerProtocol(Protocol):
    """Protocol for workflow compilers.

    Minimal protocol for compiling YAML workflows into executable graphs.
    Follows ISP with only 1 method: compile().

    Responsibility:
    - Parse YAML from file/string
    - Validate workflow definition
    - Build executable StateGraph
    - Return compiled graph

    Non-responsibility:
    - Execution (handled by WorkflowExecutorProtocol)
    - Caching (handled by decorator/wrapper)
    - Observability (handled by decorator/wrapper)
    """

    def compile(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> "CompiledGraphProtocol":
        """Compile a workflow from YAML source.

        Args:
            source: YAML file path or YAML string content
            workflow_name: Name of workflow to compile (for multi-workflow files)
            validate: Whether to validate workflow definition before compilation

        Returns:
            CompiledGraphProtocol: Executable compiled graph

        Raises:
            ValueError: If source is invalid or validation fails
            FileNotFoundError: If YAML file doesn't exist

        Example:
            compiler: WorkflowCompilerProtocol = get_compiler()
            compiled = compiler.compile("workflow.yaml")
            result = await compiled.invoke({"input": "data"})
        """
        ...


class CompiledGraphProtocol(Protocol):
    """Protocol for compiled workflow graphs.

    Minimal protocol for executing compiled workflows.
    Follows ISP with 2 methods: invoke() and stream().

    Responsibility:
    - Execute compiled graph with initial state
    - Stream execution events
    - Return execution result

    Non-responsibility:
    - Compilation (handled by WorkflowCompilerProtocol)
    - Node execution logic (handled by node executors)
    - State management (handled by StateGraph)
    """

    async def invoke(
        self,
        initial_state: Dict[str, Any],
        *,
        thread_id: Optional[str] = None,
        checkpoint: Optional[str] = None,
    ) -> "ExecutionResultProtocol":
        """Execute the compiled workflow.

        Args:
            initial_state: Initial workflow state
            thread_id: Optional thread ID for checkpointing
            checkpoint: Optional checkpoint name to resume from

        Returns:
            ExecutionResultProtocol with execution outcome

        Example:
            result = await compiled.invoke({"query": "search term"})
            print(result.final_state)
        """
        ...

    async def stream(
        self,
        initial_state: Dict[str, Any],
        *,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator["ExecutionEventProtocol"]:
        """Stream execution events from the compiled workflow.

        Args:
            initial_state: Initial workflow state
            thread_id: Optional thread ID for checkpointing

        Yields:
            ExecutionEventProtocol: Execution events as they occur

        Example:
            async for event in compiled.stream({"query": "term"}):
                print(f"{event.node_id}: {event.event_type}")
        """
        ...

    @property
    def graph(self) -> "StateGraph":
        """Get the underlying StateGraph.

        Returns:
            StateGraph instance

        Example:
            graph = compiled.graph
            # Inspect graph structure
        """
        ...


class ExecutionResultProtocol(Protocol):
    """Protocol for workflow execution results.

    Minimal protocol for execution results.
    Follows ISP with 2 properties.

    Responsibility:
    - Provide final state after execution
    - Provide execution metrics

    Non-responsibility:
    - Result formatting (handled by caller)
    - Result storage (handled by caller)
    """

    @property
    def final_state(self) -> "WorkflowState":
        """Get final workflow state after execution.

        Returns:
            WorkflowState: Final state dict
        """
        ...

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get execution metrics.

        Returns:
            Dict with metrics like:
            - duration_seconds: Execution time
            - nodes_executed: Number of nodes executed
            - total_tool_calls: Total tool calls across all agents
        """
        ...


class ExecutionEventProtocol(Protocol):
    """Protocol for workflow execution events.

    Minimal protocol for streaming execution events.
    Follows ISP with 3 properties.

    Responsibility:
    - Provide event data during streaming execution

    Non-responsibility:
    - Event filtering (handled by caller)
    - Event storage (handled by caller)
    """

    @property
    def node_id(self) -> str:
        """Get node ID that generated this event.

        Returns:
            str: Node identifier
        """
        ...

    @property
    def event_type(self) -> str:
        """Get event type.

        Returns:
            str: Event type (e.g., "node_start", "node_complete", "error")
        """
        ...

    @property
    def data(self) -> Dict[str, Any]:
        """Get event data.

        Returns:
            Dict with event-specific data
        """
        ...


class NodeExecutorFactoryProtocol(Protocol):
    """Protocol for node executor factories.

    Minimal protocol for creating node executors.
    Follows ISP with 2 methods: create_executor() and register_executor_type().

    Responsibility:
    - Map node types to executor functions
    - Create executor functions for workflow nodes
    - Support registration of custom node types

    Non-responsibility:
    - Node execution logic (handled by NodeExecutorProtocol)
    - Workflow compilation (handled by WorkflowCompilerProtocol)

    Design:
    - Factory pattern for extensibility (OCP compliance)
    - Open for extension: Register new node types via register_executor_type()
    - Closed for modification: Don't modify factory to add node types
    """

    def create_executor(self, node: "WorkflowNode") -> Callable:
        """Create an executor function for a workflow node.

        Args:
            node: Workflow node definition

        Returns:
            Callable: Async executor function that takes WorkflowState and
                     returns WorkflowState

        Raises:
            ValueError: If node type is not supported

        Example:
            factory: NodeExecutorFactoryProtocol = get_factory()
            executor_fn = factory.create_executor(agent_node)
            result_state = await executor_fn(initial_state)
        """
        ...

    def register_executor_type(
        self,
        node_type: str,
        executor_factory: Callable[["WorkflowNode"], Callable],
        *,
        replace: bool = False,
    ) -> None:
        """Register a custom node executor type.

        Args:
            node_type: Node type identifier (e.g., "custom_compute")
            executor_factory: Factory function that creates executor
            replace: Whether to replace existing registration

        Raises:
            ValueError: If node_type already exists and replace=False

        Example:
            def create_custom_executor(node):
                async def execute(state):
                    # Custom execution logic
                    return state
                return execute

            factory.register_executor_type("custom_compute", create_custom_executor)
        """
        ...

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
        ...


class NodeExecutorProtocol(Protocol):
    """Protocol for individual node executors.

    Minimal protocol for executing workflow nodes.
    Follows ISP with 2 methods: execute() and supports_node_type().

    Responsibility:
    - Execute a single workflow node
    - Report which node types are supported

    Non-responsibility:
    - Workflow compilation (handled by WorkflowCompilerProtocol)
    - Workflow execution coordination (handled by WorkflowExecutorProtocol)

    Design:
    - SRP compliance: Each executor handles ONE node type
    - OCP compliance: Add new node types by creating new executor classes
    """

    async def execute(self, node: "WorkflowNode", state: "WorkflowState") -> "WorkflowState":
        """Execute a workflow node.

        Args:
            node: Workflow node definition to execute
            state: Current workflow state

        Returns:
            WorkflowState: Updated state after node execution

        Raises:
            Exception: If node execution fails

        Example:
            executor: NodeExecutorProtocol = AgentNodeExecutor()
            new_state = await executor.execute(agent_node, current_state)
        """
        ...

    def supports_node_type(self, node_type: str) -> bool:
        """Check if this executor supports the given node type.

        Args:
            node_type: Node type identifier

        Returns:
            bool: True if this executor handles the node type

        Example:
            if executor.supports_node_type("agent"):
                result = await executor.execute(agent_node, state)
        """
        ...


class ExecutionContextProtocol(Protocol):
    """Protocol for execution context passed to node executors.

    Minimal protocol for execution context.
    Follows ISP with 3 properties.

    Responsibility:
    - Provide orchestrator access for agent nodes
    - Provide execution settings
    - Provide service container access

    Non-responsibility:
    - State management (handled by StateGraph)
    - Node execution logic (handled by node executors)
    """

    @property
    def orchestrator(self) -> Any:
        """Get orchestrator for agent execution.

        Returns:
            AgentOrchestrator instance (or None for non-agent nodes)

        Note:
            Returns Any to avoid circular import. In practice, this is
            an AgentOrchestrator instance.
        """
        ...

    @property
    def settings(self) -> Any:
        """Get execution settings.

        Returns:
            Settings object

        Note:
            Returns Any to avoid circular import. In practice, this is
            a Settings dataclass.
        """
        ...

    @property
    def services(self) -> Any:
        """Get service container.

        Returns:
            ServiceContainer instance

        Note:
            Returns Any to avoid circular import. In practice, this is
            a ServiceContainer instance.
        """
        ...


__all__ = [
    "WorkflowCompilerProtocol",
    "CompiledGraphProtocol",
    "ExecutionResultProtocol",
    "ExecutionEventProtocol",
    "NodeExecutorFactoryProtocol",
    "NodeExecutorProtocol",
    "ExecutionContextProtocol",
]
