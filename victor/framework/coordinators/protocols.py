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

"""Protocol interfaces for workflow coordinators.

Defines the contracts that coordinators must implement, following the
Dependency Inversion Principle (DIP) by depending on abstractions.

These protocols enable:
- Protocol-based dependency injection
- Easy testing with mock implementations
- Clear separation of concerns
- Type-safe coordinator composition
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.framework.graph import CompiledGraph
    from victor.framework.workflow_engine import WorkflowExecutionResult, WorkflowEvent
    from victor.workflows.definition import WorkflowDefinition
    from victor.workflows.hitl import HITLHandler


# =============================================================================
# Core Execution Protocols
# =============================================================================


@runtime_checkable
class IWorkflowExecutor(Protocol):
    """Protocol for workflow execution.

    Implementations must provide an execute method that runs a workflow
    and returns a structured result.
    """

    async def execute(
        self,
        workflow: Any,
        initial_state: Dict[str, Any],
        **kwargs: Any,
    ) -> "WorkflowExecutionResult":
        """Execute a workflow.

        Args:
            workflow: The workflow to execute (type varies by implementation)
            initial_state: Initial state dictionary
            **kwargs: Additional execution parameters

        Returns:
            WorkflowExecutionResult with final state and metadata
        """
        ...


@runtime_checkable
class IStreamingExecutor(Protocol):
    """Protocol for streaming workflow execution.

    Implementations must provide a stream method that yields events
    during workflow execution.
    """

    async def stream(
        self,
        workflow: Any,
        initial_state: Dict[str, Any],
        **kwargs: Any,
    ) -> AsyncIterator["WorkflowEvent"]:
        """Stream workflow execution events.

        Args:
            workflow: The workflow to execute
            initial_state: Initial state dictionary
            **kwargs: Additional execution parameters

        Yields:
            WorkflowEvent for each execution step
        """
        ...


# =============================================================================
# YAML Workflow Protocols
# =============================================================================


@runtime_checkable
class IYAMLLoader(Protocol):
    """Protocol for loading workflows from YAML files.

    Implementations handle YAML parsing, caching, and registry integration.
    """

    def load_workflow_from_file(
        self,
        path: Union[str, Path],
        workflow_name: Optional[str] = None,
        condition_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        transform_registry: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> "WorkflowDefinition":
        """Load a workflow definition from a YAML file.

        Args:
            path: Path to the YAML file
            workflow_name: Specific workflow to load (if file contains multiple)
            condition_registry: Custom condition functions for escape hatches
            transform_registry: Custom transform functions for escape hatches

        Returns:
            Parsed WorkflowDefinition

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the workflow name isn't found
        """
        ...


@runtime_checkable
class IYAMLWorkflowCoordinator(Protocol):
    """Protocol for YAML workflow coordination.

    Combines loading, execution, and streaming for YAML-defined workflows.
    """

    async def execute(
        self,
        yaml_path: Union[str, Path],
        initial_state: Optional[Dict[str, Any]] = None,
        workflow_name: Optional[str] = None,
        condition_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        transform_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        **kwargs: Any,
    ) -> "WorkflowExecutionResult":
        """Execute a YAML-defined workflow.

        Args:
            yaml_path: Path to YAML workflow file
            initial_state: Initial workflow state
            workflow_name: Specific workflow to load
            condition_registry: Custom condition functions
            transform_registry: Custom transform functions
            **kwargs: Additional execution parameters

        Returns:
            WorkflowExecutionResult with final state and metadata
        """
        ...

    async def stream(
        self,
        yaml_path: Union[str, Path],
        initial_state: Optional[Dict[str, Any]] = None,
        workflow_name: Optional[str] = None,
        condition_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        transform_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator["WorkflowEvent"]:
        """Stream events from YAML workflow execution.

        Args:
            yaml_path: Path to YAML workflow file
            initial_state: Initial workflow state
            workflow_name: Specific workflow to load
            condition_registry: Custom condition functions
            transform_registry: Custom transform functions
            **kwargs: Additional execution parameters

        Yields:
            WorkflowEvent for each execution step
        """
        ...


# =============================================================================
# Graph Execution Protocols
# =============================================================================


@runtime_checkable
class IGraphExecutor(Protocol):
    """Protocol for StateGraph/CompiledGraph execution.

    Implementations handle graph invocation with LSP-compliant result handling.
    """

    async def execute(
        self,
        graph: "CompiledGraph",
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "WorkflowExecutionResult":
        """Execute a compiled StateGraph.

        Args:
            graph: Compiled StateGraph to execute
            initial_state: Initial workflow state
            **kwargs: Additional execution parameters

        Returns:
            WorkflowExecutionResult with final state and metadata
        """
        ...

    async def stream(
        self,
        graph: "CompiledGraph",
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator["WorkflowEvent"]:
        """Stream events from StateGraph execution.

        Args:
            graph: Compiled StateGraph to execute
            initial_state: Initial workflow state
            **kwargs: Additional execution parameters

        Yields:
            WorkflowEvent for each execution step
        """
        ...


# =============================================================================
# HITL Protocols
# =============================================================================


@runtime_checkable
class IHITLExecutor(Protocol):
    """Protocol for Human-in-the-Loop workflow execution.

    Implementations handle HITL integration for approval nodes.
    """

    async def execute_with_hitl(
        self,
        yaml_path: Union[str, Path],
        initial_state: Optional[Dict[str, Any]] = None,
        approval_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
        **kwargs: Any,
    ) -> "WorkflowExecutionResult":
        """Execute workflow with HITL approval nodes.

        Args:
            yaml_path: Path to YAML workflow file
            initial_state: Initial workflow state
            approval_callback: Callback for approval decisions
            **kwargs: Additional execution parameters

        Returns:
            WorkflowExecutionResult with HITL request history
        """
        ...

    def set_handler(self, handler: "HITLHandler") -> None:
        """Set custom HITL handler.

        Args:
            handler: HITLHandler for approval nodes
        """
        ...


# =============================================================================
# Cache Management Protocols
# =============================================================================


@runtime_checkable
class ICacheManager(Protocol):
    """Protocol for workflow cache management.

    Implementations handle cache configuration and clearing.
    """

    def enable_caching(self, ttl_seconds: int = 3600) -> None:
        """Enable result caching.

        Args:
            ttl_seconds: Cache time-to-live
        """
        ...

    def disable_caching(self) -> None:
        """Disable result caching."""
        ...

    def clear_cache(self) -> None:
        """Clear all cached results."""
        ...

    @property
    def caching_enabled(self) -> bool:
        """Check if caching is enabled."""
        ...


__all__ = [
    "IWorkflowExecutor",
    "IStreamingExecutor",
    "IYAMLLoader",
    "IYAMLWorkflowCoordinator",
    "IGraphExecutor",
    "IHITLExecutor",
    "ICacheManager",
]
