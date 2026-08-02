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

"""WorkflowEngine - High-level facade for workflow execution.

This module provides a unified facade over the workflow execution system,
promoting workflow capabilities from victor/workflows/ to the framework layer.

Design Pattern: Facade + Coordinator
====================================
WorkflowEngine provides a simplified interface to the complex workflow
subsystem by delegating to focused coordinators that each handle a single
domain (SRP compliance).

Architecture:
    WorkflowEngine (Facade)
    ├── YAMLWorkflowCoordinator     # execute_yaml(), stream_yaml()
    ├── GraphTurnExecutor   # execute_graph(), stream_graph()
    ├── HITLCoordinator             # execute_with_hitl()
    └── CacheCoordinator            # enable_caching(), clear_cache()

Key Features:
- Create workflows from Python code or YAML
- Execute workflows with optional HITL (Human-in-the-Loop)
- Stream workflow events for real-time UI updates
- Cache workflow results for efficiency
- Checkpoint workflow state for recovery

Usage:
    from victor.framework.workflow_engine import WorkflowEngine

    # Create engine
    engine = WorkflowEngine()

    # Execute a workflow from YAML
    result = await engine.execute_yaml(
        "path/to/workflow.yaml",
        initial_state={"input": "data"},
    )

    # Execute with streaming
    async for event in engine.stream_yaml("workflow.yaml", state):
        print(f"{event.node_id}: {event.event_type}")

    # Execute StateGraph directly
    from victor.framework import StateGraph
    graph = StateGraph(MyState)
    # ... configure graph ...
    result = await engine.execute_graph(graph.compile(), initial_state)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)
from enum import Enum

if TYPE_CHECKING:
    from victor.framework.graph import CompiledGraph, StateGraph
    from victor.framework.coordinators import (
        YAMLWorkflowCoordinator,
        GraphTurnExecutor,
        HITLCoordinator,
        CacheCoordinator,
    )
    from victor.framework.coordinators.protocols import (
        IStreamingExecutor,
        IWorkflowExecutor,
    )
    from victor.workflows.context import WorkflowResult
    from victor.workflows.streaming import WorkflowStreamChunk
    from victor.workflows.hitl import HITLHandler, HITLExecutor
    from victor.workflows.cache import WorkflowCacheManager
    from victor.workflows.definition import WorkflowDefinition
    from victor.workflows.graph_dsl import WorkflowGraph
    from victor.workflows.node_runners import NodeRunnerRegistry
    from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

logger = logging.getLogger(__name__)

from victor.workflows.runtime_executor_factory import (
    create_legacy_streaming_workflow_executor,
    create_legacy_workflow_executor,
)

StateType = TypeVar("StateType", bound=Dict[str, Any])


# =============================================================================
# Error Types
# =============================================================================


class WorkflowErrorType(str, Enum):
    """Typed error categories for workflow execution.

    Error types indicate the category of failure:
    - Validation: Workflow definition or input validation failed
    - Execution: Node execution failed during workflow run
    - Timeout: Workflow or node exceeded time limit
    - Cancelled: Workflow was cancelled by user or system
    - Resource: Required resources (CPU, memory, etc.) were unavailable
    """

    VALIDATION = "validation"
    """Workflow definition or input validation failed."""

    EXECUTION = "execution"
    """Node execution failed during workflow run."""

    TIMEOUT = "timeout"
    """Workflow or node exceeded time limit."""

    CANCELLED = "cancelled"
    """Workflow was cancelled by user or system."""

    RESOURCE = "resource"
    """Required resources (CPU, memory, etc.) were unavailable."""


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class WorkflowEngineConfig:
    """Configuration for WorkflowEngine.

    Attributes:
        enable_caching: Whether to cache workflow results.
        cache_ttl_seconds: Cache time-to-live in seconds.
        enable_hitl: Whether to enable HITL for approval nodes.
        hitl_timeout_seconds: Timeout for HITL approval.
        enable_checkpoints: Whether to checkpoint state for recovery.
        max_iterations: Maximum iterations for cyclic workflows.
        enable_streaming: Whether to enable event streaming.
        parallel_execution: Whether to run parallel nodes concurrently.
    """

    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_hitl: bool = False
    hitl_timeout_seconds: int = 300
    enable_checkpoints: bool = True
    max_iterations: int = 100
    enable_streaming: bool = True
    parallel_execution: bool = True

    def __post_init__(self) -> None:
        """Validate workflow engine configuration limits."""
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.cache_ttl_seconds < 0:
            raise ValueError(f"cache_ttl_seconds must be >= 0, got {self.cache_ttl_seconds}")
        if self.hitl_timeout_seconds < 0:
            raise ValueError(f"hitl_timeout_seconds must be >= 0, got {self.hitl_timeout_seconds}")


@dataclass
class WorkflowExecutionResult:
    """Result of workflow execution.

    Attributes:
        success: Whether execution completed successfully.
        final_state: Final workflow state.
        nodes_executed: List of node IDs that were executed.
        duration_seconds: Total execution time.
        error: Error message if execution failed (backward compat).
        error_type: Typed error category using WorkflowErrorType.
        error_message: Detailed error message (modern alternative to error field).
        checkpoints: List of checkpoint IDs created.
        hitl_requests: HITL requests that were made.
        cached: Whether result was from cache.
    """

    success: bool
    final_state: Dict[str, Any] = field(default_factory=dict)
    nodes_executed: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[str] = None  # Backward compat
    error_type: Optional[WorkflowErrorType] = None
    error_message: Optional[str] = None
    checkpoints: List[str] = field(default_factory=list)
    hitl_requests: List[Dict[str, Any]] = field(default_factory=list)
    cached: bool = False


@dataclass
class WorkflowEvent:
    """Event emitted during workflow execution.

    Attributes:
        event_type: Type of event (node_start, node_end, error, hitl, etc.)
        node_id: ID of the node that emitted the event.
        timestamp: Unix timestamp of event.
        data: Event-specific data.
        state_snapshot: Optional snapshot of workflow state.
    """

    event_type: str
    node_id: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    state_snapshot: Optional[Dict[str, Any]] = None


# =============================================================================
# Protocol
# =============================================================================


@runtime_checkable
class WorkflowEngineProtocol(Protocol):
    """Protocol for workflow engine implementations."""

    async def execute_yaml(
        self,
        yaml_path: Union[str, Path],
        initial_state: Dict[str, Any],
        **kwargs: Any,
    ) -> WorkflowExecutionResult:
        """Execute a workflow from YAML file."""
        ...

    async def execute_graph(
        self,
        graph: "CompiledGraph",
        initial_state: Dict[str, Any],
        **kwargs: Any,
    ) -> WorkflowExecutionResult:
        """Execute a compiled StateGraph."""
        ...

    async def stream_yaml(
        self,
        yaml_path: Union[str, Path],
        initial_state: Dict[str, Any],
        **kwargs: Any,
    ) -> AsyncIterator[WorkflowEvent]:
        """Stream events from YAML workflow execution."""
        ...

    async def stream_graph(
        self,
        graph: "CompiledGraph",
        initial_state: Dict[str, Any],
        **kwargs: Any,
    ) -> AsyncIterator[WorkflowEvent]:
        """Stream events from StateGraph execution."""
        ...


# =============================================================================
# WorkflowEngine
# =============================================================================


class WorkflowEngine:
    """High-level facade for workflow execution.

    Provides a unified API for executing workflows defined as:
    - YAML files (declarative)
    - StateGraph objects (programmatic)
    - WorkflowDefinition objects (legacy)

    Features:
    - Automatic HITL integration for approval nodes
    - Result caching for repeated executions
    - Checkpointing for recovery
    - Event streaming for real-time updates
    - Parallel node execution
    """

    def __init__(
        self,
        config: Optional[WorkflowEngineConfig] = None,
        hitl_handler: Optional["HITLHandler"] = None,
        cache_manager: Optional["WorkflowCacheManager"] = None,
        runner_registry: Optional["NodeRunnerRegistry"] = None,
    ) -> None:
        """Initialize WorkflowEngine.

        Args:
            config: Engine configuration.
            hitl_handler: Custom HITL handler for approval nodes.
            cache_manager: Custom cache manager for results.
            runner_registry: Optional NodeRunner registry for unified execution.
        """
        self._config = config or WorkflowEngineConfig()
        self._hitl_handler = hitl_handler
        self._cache_manager = cache_manager
        self._runner_registry = runner_registry

        # Lazy-loaded coordinators (SRP split)
        self._yaml_coordinator: Optional["YAMLWorkflowCoordinator"] = None
        self._graph_coordinator: Optional["GraphTurnExecutor"] = None
        self._hitl_coordinator: Optional["HITLCoordinator"] = None
        self._cache_coordinator: Optional["CacheCoordinator"] = None

        # Lazy-loaded executors (for backward compatibility)
        self._executor: Optional["IWorkflowExecutor"] = None
        self._streaming_executor: Optional["IStreamingExecutor"] = None
        self._hitl_executor: Optional["HITLExecutor"] = None

        # Lazy-loaded unified compiler for consistent compilation and caching
        self._unified_compiler: Optional["UnifiedWorkflowCompiler"] = None

    def _emit_workflow_event(
        self,
        event_type: str,
        workflow_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Emit a workflow event to the EventBus for observability.

        Args:
            event_type: Type of event (workflow_started, workflow_completed, etc.)
            workflow_id: Workflow execution identifier
            data: Event payload data
        """
        try:
            from victor.core.events import get_observability_bus as get_event_bus

            bus = get_event_bus()
            bus.emit_lifecycle_event(
                event_type,
                {
                    "workflow_id": workflow_id,
                    "source": "WorkflowEngine",
                    **data,
                },
            )
        except Exception as e:
            # Don't let event emission failures break workflow execution
            logger.debug(f"Failed to emit {event_type} event: {e}")

    @property
    def config(self) -> WorkflowEngineConfig:
        """Get the engine configuration."""
        return self._config

    # =========================================================================
    # Execution Methods
    # =========================================================================

    async def execute_yaml(
        self,
        yaml_path: Union[str, Path],
        initial_state: Optional[Dict[str, Any]] = None,
        workflow_name: Optional[str] = None,
        condition_registry: Optional[Dict[str, Callable]] = None,
        transform_registry: Optional[Dict[str, Callable]] = None,
        use_unified_compiler: bool = True,
        **kwargs: Any,
    ) -> WorkflowExecutionResult:
        """Execute a workflow from YAML file.

        Uses UnifiedWorkflowCompiler for consistent compilation and caching,
        then executes via CompiledGraph.invoke().

        Args:
            yaml_path: Path to YAML workflow file.
            initial_state: Initial workflow state.
            workflow_name: Specific workflow to load from file.
            condition_registry: Custom condition functions.
            transform_registry: Custom transform functions.
            use_unified_compiler: Whether to use unified compiler (default True).
                Set to False for backward compatibility with coordinator.
            **kwargs: Additional execution parameters.

        Returns:
            WorkflowExecutionResult with final state and metadata.
        """
        import time

        if use_unified_compiler:
            # Use unified compiler for consistent compilation and caching
            start_time = time.time()
            try:
                compiler = self._get_unified_compiler()
                compiled = compiler.compile_yaml(
                    Path(yaml_path),
                    workflow_name=workflow_name,
                    condition_registry=condition_registry,
                    transform_registry=transform_registry,
                )

                # Extract thread_id from kwargs if provided for checkpointing
                thread_id = kwargs.pop("thread_id", None)

                # Execute via CompiledGraph.invoke()
                result = await compiled.invoke(
                    initial_state or {},
                    thread_id=thread_id,
                    **kwargs,
                )
                duration = time.time() - start_time

                # Handle polymorphic result types
                # Result can be GraphExecutionResult object or dict
                from victor.framework.graph import GraphExecutionResult

                if isinstance(result, GraphExecutionResult):
                    final_state = (
                        result.state if isinstance(result.state, dict) else {"result": result.state}
                    )
                    nodes_executed = result.node_history
                    success = result.success
                    error = result.error
                elif isinstance(result, dict):
                    # Direct dict result
                    final_state = result
                    nodes_executed = (
                        result.pop("_nodes_executed", []) if "_nodes_executed" in result else []
                    )
                    success = True
                    error = None
                else:
                    # Fallback for other result types
                    final_state = {"result": result}
                    nodes_executed = []
                    success = True
                    error = None

                return WorkflowExecutionResult(
                    success=success,
                    final_state=final_state,
                    nodes_executed=nodes_executed,
                    duration_seconds=duration,
                    error=error,
                )

            except Exception as e:
                logger.error(f"YAML workflow execution failed: {e}")
                return WorkflowExecutionResult(
                    success=False,
                    error=str(e),
                    duration_seconds=time.time() - start_time,
                )
        else:
            # Fall back to coordinator for backward compatibility
            coordinator = self._get_yaml_coordinator()
            return await coordinator.execute(
                yaml_path=yaml_path,
                initial_state=initial_state,
                workflow_name=workflow_name,
                condition_registry=condition_registry,
                transform_registry=transform_registry,
                **kwargs,
            )

    async def execute_graph(
        self,
        graph: "CompiledGraph",
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> WorkflowExecutionResult:
        """Execute a compiled StateGraph.

        Delegates to GraphTurnExecutor for SRP-compliant execution
        with LSP-compliant polymorphic result handling.

        Args:
            graph: Compiled StateGraph to execute.
            initial_state: Initial workflow state.
            **kwargs: Additional execution parameters.

        Returns:
            WorkflowExecutionResult with final state and metadata.
        """
        coordinator = self._get_graph_coordinator()
        return await coordinator.execute(
            graph=graph,
            initial_state=initial_state,
            **kwargs,
        )

    async def execute_definition(
        self,
        workflow: "WorkflowDefinition",
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> WorkflowExecutionResult:
        """Execute a WorkflowDefinition.

        Args:
            workflow: WorkflowDefinition to execute.
            initial_state: Initial workflow state.
            **kwargs: Additional execution parameters.

        Returns:
            WorkflowExecutionResult with final state and metadata.
        """
        import time
        import uuid

        start_time = time.time()
        workflow_id = kwargs.get("workflow_id") or uuid.uuid4().hex

        # Emit workflow started event
        self._emit_workflow_event(
            "workflow_started",
            workflow_id,
            {
                "workflow_name": getattr(workflow, "name", "unknown"),
                "node_count": len(getattr(workflow, "nodes", [])),
            },
        )

        try:
            executor = self._get_executor()
            result = await executor.execute(
                workflow,
                initial_context=initial_state or {},
            )

            duration = time.time() - start_time

            # Emit workflow completed event
            self._emit_workflow_event(
                "workflow_completed",
                workflow_id,
                {
                    "success": result.success,
                    "duration": duration,
                    "nodes_executed": result.nodes_executed,
                },
            )

            return WorkflowExecutionResult(
                success=result.success,
                final_state=result.final_state,
                nodes_executed=result.nodes_executed,
                duration_seconds=duration,
                error=result.error if not result.success else None,
            )

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            duration = time.time() - start_time

            # Emit workflow error event
            self._emit_workflow_event(
                "workflow_error",
                workflow_id,
                {
                    "error": str(e),
                    "duration": duration,
                },
            )

            return WorkflowExecutionResult(
                success=False,
                error=str(e),
                duration_seconds=duration,
            )

    async def execute_workflow_graph(
        self,
        graph: "WorkflowGraph",
        initial_state: Optional[Dict[str, Any]] = None,
        use_node_runners: bool = False,
        **kwargs: Any,
    ) -> WorkflowExecutionResult:
        """Execute a WorkflowGraph via CompiledGraph (unified execution path).

        Delegates to GraphTurnExecutor for SRP-compliant execution.
        This method compiles a WorkflowGraph to CompiledGraph and executes
        it through the single CompiledGraph.invoke() engine, providing a
        unified execution path for all workflow types.

        Args:
            graph: WorkflowGraph to compile and execute.
            initial_state: Initial workflow state.
            use_node_runners: Whether to use NodeRunner protocol for execution.
            **kwargs: Additional execution parameters.

        Returns:
            WorkflowExecutionResult with final state and metadata.

        Example:
            from victor.workflows.graph_dsl import WorkflowGraph, State

            @dataclass
            class MyState(State):
                value: int = 0

            graph = WorkflowGraph(MyState)
            graph.add_node("process", lambda s: s)
            graph.set_entry_point("process")
            graph.set_finish_point("process")

            result = await engine.execute_workflow_graph(graph, {"value": 42})
        """
        coordinator = self._get_graph_coordinator()
        return await coordinator.execute_workflow_graph(
            graph=graph,
            initial_state=initial_state,
            use_node_runners=use_node_runners,
            **kwargs,
        )

    async def execute_definition_compiled(
        self,
        workflow: "WorkflowDefinition",
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> WorkflowExecutionResult:
        """Execute a WorkflowDefinition via CompiledGraph (unified execution path).

        Delegates to GraphTurnExecutor for SRP-compliant execution.
        This method compiles a WorkflowDefinition to CompiledGraph and executes
        it through the single CompiledGraph.invoke() engine.

        Args:
            workflow: WorkflowDefinition to compile and execute.
            initial_state: Initial workflow state.
            **kwargs: Additional execution parameters.

        Returns:
            WorkflowExecutionResult with final state and metadata.
        """
        coordinator = self._get_graph_coordinator()
        return await coordinator.execute_definition_compiled(
            workflow=workflow,
            initial_state=initial_state,
            **kwargs,
        )

    def set_runner_registry(self, registry: "NodeRunnerRegistry") -> None:
        """Set the NodeRunner registry for unified execution.

        Args:
            registry: NodeRunnerRegistry with configured runners.
        """
        self._runner_registry = registry
        # Reset unified compiler to use new registry
        if self._unified_compiler is not None:
            self._unified_compiler.set_runner_registry(registry)
        # Update graph coordinator if it exists
        if self._graph_coordinator is not None:
            self._graph_coordinator.set_runner_registry(registry)

    # =========================================================================
    # Streaming Methods
    # =========================================================================

    async def stream_yaml(
        self,
        yaml_path: Union[str, Path],
        initial_state: Optional[Dict[str, Any]] = None,
        workflow_name: Optional[str] = None,
        condition_registry: Optional[Dict[str, Callable]] = None,
        transform_registry: Optional[Dict[str, Callable]] = None,
        use_unified_compiler: bool = True,
        **kwargs: Any,
    ) -> AsyncIterator[WorkflowEvent]:
        """Stream events from YAML workflow execution.

        Uses UnifiedWorkflowCompiler for consistent compilation and caching,
        then streams via CompiledGraph.stream().

        Args:
            yaml_path: Path to YAML workflow file.
            initial_state: Initial workflow state.
            workflow_name: Specific workflow to load from file.
            condition_registry: Custom condition functions.
            transform_registry: Custom transform functions.
            use_unified_compiler: Whether to use unified compiler (default True).
                Set to False for backward compatibility with coordinator.
            **kwargs: Additional execution parameters.

        Yields:
            WorkflowEvent for each execution step.
        """
        import time

        if use_unified_compiler:
            # Use unified compiler for consistent compilation and caching
            try:
                compiler = self._get_unified_compiler()
                compiled = compiler.compile_yaml(
                    Path(yaml_path),
                    workflow_name=workflow_name,
                    condition_registry=condition_registry,
                    transform_registry=transform_registry,
                )

                # Stream via CompiledGraph.stream()
                async for event in compiled.stream(initial_state or {}, **kwargs):
                    # Convert CompiledGraph events to WorkflowEvent format
                    if isinstance(event, dict):
                        yield WorkflowEvent(
                            event_type=event.get("event_type", "state_update"),
                            node_id=event.get("node_id", ""),
                            timestamp=time.time(),
                            data=event.get("data", {}),
                            state_snapshot=event.get("state", None),
                        )
                    else:
                        # Assume it's already a compatible event type
                        yield WorkflowEvent(
                            event_type=getattr(event, "event_type", "state_update"),
                            node_id=getattr(event, "node_id", ""),
                            timestamp=time.time(),
                            data=getattr(event, "data", {}),
                            state_snapshot=getattr(event, "state", None),
                        )

            except Exception as e:
                logger.error(f"Streaming YAML workflow failed: {e}")
                yield WorkflowEvent(
                    event_type="error",
                    node_id="",
                    timestamp=time.time(),
                    data={"error": str(e)},
                )
        else:
            # Fall back to coordinator for backward compatibility
            coordinator = self._get_yaml_coordinator()
            async for event in coordinator.stream(
                yaml_path=yaml_path,
                initial_state=initial_state,
                workflow_name=workflow_name,
                condition_registry=condition_registry,
                transform_registry=transform_registry,
                **kwargs,
            ):
                yield event

    async def stream_graph(
        self,
        graph: "CompiledGraph",
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[WorkflowEvent]:
        """Stream events from StateGraph execution.

        Delegates to GraphTurnExecutor for SRP-compliant streaming.

        Args:
            graph: Compiled StateGraph to execute.
            initial_state: Initial workflow state.
            **kwargs: Additional execution parameters.

        Yields:
            WorkflowEvent for each execution step.
        """
        coordinator = self._get_graph_coordinator()
        async for event in coordinator.stream(
            graph=graph,
            initial_state=initial_state,
            **kwargs,
        ):
            yield event

    # =========================================================================
    # HITL Integration
    # =========================================================================

    def set_hitl_handler(self, handler: "HITLHandler") -> None:
        """Set custom HITL handler.

        Args:
            handler: HITLHandler for approval nodes.
        """
        self._hitl_handler = handler
        # Reset executor and coordinator to use new handler
        self._hitl_executor = None
        if self._hitl_coordinator is not None:
            self._hitl_coordinator.set_handler(handler)

    async def execute_with_hitl(
        self,
        yaml_path: Union[str, Path],
        initial_state: Optional[Dict[str, Any]] = None,
        approval_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
        **kwargs: Any,
    ) -> WorkflowExecutionResult:
        """Execute workflow with HITL approval nodes.

        Delegates to HITLCoordinator for SRP-compliant execution.

        Args:
            yaml_path: Path to YAML workflow file.
            initial_state: Initial workflow state.
            approval_callback: Callback for approval decisions.
            **kwargs: Additional execution parameters.

        Returns:
            WorkflowExecutionResult with HITL request history.
        """
        coordinator = self._get_hitl_coordinator()
        return await coordinator.execute(
            yaml_path=yaml_path,
            initial_state=initial_state,
            approval_callback=approval_callback,
            **kwargs,
        )

    # =========================================================================
    # Caching
    # =========================================================================

    def enable_caching(self, ttl_seconds: int = 3600) -> None:
        """Enable result caching.

        Delegates to CacheCoordinator for SRP-compliant cache management.

        Args:
            ttl_seconds: Cache time-to-live.
        """
        self._config.enable_caching = True
        self._config.cache_ttl_seconds = ttl_seconds
        coordinator = self._get_cache_coordinator()
        coordinator.enable_caching(ttl_seconds=ttl_seconds)

    def disable_caching(self) -> None:
        """Disable result caching.

        Delegates to CacheCoordinator for SRP-compliant cache management.
        """
        self._config.enable_caching = False
        coordinator = self._get_cache_coordinator()
        coordinator.disable_caching()

    def clear_cache(self) -> None:
        """Clear all cached results.

        Delegates to CacheCoordinator for SRP-compliant cache management.
        """
        coordinator = self._get_cache_coordinator()
        coordinator.clear_cache()
        # Also clear the cache manager directly for backward compatibility
        if self._cache_manager:
            self._cache_manager.clear_all()

    def clear_workflow_cache(self) -> int:
        """Clear all workflow caches via unified compiler.

        Clears both definition cache (parsed YAML workflows) and
        execution cache (workflow results).

        Returns:
            Total number of cache entries cleared.
        """
        compiler = self._get_unified_compiler()
        return compiler.clear_cache()

    def get_workflow_cache_stats(self) -> Dict[str, Any]:
        """Get workflow cache statistics.

        Returns comprehensive cache statistics including:
        - definition_cache: Stats for parsed workflow definitions
        - execution_cache: Stats for workflow execution results
        - caching_enabled: Whether caching is currently enabled

        Returns:
            Dictionary with cache statistics.
        """
        compiler = self._get_unified_compiler()
        return compiler.get_cache_stats()

    def invalidate_yaml_cache(self, yaml_path: Union[str, Path]) -> int:
        """Invalidate cached definitions for a specific YAML file.

        Use this when a YAML file has been modified and the cache
        should be refreshed.

        Args:
            yaml_path: Path to YAML file to invalidate.

        Returns:
            Number of cache entries invalidated.
        """
        compiler = self._get_unified_compiler()
        return compiler.invalidate_yaml(yaml_path)

    # =========================================================================
    # Coordinator Getters (SRP Split)
    # =========================================================================

    def _get_yaml_coordinator(self) -> "YAMLWorkflowCoordinator":
        """Get or create YAML workflow coordinator."""
        if self._yaml_coordinator is None:
            from victor.framework.coordinators import YAMLWorkflowCoordinator

            self._yaml_coordinator = YAMLWorkflowCoordinator()
        return self._yaml_coordinator

    def _get_graph_coordinator(self) -> "GraphTurnExecutor":
        """Get or create graph execution coordinator."""
        if self._graph_coordinator is None:
            from victor.framework.coordinators import GraphTurnExecutor

            self._graph_coordinator = GraphTurnExecutor(runner_registry=self._runner_registry)
        return self._graph_coordinator

    def _get_hitl_coordinator(self) -> "HITLCoordinator":
        """Get or create HITL coordinator."""
        if self._hitl_coordinator is None:
            from victor.framework.coordinators import HITLCoordinator

            self._hitl_coordinator = HITLCoordinator(
                handler=self._hitl_handler,
                timeout_seconds=self._config.hitl_timeout_seconds,
            )
        return self._hitl_coordinator

    def _get_cache_coordinator(self) -> "CacheCoordinator":
        """Get or create cache coordinator."""
        if self._cache_coordinator is None:
            from victor.framework.coordinators import CacheCoordinator

            self._cache_coordinator = CacheCoordinator(
                cache_manager=self._cache_manager,
            )
        return self._cache_coordinator

    def _get_unified_compiler(self) -> "UnifiedWorkflowCompiler":
        """Get or create the unified compiler.

        **Architecture Note**: UnifiedWorkflowCompiler is the canonical compiler API.
        WorkflowEngine uses it for:
        - Two-level caching (definition + execution) for performance
        - Cache management APIs (clear_cache, get_cache_stats, invalidate_yaml)
        - Runner registry integration for custom node execution
        - Escape hatch integration (condition_registry, transform_registry)

        For DI-friendly compile-only usage without caching or execution,
        see WorkflowCompiler in victor.workflows.compiler.

        Returns:
            UnifiedWorkflowCompiler instance with shared caches.
        """
        if self._unified_compiler is None:
            from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
            from victor.workflows.cache import (
                get_workflow_definition_cache,
                get_workflow_cache_manager,
            )

            self._unified_compiler = UnifiedWorkflowCompiler(
                definition_cache=get_workflow_definition_cache(),
                execution_cache=get_workflow_cache_manager(),
                runner_registry=self._runner_registry,
                enable_caching=self._config.enable_caching,
            )
        return self._unified_compiler

    # =========================================================================
    # Helpers (for backward compatibility)
    # =========================================================================

    def _get_executor(self) -> "IWorkflowExecutor":
        """Get or create workflow executor."""
        if self._executor is None:
            self._executor = create_legacy_workflow_executor()
        return self._executor

    def _get_streaming_executor(self) -> "IStreamingExecutor":
        """Get or create streaming executor."""
        if self._streaming_executor is None:
            self._streaming_executor = create_legacy_streaming_workflow_executor()
        return self._streaming_executor


# =============================================================================
# Factory Functions
# =============================================================================


def create_workflow_engine(
    config: Optional[WorkflowEngineConfig] = None,
    hitl_handler: Optional["HITLHandler"] = None,
    cache_manager: Optional["WorkflowCacheManager"] = None,
) -> WorkflowEngine:
    """Create a WorkflowEngine with configuration.

    Args:
        config: Engine configuration.
        hitl_handler: Custom HITL handler.
        cache_manager: Custom cache manager.

    Returns:
        Configured WorkflowEngine instance.
    """
    return WorkflowEngine(
        config=config,
        hitl_handler=hitl_handler,
        cache_manager=cache_manager,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_yaml_workflow(
    yaml_path: Union[str, Path],
    initial_state: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> WorkflowExecutionResult:
    """Convenience function to run a YAML workflow.

    Args:
        yaml_path: Path to YAML workflow file.
        initial_state: Initial state.
        **kwargs: Additional parameters.

    Returns:
        WorkflowExecutionResult from workflow execution.
    """
    engine = create_workflow_engine()
    return await engine.execute_yaml(yaml_path, initial_state, **kwargs)


async def run_graph_workflow(
    graph: "CompiledGraph",
    initial_state: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> WorkflowExecutionResult:
    """Convenience function to run a StateGraph workflow.

    Args:
        graph: Compiled StateGraph.
        initial_state: Initial state.
        **kwargs: Additional parameters.

    Returns:
        WorkflowExecutionResult from workflow execution.
    """
    engine = create_workflow_engine()
    return await engine.execute_graph(graph, initial_state, **kwargs)
