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

"""YAML Workflow Coordinator.

Handles loading, execution, and streaming of YAML-defined workflows.
This coordinator encapsulates all YAML workflow-related functionality,
following the Single Responsibility Principle.

Features:
- Load workflows from YAML files via UnifiedWorkflowCompiler
- Execute workflows with two-level caching (definition + node)
- Stream execution events for real-time UI updates
- Support for condition and transform registries (escape hatches)
- Checkpointing support for resumable workflows
- Consistent execution via UnifiedWorkflowCompiler
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

if TYPE_CHECKING:
    from victor.framework.coordinators.protocols import (
        IStreamingExecutor,
        IWorkflowExecutor,
    )
    from victor.framework.workflow_engine import WorkflowExecutionResult, WorkflowEvent
    from victor.workflows.cache import WorkflowDefinitionCache
    from victor.workflows.definition import WorkflowDefinition
    from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

from victor.workflows.runtime_executor_factory import (
    create_legacy_streaming_workflow_executor,
    create_legacy_workflow_executor,
)

logger = logging.getLogger(__name__)


class YAMLWorkflowCoordinator:
    """Coordinator for YAML workflow execution.

    Handles all aspects of YAML-defined workflow execution including:
    - Loading and parsing YAML files
    - Caching parsed definitions for performance
    - Executing workflows via WorkflowExecutor
    - Streaming execution events via StreamingWorkflowExecutor

    Example:
        coordinator = YAMLWorkflowCoordinator()

        # Execute a workflow
        result = await coordinator.execute(
            "path/to/workflow.yaml",
            initial_state={"input": "data"},
        )

        # Stream execution events
        async for event in coordinator.stream("workflow.yaml", state):
            print(f"{event.node_id}: {event.event_type}")
    """

    def __init__(
        self,
        executor: Optional["IWorkflowExecutor"] = None,
        streaming_executor: Optional["IStreamingExecutor"] = None,
        definition_cache: Optional["WorkflowDefinitionCache"] = None,
        unified_compiler: Optional["UnifiedWorkflowCompiler"] = None,
        use_unified_compiler: bool = True,
    ) -> None:
        """Initialize the YAML workflow coordinator.

        Args:
            executor: Optional WorkflowExecutor instance (legacy)
            streaming_executor: Optional StreamingWorkflowExecutor instance (legacy)
            definition_cache: Optional WorkflowDefinitionCache for caching
            unified_compiler: Optional UnifiedWorkflowCompiler instance
            use_unified_compiler: Whether to use unified compiler (default True)
        """
        self._executor = executor
        self._streaming_executor = streaming_executor
        self._definition_cache = definition_cache
        self._unified_compiler = unified_compiler
        self._use_unified_compiler = use_unified_compiler

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

    def _get_definition_cache(self) -> "WorkflowDefinitionCache":
        """Get or create definition cache."""
        if self._definition_cache is None:
            from victor.workflows.cache import get_workflow_definition_cache

            self._definition_cache = get_workflow_definition_cache()
        return self._definition_cache

    def _get_unified_compiler(self) -> "UnifiedWorkflowCompiler":
        """Get or create unified compiler with caching enabled.

        **Architecture Note**: UnifiedWorkflowCompiler is the canonical compiler API.
        It supports escape hatches (condition_registry, transform_registry),
        two-level caching, and multi-source compilation (YAML files, strings,
        WorkflowDefinition, WorkflowGraph). For DI-friendly compile-only usage,
        see WorkflowCompiler in victor.workflows.compiler.
        """
        if self._unified_compiler is None:
            from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

            self._unified_compiler = UnifiedWorkflowCompiler(enable_caching=True)
        return self._unified_compiler

    def _compute_config_hash(
        self,
        condition_registry: Optional[Dict[str, Callable[..., Any]]],
        transform_registry: Optional[Dict[str, Callable[..., Any]]],
    ) -> int:
        """Compute hash for cache key based on registries.

        Args:
            condition_registry: Condition functions
            transform_registry: Transform functions

        Returns:
            Hash value for cache key
        """
        # Hash based on function names in registries
        condition_names = tuple(sorted(condition_registry.keys())) if condition_registry else ()
        transform_names = tuple(sorted(transform_registry.keys())) if transform_registry else ()
        return hash((condition_names, transform_names))

    def load_workflow(
        self,
        yaml_path: Union[str, Path],
        workflow_name: Optional[str] = None,
        condition_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        transform_registry: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> "WorkflowDefinition":
        """Load a workflow definition from a YAML file.

        Uses the definition cache to avoid redundant parsing.

        Args:
            yaml_path: Path to YAML workflow file
            workflow_name: Specific workflow to load from file
            condition_registry: Custom condition functions
            transform_registry: Custom transform functions

        Returns:
            Parsed WorkflowDefinition

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the workflow name isn't found
        """
        from victor.workflows.yaml_loader import (
            load_workflow_from_file,
            YAMLWorkflowConfig,
        )

        path = Path(yaml_path)
        name = workflow_name or "default"
        config_hash = self._compute_config_hash(condition_registry, transform_registry)

        # Try cache first
        cache = self._get_definition_cache()
        cached_def = cache.get(path, name, config_hash)
        if cached_def is not None:
            logger.debug(f"Using cached workflow definition: {name} from {path}")
            return cached_def

        # Parse YAML
        config = YAMLWorkflowConfig(
            condition_registry=condition_registry or {},
            transform_registry=transform_registry or {},
        )

        result = load_workflow_from_file(
            str(yaml_path),
            workflow_name=workflow_name,
            config=config,
        )

        # Handle dict or single workflow result
        if isinstance(result, dict):
            if not result:
                raise ValueError(f"No workflows found in {yaml_path}")
            workflow_def = next(iter(result.values()))
        else:
            workflow_def = result

        # Cache the definition
        cache.put(path, name, config_hash, workflow_def)
        logger.debug(f"Cached workflow definition: {name} from {path}")

        return workflow_def

    async def execute(
        self,
        yaml_path: Union[str, Path],
        initial_state: Optional[Dict[str, Any]] = None,
        workflow_name: Optional[str] = None,
        condition_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        transform_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> "WorkflowExecutionResult":
        """Execute a YAML-defined workflow.

        Args:
            yaml_path: Path to YAML workflow file
            initial_state: Initial workflow state
            workflow_name: Specific workflow to load from file
            condition_registry: Custom condition functions
            transform_registry: Custom transform functions
            thread_id: Thread ID for checkpointing (resumable workflows)
            **kwargs: Additional execution parameters

        Returns:
            WorkflowExecutionResult with final state and metadata
        """
        from victor.framework.workflow_engine import WorkflowExecutionResult

        start_time = time.time()
        hitl_requests: List[Dict[str, Any]] = []

        try:
            if self._use_unified_compiler:
                # Use unified compiler for consistent caching and execution
                compiler = self._get_unified_compiler()
                compiled = compiler.compile_yaml(
                    Path(yaml_path),
                    workflow_name=workflow_name,
                    condition_registry=condition_registry,
                    transform_registry=transform_registry,
                )

                # Execute via CachedCompiledGraph.invoke()
                result = await compiled.invoke(
                    initial_state or {},
                    thread_id=thread_id,
                    **kwargs,
                )
                duration = time.time() - start_time

                # Handle polymorphic result types (LSP compliance)
                if hasattr(result, "state"):
                    final_state = (
                        result.state if isinstance(result.state, dict) else {"result": result.state}
                    )
                    nodes_executed = getattr(result, "node_history", [])
                    success = getattr(result, "success", True)
                    error = getattr(result, "error", None)
                elif isinstance(result, dict):
                    final_state = result
                    nodes_executed = (
                        result.pop("_nodes_executed", []) if "_nodes_executed" in result else []
                    )
                    success = True
                    error = None
                else:
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
                    hitl_requests=hitl_requests,
                )
            else:
                # Legacy path: Use old WorkflowExecutor
                from victor.workflows.context import WorkflowContext

                # Load workflow (uses cache)
                workflow_def = self.load_workflow(
                    yaml_path,
                    workflow_name=workflow_name,
                    condition_registry=condition_registry,
                    transform_registry=transform_registry,
                )

                # Create executor
                executor = self._get_executor()

                # Create context
                context = WorkflowContext(
                    workflow=workflow_def,
                    initial_state=initial_state or {},
                )

                # Execute
                result = await executor.execute(context)

                duration = time.time() - start_time

                return WorkflowExecutionResult(
                    success=result.success,
                    final_state=result.final_state,
                    nodes_executed=result.nodes_executed,
                    duration_seconds=duration,
                    error=result.error if not result.success else None,
                    hitl_requests=hitl_requests,
                )

        except Exception as e:
            logger.error(f"YAML workflow execution failed: {e}")
            return WorkflowExecutionResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def stream(
        self,
        yaml_path: Union[str, Path],
        initial_state: Optional[Dict[str, Any]] = None,
        workflow_name: Optional[str] = None,
        condition_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        transform_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator["WorkflowEvent"]:
        """Stream events from YAML workflow execution.

        Args:
            yaml_path: Path to YAML workflow file
            initial_state: Initial workflow state
            workflow_name: Specific workflow to load from file
            condition_registry: Custom condition functions
            transform_registry: Custom transform functions
            thread_id: Thread ID for checkpointing (resumable workflows)
            **kwargs: Additional execution parameters

        Yields:
            WorkflowEvent for each execution step
        """
        from victor.framework.workflow_engine import WorkflowEvent

        try:
            if self._use_unified_compiler:
                # Use unified compiler for consistent streaming
                compiler = self._get_unified_compiler()
                compiled = compiler.compile_yaml(
                    Path(yaml_path),
                    workflow_name=workflow_name,
                    condition_registry=condition_registry,
                    transform_registry=transform_registry,
                )

                # Stream via CachedCompiledGraph.stream()
                async for node_id, state in compiled.stream(
                    initial_state or {},
                    thread_id=thread_id,
                    **kwargs,
                ):
                    yield WorkflowEvent(
                        event_type="node_complete",
                        node_id=node_id,
                        timestamp=time.time(),
                        data={},
                        state_snapshot=state if isinstance(state, dict) else {},
                    )
            else:
                # Legacy path: Use old StreamingWorkflowExecutor
                from victor.workflows.streaming import WorkflowStreamContext

                # Load workflow (uses cache)
                workflow_def = self.load_workflow(
                    yaml_path,
                    workflow_name=workflow_name,
                    condition_registry=condition_registry,
                    transform_registry=transform_registry,
                )

                # Create streaming executor
                executor = self._get_streaming_executor()

                # Create stream context
                context = WorkflowStreamContext(
                    workflow=workflow_def,
                    initial_state=initial_state or {},
                )

                # Stream execution
                async for chunk in executor.stream(context):
                    yield WorkflowEvent(
                        event_type=(
                            chunk.event_type.value
                            if hasattr(chunk.event_type, "value")
                            else str(chunk.event_type)
                        ),
                        node_id=chunk.node_id or "",
                        timestamp=time.time(),
                        data={"content": chunk.content} if chunk.content else {},
                        state_snapshot=chunk.state_snapshot,
                    )

        except Exception as e:
            logger.error(f"Streaming YAML workflow failed: {e}")
            yield WorkflowEvent(
                event_type="error",
                node_id="",
                timestamp=time.time(),
                data={"error": str(e)},
            )


__all__ = ["YAMLWorkflowCoordinator"]
