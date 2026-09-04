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

"""Canonical entry point for all workflow compilation.

This is the single module callers should import for any workflow compilation
path. ``graph_compiler.py`` and ``yaml_to_graph_compiler.py`` are internal
implementation modules — not alternatives. This module consolidates:

1. WorkflowGraphCompiler (graph_dsl -> CompiledGraph)
2. YAMLToStateGraphCompiler (YAML -> StateGraph -> CompiledGraph)
3. WorkflowDefinitionCompiler (WorkflowDefinition -> CompiledGraph)

.. note::
    The DI-based ``WorkflowCompiler`` in ``victor.workflows.compiler.unified_compiler``
    is deprecated. All code should use this ``UnifiedWorkflowCompiler`` directly.
    The DI facade remains only for backward compatibility and will be removed in 0.10.0.

Key Features:
- Single entry point for all workflow types
- Integrated caching (definition + execution)
- DRY node execution via NodeExecutorFactory
- True parallel execution via asyncio.gather
- Observability event emission

Example:
    from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
    from pathlib import Path

    # Create compiler with caching
    compiler = UnifiedWorkflowCompiler(enable_caching=True, cache_ttl=3600)

    # Compile from YAML file
    cached_graph = compiler.compile_yaml(Path("workflow.yaml"), "my_workflow")

    # Execute with automatic cache integration
    result = await cached_graph.invoke({"input": "data"})

    # Check cache stats
    print(compiler.get_cache_stats())
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from victor.workflows.executors.compatibility import CompatibilityNodeExecutorFactory
from victor.workflows.runtime_types import (
    GraphNodeResult as NodeExecutionResult,
    create_initial_workflow_state,
)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.framework.graph import CompiledGraph, GraphExecutionResult, GraphConfig
    from victor.tools.registry import ToolRegistry
    from victor.workflows.cache import (
        WorkflowDefinitionCache,
        WorkflowCacheManager,
    )
    from victor.workflows.definition import (
        AgentNode,
        ComputeNode,
        ConditionNode,
        ParallelNode,
        TeamStepWorkflow,
        TransformNode,
        WorkflowDefinition,
        WorkflowNode,
    )
    from victor.workflows.graph_dsl import WorkflowGraph
    from victor.workflows.node_runners import NodeRunnerRegistry
    from victor.workflows.compiler.boundary import LegacyWorkflowDslCompiler
    from victor.workflows.yaml_loader import YAMLWorkflowConfig

logger = logging.getLogger(__name__)

StateType = TypeVar("StateType", bound=Dict[str, Any])


# =============================================================================
# Compiler Configuration
# =============================================================================


@dataclass
class UnifiedCompilerConfig:
    """Configuration for the UnifiedWorkflowCompiler.

    Attributes:
        enable_caching: Whether to enable caching (default: True)
        cache_ttl: Cache time-to-live in seconds (default: 3600)
        max_cache_entries: Maximum cache entries (default: 500)
        validate_before_compile: Whether to validate workflows before compilation
        enable_observability: Whether to emit observability events
        use_node_runners: Whether to use NodeRunner protocol for execution
        preserve_state_type: Whether to preserve typed state or use dict
        max_iterations: Maximum workflow iterations (default: 25)
        execution_timeout: Overall execution timeout in seconds
        enable_checkpointing: Whether to enable state checkpointing
    """

    enable_caching: bool = True
    cache_ttl: int = 3600
    max_cache_entries: int = 500
    validate_before_compile: bool = True
    enable_observability: bool = False
    use_node_runners: bool = False
    preserve_state_type: bool = False
    max_iterations: int = 25
    execution_timeout: Optional[float] = None
    enable_checkpointing: bool = True


# `NodeExecutionResult` remains exported here as a deprecated compatibility alias.


# =============================================================================
# Shared Factory Compatibility
# =============================================================================


class NodeExecutorFactory(CompatibilityNodeExecutorFactory):
    """Compatibility shim over the canonical workflow node executor factory."""

    def __init__(
        self,
        orchestrator: Optional["AgentOrchestrator"] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        runner_registry: Optional["NodeRunnerRegistry"] = None,
        emitter: Optional[Any] = None,
    ):
        super().__init__(
            orchestrator=orchestrator,
            tool_registry=tool_registry,
        )
        self._runner_registry = runner_registry
        self._emitter = emitter
        self._mutable_state_keys = frozenset(
            {"_parallel_results", "_node_results", "_errors", "_checkpoints"}
        )

    def _copy_state_for_parallel(self, state: Dict[str, Any]) -> Dict[str, Any]:
        child_state = dict(state)
        for key in self._mutable_state_keys:
            if key in child_state:
                child_state[key] = copy.deepcopy(child_state[key])
        return child_state

    def register_executor_type(
        self,
        node_type: str,
        executor_class: Any,
        *,
        replace: bool = False,
    ) -> None:
        self._delegate.register_executor_type(node_type, executor_class, replace=replace)

    def create_executor(
        self,
        node: "WorkflowNode",
    ) -> Callable[[Dict[str, Any]], Any]:
        original_resolver = self._delegate._resolve_execution_context
        self._delegate._resolve_execution_context = self._resolve_execution_context
        try:
            return self._delegate.create_executor(node)
        finally:
            self._delegate._resolve_execution_context = original_resolver

    def create_agent_executor(self, node: "AgentNode") -> Callable[[Dict[str, Any]], Any]:
        return self.create_executor(node)

    def create_compute_executor(self, node: "ComputeNode") -> Callable[[Dict[str, Any]], Any]:
        return self.create_executor(node)

    def create_condition_router(self, node: "ConditionNode") -> Callable[[Dict[str, Any]], Any]:
        return self.create_executor(node)

    def create_parallel_executor(
        self,
        node: "ParallelNode",
        child_nodes: Optional[List["WorkflowNode"]] = None,
    ) -> Callable[[Dict[str, Any]], Any]:
        return self.create_executor(node)

    def create_transform_executor(self, node: "TransformNode") -> Callable[[Dict[str, Any]], Any]:
        return self.create_executor(node)

    def create_team_executor(self, node: "TeamStepWorkflow") -> Callable[[Dict[str, Any]], Any]:
        return self.create_executor(node)

    def supports_node_type(self, node_type: str) -> bool:
        return self._delegate.supports_node_type(node_type)


# =============================================================================
# Cached Compiled Graph Wrapper
# =============================================================================


@dataclass
class CachedCompiledGraph:
    """CompiledGraph wrapper with cache integration.

    Provides automatic cache lookup and storage for execution results,
    enabling efficient repeated workflow execution.

    Attributes:
        compiled_graph: The underlying CompiledGraph
        workflow_name: Name of the workflow
        source_path: Path to the source YAML file (if applicable)
        compiled_at: Timestamp when compilation occurred
        source_mtime: Modification time of source file (for invalidation)
        cache_key: Unique key for cache lookups
        max_execution_timeout_seconds: Workflow-level timeout
        default_node_timeout_seconds: Default timeout for nodes
        max_iterations: Maximum workflow iterations
        max_retries: Maximum retries for workflow
    """

    compiled_graph: "CompiledGraph"
    workflow_name: str
    source_path: Optional[Path] = None
    compiled_at: float = field(default_factory=time.time)
    source_mtime: Optional[float] = None
    cache_key: str = ""
    max_execution_timeout_seconds: Optional[float] = None
    default_node_timeout_seconds: Optional[float] = None
    max_iterations: int = 25
    max_retries: int = 0

    async def invoke(
        self,
        input_state: Dict[str, Any],
        *,
        config: Optional["GraphConfig"] = None,
        thread_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> "GraphExecutionResult":
        """Execute the compiled workflow.

        Args:
            input_state: Initial state for execution
            config: Optional execution configuration override
            thread_id: Thread ID for checkpointing
            use_cache: Whether to use execution cache (for future use)

        Returns:
            GraphExecutionResult with final state
        """
        from victor.framework.graph import GraphExecutionResult

        # Prepare state with metadata
        exec_state = self._prepare_state(input_state)
        exec_state["_max_iterations"] = self.max_iterations

        # Execute with optional workflow-level timeout
        if self.max_execution_timeout_seconds:
            try:
                result = await asyncio.wait_for(
                    self.compiled_graph.invoke(
                        exec_state,
                        config=config,
                        thread_id=thread_id,
                    ),
                    timeout=self.max_execution_timeout_seconds,
                )
                return result
            except asyncio.TimeoutError:
                logger.warning(
                    f"Workflow '{self.workflow_name}' timed out after "
                    f"{self.max_execution_timeout_seconds}s"
                )
                return GraphExecutionResult(
                    state=exec_state,
                    success=False,
                    error=f"Workflow execution timed out after {self.max_execution_timeout_seconds}s",
                    iterations=0,
                    duration=self.max_execution_timeout_seconds,
                    node_history=[],
                )
        else:
            return await self.compiled_graph.invoke(
                exec_state,
                config=config,
                thread_id=thread_id,
            )

    async def stream(
        self,
        input_state: Dict[str, Any],
        *,
        config: Optional["GraphConfig"] = None,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
        """Stream workflow execution yielding state after each node.

        Args:
            input_state: Initial state for execution
            config: Optional execution configuration override
            thread_id: Thread ID for checkpointing

        Yields:
            Tuple of (node_id, state) after each node execution
        """
        exec_state = self._prepare_state(input_state)

        async for node_id, state in self.compiled_graph.stream(
            exec_state,
            config=config,
            thread_id=thread_id,
        ):
            yield node_id, state

    def _prepare_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare initial state with workflow metadata."""
        exec_state = create_initial_workflow_state(
            workflow_id=state.get("_workflow_id"),
            workflow_name=state.get("_workflow_name", self.workflow_name),
            current_node=state.get("_current_node", ""),
            initial_state=state,
        )
        return exec_state

    def get_graph_schema(self) -> Dict[str, Any]:
        """Get graph structure as dictionary.

        Returns:
            Dictionary describing nodes and edges
        """
        return self.compiled_graph.get_graph_schema()

    @property
    def age_seconds(self) -> float:
        """Get age of this cached compilation in seconds."""
        return time.time() - self.compiled_at


# =============================================================================
# Unified Workflow Compiler
# =============================================================================


class UnifiedWorkflowCompiler:
    """Canonical compiler for all workflow types with integrated caching.

    This is the primary workflow compiler used by the framework layer
    (WorkflowEngine, YAMLCoordinator, BaseYAMLProvider, WorkflowScheduler).

    Consolidates compilation paths for:
    - YAML workflow files
    - YAML workflow content strings
    - WorkflowDefinition objects
    - WorkflowGraph objects

    All paths produce CachedCompiledGraph instances that can be executed
    through the unified CompiledGraph.invoke() engine.

    For DI-friendly compile-only usage without caching or execution,
    see ``WorkflowCompiler`` in ``victor.workflows.compiler``.

    Example:
        # Create compiler with caching
        compiler = UnifiedWorkflowCompiler(enable_caching=True)

        # Compile from YAML
        graph = compiler.compile_yaml(Path("workflow.yaml"), "my_workflow")

        # Compile from definition
        graph = compiler.compile_definition(my_definition)

        # Execute
        result = await graph.invoke({"input": "data"})

        # Check cache
        print(compiler.get_cache_stats())
    """

    def __init__(
        self,
        definition_cache: Optional["WorkflowDefinitionCache"] = None,
        execution_cache: Optional["WorkflowCacheManager"] = None,
        orchestrator: Optional["AgentOrchestrator"] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        runner_registry: Optional["NodeRunnerRegistry"] = None,
        emitter: Optional[Any] = None,  # ObservabilityEmitter
        enable_caching: bool = True,
        cache_ttl: int = 3600,
        config: Optional[UnifiedCompilerConfig] = None,
    ) -> None:
        """Initialize the unified compiler.

        Args:
            definition_cache: Cache for parsed YAML definitions
            execution_cache: Cache for execution results
            orchestrator: Agent orchestrator for agent nodes
            tool_registry: Tool registry for compute nodes
            runner_registry: NodeRunner registry for unified execution
            emitter: ObservabilityEmitter for streaming events
            enable_caching: Whether to enable caching (default: True)
            cache_ttl: Cache TTL in seconds (default: 3600)
            config: Full compiler configuration (overrides other params)
        """
        # Use config if provided, otherwise build from params
        if config:
            self._config = config
        else:
            self._config = UnifiedCompilerConfig(
                enable_caching=enable_caching,
                cache_ttl=cache_ttl,
            )

        # Store caches
        self._definition_cache = definition_cache
        self._execution_cache = execution_cache

        # Store dependencies
        self._orchestrator = orchestrator
        self._tool_registry = tool_registry or self._get_default_tool_registry()
        self._runner_registry = runner_registry
        self._emitter = emitter

        # Create node executor factory
        self._executor_factory = NodeExecutorFactory(
            orchestrator=self._orchestrator,
            tool_registry=self._tool_registry,
            runner_registry=self._runner_registry,
            emitter=self._emitter,
        )

        # Lazy-loaded compilers
        self._graph_compiler: Optional["LegacyWorkflowDslCompiler"] = None
        self._definition_validator: Optional[Any] = None
        self._definition_graph_compiler: Optional[Any] = None

        # Compiled-graph cache: skips both parse AND compile/validate on a
        # hit (co-design review item 13). Keyed on the existing definition
        # cache key plus a content-hash of every $ref-expanded file, since
        # mtime alone is unreliable across checkouts (see
        # _generate_yaml_cache_key's docstring and the review's 72%
        # false-stale reindex finding). Bounded LRU AND TTL-bounded (shares
        # cache_ttl with the definition cache) — some staleness classes
        # (e.g. a condition/transform function redefined under an unchanged
        # registry name) have no content-based invalidation signal at all.
        self._graph_cache: "OrderedDict[str, Tuple[CompiledGraph, WorkflowDefinition, float]]" = (
            OrderedDict()
        )
        self._graph_cache_max_entries = 32

        # Compilation stats
        self._compile_stats = {
            "yaml_compiles": 0,
            "yaml_content_compiles": 0,
            "definition_compiles": 0,
            "graph_compiles": 0,
            "cache_hits": 0,
            "graph_cache_hits": 0,
        }

    def _get_default_tool_registry(self) -> Optional["ToolRegistry"]:
        """Get the default tool registry if available."""
        try:
            from victor.tools.registry import get_tool_registry

            return get_tool_registry()
        except Exception:
            return None

    def _get_definition_cache(self) -> "WorkflowDefinitionCache":
        """Get or create definition cache."""
        if self._definition_cache is None:
            from victor.workflows.cache import get_workflow_definition_cache

            self._definition_cache = get_workflow_definition_cache()
        return self._definition_cache

    def _get_execution_cache(self) -> "WorkflowCacheManager":
        """Get or create execution cache."""
        if self._execution_cache is None:
            from victor.workflows.cache import get_workflow_cache_manager

            self._execution_cache = get_workflow_cache_manager()
        return self._execution_cache

    def _get_graph_compiler(self) -> "LegacyWorkflowDslCompiler":
        """Get or create the canonical WorkflowGraph DSL adapter."""
        if self._graph_compiler is None:
            from victor.workflows.compiler.boundary import LegacyWorkflowDslCompiler

            self._graph_compiler = LegacyWorkflowDslCompiler(
                runner_registry=self._runner_registry,
                validate_before_compile=self._config.validate_before_compile,
                preserve_state_type=self._config.preserve_state_type,
                emitter=self._emitter if self._config.enable_observability else None,
                enable_observability=self._config.enable_observability,
            )
        return self._graph_compiler

    def _get_definition_validator(self) -> Any:
        """Get or create the shared definition validator stage."""
        if self._definition_validator is None:
            from victor.workflows.compiler.boundary import WorkflowDefinitionValidator

            self._definition_validator = WorkflowDefinitionValidator()
        return self._definition_validator

    def _get_definition_graph_compiler(self) -> Any:
        """Get or create the shared native definition compiler backend."""
        if self._definition_graph_compiler is None:
            from victor.workflows.compiler.boundary import NativeWorkflowGraphCompiler

            self._definition_graph_compiler = NativeWorkflowGraphCompiler(
                node_executor_factory=self._executor_factory,
                enable_checkpointing=self._config.enable_checkpointing,
            )
        return self._definition_graph_compiler

    def _compile_definition_graph(
        self,
        definition: "WorkflowDefinition",
        *,
        source: str,
        workflow_name: Optional[str] = None,
        source_path: Optional[Path] = None,
        validate: Optional[bool] = None,
    ) -> "CompiledGraph":
        """Compile a workflow definition through the shared boundary backend."""
        from victor.workflows.compiler.boundary import (
            ParsedWorkflowDefinition,
            WorkflowCompilationRequest,
        )

        should_validate = self._config.validate_before_compile if validate is None else validate
        parsed = ParsedWorkflowDefinition(
            request=WorkflowCompilationRequest(
                source=source,
                workflow_name=workflow_name or definition.name,
                validate=should_validate,
            ),
            workflow=definition,
            source_path=source_path,
        )

        if should_validate:
            parsed = self._get_definition_validator().validate(parsed)

        return self._get_definition_graph_compiler().compile(parsed)

    def _compute_config_hash(
        self,
        condition_registry: Optional[Dict[str, Callable[..., Any]]],
        transform_registry: Optional[Dict[str, Callable[..., Any]]],
    ) -> int:
        """Compute hash for cache key based on registries."""
        condition_names = tuple(sorted(condition_registry.keys())) if condition_registry else ()
        transform_names = tuple(sorted(transform_registry.keys())) if transform_registry else ()
        return hash((condition_names, transform_names))

    def _create_yaml_loader(
        self,
        *,
        condition_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        transform_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        base_dir: Optional[Path] = None,
    ) -> Any:
        """Build a YAML loader for the deprecated unified compiler facade."""
        from victor.workflows.yaml_loader import YAMLWorkflowConfig, YAMLWorkflowLoader

        config = YAMLWorkflowConfig(
            condition_registry=condition_registry or {},
            transform_registry=transform_registry or {},
            base_dir=base_dir,
        )
        return YAMLWorkflowLoader(
            enable_cache=self._config.enable_caching,
            cache_ttl=self._config.cache_ttl,
            config=config,
        )

    def _create_workflow_parser(
        self,
        *,
        workflow_name: Optional[str] = None,
        condition_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        transform_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        base_dir: Optional[Path] = None,
    ) -> Any:
        """Build the shared parser stage for deprecated YAML compilation paths."""
        from victor.workflows.compiler.boundary import WorkflowParser

        loader = self._create_yaml_loader(
            condition_registry=condition_registry,
            transform_registry=transform_registry,
            base_dir=base_dir,
        )
        return WorkflowParser(loader)

    def _parse_workflow_definition(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        condition_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        transform_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        base_dir: Optional[Path] = None,
    ) -> Any:
        """Parse and normalize a workflow source through the shared boundary parser."""
        from victor.workflows.compiler.boundary import WorkflowCompilationRequest

        parser = self._create_workflow_parser(
            workflow_name=workflow_name,
            condition_registry=condition_registry,
            transform_registry=transform_registry,
            base_dir=base_dir,
        )
        return parser.parse(
            WorkflowCompilationRequest(
                source=source,
                workflow_name=workflow_name,
                validate=self._config.validate_before_compile,
            )
        )

    # =========================================================================
    # YAML Compilation
    # =========================================================================

    def compile_yaml(
        self,
        yaml_path: Union[str, Path],
        workflow_name: Optional[str] = None,
        condition_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        transform_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        **kwargs: Any,
    ) -> CachedCompiledGraph:
        """Compile a workflow from a YAML file.

        Uses the definition cache to avoid redundant parsing, then compiles
        the WorkflowDefinition to a CompiledGraph for execution.

        Args:
            yaml_path: Path to the YAML file
            workflow_name: Specific workflow to compile (if file has multiple)
            condition_registry: Custom condition functions (escape hatches)
            transform_registry: Custom transform functions (escape hatches)
            **kwargs: Additional compilation options

        Returns:
            CachedCompiledGraph ready for execution

        Raises:
            FileNotFoundError: If YAML file not found
            ValueError: If workflow validation fails
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")

        name = workflow_name or "default"
        config_hash = self._compute_config_hash(condition_registry, transform_registry)

        # Generate cache key
        cache_key = self._generate_yaml_cache_key(path, workflow_name)

        # Get source file mtime for cache validation
        try:
            source_mtime = path.stat().st_mtime
        except (OSError, FileNotFoundError):
            source_mtime = None

        graph_cache_key: Optional[str] = None
        has_refs = False
        if self._config.enable_caching:
            ref_files_hash = self._hash_ref_files(path)
            has_refs = bool(ref_files_hash)
            graph_cache_key = f"yaml:{cache_key}:{config_hash}:{ref_files_hash}"
            cached_graph = self._get_cached_graph(graph_cache_key)
            if cached_graph is not None:
                compiled, workflow_def = cached_graph
                logger.debug(f"Compiled-graph cache hit for {path}:{workflow_name}")
                return CachedCompiledGraph(
                    compiled_graph=compiled,
                    workflow_name=name,
                    source_path=path,
                    source_mtime=source_mtime,
                    cache_key=cache_key,
                    max_execution_timeout_seconds=workflow_def.max_execution_timeout_seconds,
                    default_node_timeout_seconds=workflow_def.default_node_timeout_seconds,
                    max_iterations=workflow_def.max_iterations,
                    max_retries=workflow_def.max_retries,
                )

        # Check definition cache. Skipped for $ref-using files: this cache's
        # key is mtime-only on the main file, with no visibility into
        # referenced-file content, so it would serve a stale parsed
        # definition after only a referenced file changes. The graph cache
        # above (whose key does hash referenced files) is the sole cache
        # layer for $ref-using workflows.
        if self._config.enable_caching and not has_refs:
            cache = self._get_definition_cache()
            cached_def = cache.get(path, name, config_hash)
            if cached_def is not None:
                self._compile_stats["cache_hits"] += 1
                logger.debug(f"Definition cache hit for {path}:{workflow_name}")
                compiled = self._compile_definition_graph(
                    cached_def,
                    source=str(path),
                    workflow_name=name,
                    source_path=path,
                )
                if graph_cache_key is not None:
                    self._store_cached_graph(graph_cache_key, compiled, cached_def)
                return CachedCompiledGraph(
                    compiled_graph=compiled,
                    workflow_name=name,
                    source_path=path,
                    source_mtime=source_mtime,
                    cache_key=cache_key,
                    max_execution_timeout_seconds=cached_def.max_execution_timeout_seconds,
                    default_node_timeout_seconds=cached_def.default_node_timeout_seconds,
                    max_iterations=cached_def.max_iterations,
                    max_retries=cached_def.max_retries,
                )

        parsed = self._parse_workflow_definition(
            str(path),
            workflow_name=workflow_name,
            condition_registry=condition_registry,
            transform_registry=transform_registry,
            base_dir=path.parent,
        )
        workflow_def = parsed.workflow

        # Cache the definition if enabled (skipped for $ref-using files —
        # see the read-side comment above).
        if self._config.enable_caching and not has_refs:
            cache = self._get_definition_cache()
            cache.put(path, name, config_hash, workflow_def)
            logger.debug(f"Cached workflow definition: {name} from {path}")

        self._compile_stats["yaml_compiles"] += 1

        # Compile to CompiledGraph
        compiled = self._compile_definition_graph(
            workflow_def,
            source=str(path),
            workflow_name=name,
            source_path=parsed.source_path or path,
        )
        if graph_cache_key is not None:
            self._store_cached_graph(graph_cache_key, compiled, workflow_def)
        return CachedCompiledGraph(
            compiled_graph=compiled,
            workflow_name=name,
            source_path=path,
            source_mtime=source_mtime,
            cache_key=cache_key,
            max_execution_timeout_seconds=workflow_def.max_execution_timeout_seconds,
            default_node_timeout_seconds=workflow_def.default_node_timeout_seconds,
            max_iterations=workflow_def.max_iterations,
            max_retries=workflow_def.max_retries,
        )

    def compile_yaml_content(
        self,
        yaml_content: str,
        workflow_name: str,
        condition_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        transform_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        **kwargs: Any,
    ) -> CachedCompiledGraph:
        """Compile a workflow from YAML content string.

        Args:
            yaml_content: YAML content as string
            workflow_name: Name of workflow to compile
            condition_registry: Custom condition functions (escape hatches)
            transform_registry: Custom transform functions (escape hatches)
            **kwargs: Additional compilation options

        Returns:
            CachedCompiledGraph ready for execution
        """
        # Generate cache key from content hash
        cache_key = self._generate_content_cache_key(yaml_content, workflow_name)
        parsed = self._parse_workflow_definition(
            yaml_content,
            workflow_name=workflow_name,
            condition_registry=condition_registry,
            transform_registry=transform_registry,
        )
        workflow_def = parsed.workflow

        self._compile_stats["yaml_content_compiles"] += 1

        # Compile to CompiledGraph
        compiled = self._compile_definition_graph(
            workflow_def,
            source=yaml_content,
            workflow_name=workflow_name,
            source_path=parsed.source_path,
        )
        return CachedCompiledGraph(
            compiled_graph=compiled,
            workflow_name=workflow_name,
            cache_key=cache_key,
            max_execution_timeout_seconds=workflow_def.max_execution_timeout_seconds,
            default_node_timeout_seconds=workflow_def.default_node_timeout_seconds,
            max_iterations=workflow_def.max_iterations,
            max_retries=workflow_def.max_retries,
        )

    # =========================================================================
    # Definition Compilation
    # =========================================================================

    def compile_definition(
        self,
        definition: "WorkflowDefinition",
        cache_key: Optional[str] = None,
        **kwargs: Any,
    ) -> CachedCompiledGraph:
        """Compile a WorkflowDefinition to a CachedCompiledGraph.

        Args:
            definition: The workflow definition to compile
            cache_key: Optional cache key (generated if not provided)
            **kwargs: Additional compilation options

        Returns:
            CachedCompiledGraph ready for execution

        Raises:
            ValueError: If workflow validation fails
        """
        # Generate cache key if not provided
        if not cache_key:
            cache_key = self._generate_definition_cache_key(definition)

        # The graph cache always keys off the definition's own content, never
        # a caller-supplied cache_key — an arbitrary caller key isn't
        # guaranteed to change when the definition does, which would let a
        # stale compiled graph leak across genuinely different definitions.
        # A definition that can't be fingerprinted (e.g. a custom node
        # subclass whose to_dict() isn't fully compatible) degrades to no
        # graph caching for this call rather than failing the compile.
        graph_cache_key: Optional[str] = None
        if self._config.enable_caching:
            try:
                content_key = self._generate_definition_cache_key(definition)
                # to_dict() intentionally drops behavior — TransformNode's
                # transform and ConditionNode's condition aren't JSON
                # serializable — so two structurally-identical definitions
                # that differ only in node callables would otherwise hash
                # identically and collide in the cache. id() is safe here:
                # this cache holds a strong reference to every cached
                # definition (keeping its callables alive) for as long as
                # the entry survives, so two simultaneously-reachable
                # callables can never share an id().
                callable_fingerprint = self._definition_callable_fingerprint(definition)
                graph_cache_key = f"definition:{content_key}:{callable_fingerprint}"
            except Exception:
                logger.debug(
                    "Definition not fingerprintable for graph cache key; "
                    "compiling without graph cache: %s",
                    definition.name,
                )

            cached_graph = (
                self._get_cached_graph(graph_cache_key) if graph_cache_key is not None else None
            )
            if cached_graph is not None:
                compiled, _cached_definition = cached_graph
                logger.debug(f"Compiled-graph cache hit for definition:{definition.name}")
                return CachedCompiledGraph(
                    compiled_graph=compiled,
                    workflow_name=definition.name,
                    cache_key=cache_key,
                    max_execution_timeout_seconds=definition.max_execution_timeout_seconds,
                    default_node_timeout_seconds=definition.default_node_timeout_seconds,
                    max_iterations=definition.max_iterations,
                    max_retries=definition.max_retries,
                )

        self._compile_stats["definition_compiles"] += 1

        # Compile to CompiledGraph
        compiled = self._compile_definition_graph(
            definition,
            source=f"definition://{definition.name}",
            workflow_name=definition.name,
        )
        if graph_cache_key is not None:
            self._store_cached_graph(graph_cache_key, compiled, definition)
        return CachedCompiledGraph(
            compiled_graph=compiled,
            workflow_name=definition.name,
            cache_key=cache_key,
            max_execution_timeout_seconds=definition.max_execution_timeout_seconds,
            default_node_timeout_seconds=definition.default_node_timeout_seconds,
            max_iterations=definition.max_iterations,
            max_retries=definition.max_retries,
        )

    # =========================================================================
    # WorkflowGraph Compilation
    # =========================================================================

    def compile_graph(
        self,
        graph: "WorkflowGraph",
        name: Optional[str] = None,
        cache_key: Optional[str] = None,
        **kwargs: Any,
    ) -> CachedCompiledGraph:
        """Compile a WorkflowGraph (graph_dsl) to CachedCompiledGraph.

        Args:
            graph: The WorkflowGraph to compile
            name: Optional name override
            cache_key: Optional cache key
            **kwargs: Additional compilation options

        Returns:
            CachedCompiledGraph ready for execution
        """
        # Generate cache key
        if not cache_key:
            cache_key = self._generate_graph_cache_key(graph, name)

        self._compile_stats["graph_compiles"] += 1
        workflow_name = name or getattr(graph, "name", "workflow_graph")

        # Compile to CompiledGraph
        compiled = self._get_graph_compiler().compile(graph, name)
        return CachedCompiledGraph(
            compiled_graph=compiled,
            workflow_name=workflow_name,
            cache_key=cache_key,
        )

    # =========================================================================
    # Cache Management
    # =========================================================================

    def clear_cache(self) -> int:
        """Clear all caches.

        Returns:
            Total number of entries cleared
        """
        total = 0

        # Clear the compiled-graph cache
        total += len(self._graph_cache)
        self._graph_cache.clear()

        # Clear definition cache
        if self._definition_cache is not None:
            total += self._definition_cache.clear()
        else:
            from victor.workflows.cache import get_workflow_definition_cache

            total += get_workflow_definition_cache().clear()

        # Clear execution cache
        if self._execution_cache is not None:
            total += self._execution_cache.clear_all()
        else:
            from victor.workflows.cache import get_workflow_cache_manager

            total += get_workflow_cache_manager().clear_all()

        logger.info(f"Cleared {total} cache entries")
        return total

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary with cache and compilation statistics
        """
        stats: Dict[str, Any] = {
            "compilation": dict(self._compile_stats),
            "caching_enabled": self._config.enable_caching,
            "graph_cache": {
                "size": len(self._graph_cache),
                "max_entries": self._graph_cache_max_entries,
            },
        }

        # Get definition cache stats
        if self._config.enable_caching:
            if self._definition_cache is not None:
                stats["definition_cache"] = self._definition_cache.get_stats()
            else:
                from victor.workflows.cache import get_workflow_definition_cache

                stats["definition_cache"] = get_workflow_definition_cache().get_stats()

            # Get execution cache stats
            if self._execution_cache is not None:
                stats["execution_cache"] = self._execution_cache.get_all_stats()
            else:
                from victor.workflows.cache import get_workflow_cache_manager

                stats["execution_cache"] = get_workflow_cache_manager().get_all_stats()
        else:
            stats["definition_cache"] = {"enabled": False}
            stats["execution_cache"] = {}

        return stats

    def invalidate_yaml(self, yaml_path: Union[str, Path]) -> int:
        """Invalidate cached definitions for a specific YAML file.

        Args:
            yaml_path: Path to YAML file to invalidate

        Returns:
            Number of cache entries invalidated
        """
        if not self._config.enable_caching:
            return 0

        path = Path(yaml_path)
        cache = self._get_definition_cache()
        count = cache.invalidate(path)
        if count > 0:
            logger.info(f"Invalidated {count} cache entries for: {path}")
        return count

    def set_runner_registry(self, registry: "NodeRunnerRegistry") -> None:
        """Set the NodeRunner registry for execution.

        Args:
            registry: NodeRunnerRegistry with configured runners
        """
        self._runner_registry = registry
        # Reset compilers to use new registry
        self._graph_compiler = None
        self._definition_graph_compiler = None
        # Update executor factory
        self._executor_factory._runner_registry = registry

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _generate_yaml_cache_key(
        self,
        yaml_path: Path,
        workflow_name: Optional[str],
    ) -> str:
        """Generate cache key for YAML file compilation.

        .. note::
            The cache key currently includes only the main YAML file's
            mtime.  Workflows that use ``$ref`` to include external node
            definitions (resolved in ``yaml_loader._expand_refs``) will
            **not** be invalidated when those referenced files change.
            Incorporating ``$ref`` mtimes would require pre-parsing the
            YAML before generating the cache key, creating a circular
            dependency with the compilation cache.

        TODO: To support ``$ref`` cache invalidation, consider a
        two-phase approach: (1) quick-scan the raw YAML text for
        ``$ref:`` patterns, resolve file paths, and collect their
        mtimes before hashing; or (2) store the set of referenced file
        paths after first compilation and verify their mtimes on
        subsequent cache lookups.
        """
        try:
            mtime = yaml_path.stat().st_mtime
        except OSError:
            mtime = 0

        key_data = f"{yaml_path.resolve()}:{workflow_name or ''}:{mtime}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _generate_content_cache_key(
        self,
        content: str,
        workflow_name: str,
    ) -> str:
        """Generate cache key for YAML content compilation."""
        key_data = f"content:{workflow_name}:{content}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _generate_definition_cache_key(
        self,
        definition: "WorkflowDefinition",
    ) -> str:
        """Generate cache key for definition compilation."""
        # Use definition's serialized form for key
        key_data = json.dumps(definition.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _definition_callable_fingerprint(self, definition: "WorkflowDefinition") -> str:
        """Identity fingerprint for node callables the content hash can't see.

        Covers any node attribute that's callable (TransformNode.transform,
        ConditionNode.condition today) without hardcoding field names, so a
        future node type with its own callable field is covered too.
        """
        parts = []
        for node_id in sorted(definition.nodes.keys()):
            node = definition.nodes[node_id]
            for attr_name, value in sorted(vars(node).items()):
                if callable(value):
                    parts.append(f"{node_id}.{attr_name}:{id(value)}")
        return ":".join(parts)

    def _generate_graph_cache_key(
        self,
        graph: "WorkflowGraph",
        name: Optional[str],
    ) -> str:
        """Generate cache key for WorkflowGraph compilation."""
        # Use node and edge structure for key
        key_parts = [
            name or "",
            str(sorted(graph._nodes.keys())),
            str(sorted(graph._edges.keys())),
        ]
        key_data = ":".join(key_parts)
        return hashlib.sha256(key_data.encode()).hexdigest()

    _REF_PATTERN = re.compile(r"\$ref:\s*['\"]?([^'\"\s#]+)")

    def _hash_ref_files(self, yaml_path: Path) -> str:
        """Content-hash every ``$ref``-expanded file a YAML workflow points at.

        Quick-scans the raw text for ``$ref:`` targets (single level — the
        loader's ``_resolve_ref`` doesn't itself recursively expand refs
        inside a referenced file, so one scan of the root file is enough)
        instead of parsing, so this stays cheap enough to run on every
        compiled-graph cache lookup. Content, not mtime: mtime is unreliable
        across checkouts/clones (co-design review's 72% false-stale finding).
        """
        try:
            text = yaml_path.read_text()
        except OSError:
            return ""

        refs = sorted(set(self._REF_PATTERN.findall(text)))
        if not refs:
            # No $ref present: an empty string, distinct from any real
            # digest (including hashlib.sha256().hexdigest() on zero
            # updates), so callers can cheaply test "has refs at all".
            return ""

        hasher = hashlib.sha256()
        for ref in refs:
            candidate = Path(ref)
            if not candidate.is_absolute():
                candidate = yaml_path.parent / candidate
            try:
                resolved = candidate.resolve()
                content = resolved.read_bytes()
            except OSError:
                continue
            hasher.update(str(resolved).encode())
            hasher.update(content)
        return hasher.hexdigest()

    def _get_cached_graph(self, key: str) -> Optional[Tuple["CompiledGraph", "WorkflowDefinition"]]:
        """Bounded-LRU, TTL-bounded lookup for a previously compiled graph.

        The TTL matches ``cache_ttl`` (same knob the definition cache uses)
        so a stale entry self-heals within the same bound callers already
        expect, rather than staying valid indefinitely once warm — some
        staleness classes (e.g. a condition/transform registered under an
        unchanged name but different behavior) have no other invalidation
        signal at all.
        """
        entry = self._graph_cache.get(key)
        if entry is None:
            return None
        compiled, definition, stored_at = entry
        if time.time() - stored_at > self._config.cache_ttl:
            del self._graph_cache[key]
            return None
        self._graph_cache.move_to_end(key)
        self._compile_stats["graph_cache_hits"] += 1
        return compiled, definition

    def _store_cached_graph(
        self,
        key: str,
        compiled: "CompiledGraph",
        definition: "WorkflowDefinition",
    ) -> None:
        """Store a compiled graph, evicting the least-recently-used entry."""
        self._graph_cache[key] = (compiled, definition, time.time())
        self._graph_cache.move_to_end(key)
        while len(self._graph_cache) > self._graph_cache_max_entries:
            self._graph_cache.popitem(last=False)


# =============================================================================
# Convenience Functions
# =============================================================================


def compile_workflow(
    source: Union[Path, str, "WorkflowDefinition"],
    workflow_name: Optional[str] = None,
    enable_caching: bool = True,
    **kwargs: Any,
) -> CachedCompiledGraph:
    """Compile a workflow from various sources.

    Convenience function for one-off compilation.

    Args:
        source: YAML path, YAML content string, or WorkflowDefinition
        workflow_name: Name of workflow (required for YAML content)
        enable_caching: Whether to enable caching
        **kwargs: Additional compilation options

    Returns:
        CachedCompiledGraph ready for execution
    """
    compiler = UnifiedWorkflowCompiler(enable_caching=enable_caching)

    if isinstance(source, Path):
        return compiler.compile_yaml(source, workflow_name, **kwargs)
    elif isinstance(source, str):
        # Check if it's a file path or YAML content
        if Path(source).exists():
            return compiler.compile_yaml(Path(source), workflow_name, **kwargs)
        else:
            if not workflow_name:
                raise ValueError("workflow_name required for YAML content")
            return compiler.compile_yaml_content(source, workflow_name, **kwargs)
    else:
        # Assume WorkflowDefinition
        return compiler.compile_definition(source, **kwargs)


async def compile_and_execute(
    source: Union[Path, str, "WorkflowDefinition"],
    initial_state: Optional[Dict[str, Any]] = None,
    workflow_name: Optional[str] = None,
    **kwargs: Any,
) -> "GraphExecutionResult":
    """Compile and execute a workflow in one step.

    Convenience function for one-off execution.

    Args:
        source: YAML path, YAML content string, or WorkflowDefinition
        initial_state: Initial workflow state
        workflow_name: Name of workflow (required for YAML content)
        **kwargs: Additional compilation/execution options

    Returns:
        GraphExecutionResult with final state
    """
    graph = compile_workflow(source, workflow_name, **kwargs)
    return await graph.invoke(initial_state or {})


def create_unified_compiler(
    enable_caching: bool = True,
    runner_registry: Optional["NodeRunnerRegistry"] = None,
    **kwargs: Any,
) -> UnifiedWorkflowCompiler:
    """Create a UnifiedWorkflowCompiler with default caches.

    Args:
        enable_caching: Whether to enable caching
        runner_registry: Optional NodeRunner registry
        **kwargs: Additional configuration options

    Returns:
        Configured UnifiedWorkflowCompiler instance
    """
    from victor.workflows.cache import (
        get_workflow_definition_cache,
        get_workflow_cache_manager,
    )

    return UnifiedWorkflowCompiler(
        definition_cache=get_workflow_definition_cache(),
        execution_cache=get_workflow_cache_manager(),
        runner_registry=runner_registry,
        enable_caching=enable_caching,
        **kwargs,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "UnifiedCompilerConfig",
    # Execution result
    "NodeExecutionResult",
    # Factory
    "NodeExecutorFactory",
    # Cached graph
    "CachedCompiledGraph",
    # Main compiler
    "UnifiedWorkflowCompiler",
    # Convenience functions
    "compile_workflow",
    "compile_and_execute",
    "create_unified_compiler",
]
