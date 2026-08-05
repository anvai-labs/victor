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

"""Base YAML workflow provider for vertical-specific workflow implementations.

This module provides a Template Method + Strategy pattern base class that eliminates
workflow provider duplication across verticals. Subclasses only need to specify
the escape hatches module path and optionally the workflows directory.

Design Patterns:
    - Template Method: Common algorithm in base class, subclass customization via hooks
    - Strategy: Escape hatches module loaded dynamically based on subclass specification
    - Lazy Loading: Workflows loaded on first access for performance

Example:
    class ResearchWorkflowProvider(BaseYAMLWorkflowProvider):
        '''Provides research-specific workflows.'''

        def _get_escape_hatches_module(self) -> str:
            return "victor.research.escape_hatches"

    # Usage
    provider = ResearchWorkflowProvider()
    async for node_id, state in provider.stream_compiled_workflow("deep_research", {}):
        print(f"Completed: {node_id}")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from victor.core.verticals.import_resolver import (
    import_module_with_fallback,
    module_import_candidates,
)
from victor.core.verticals.protocols import WorkflowProviderProtocol
from victor.workflows.definition import WorkflowDefinition

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.framework.graph import GraphExecutionResult
    from victor.workflows.unified_compiler import (
        CachedCompiledGraph,
        UnifiedWorkflowCompiler,
    )
    from victor.workflows.yaml_loader import YAMLWorkflowConfig


class BaseYAMLWorkflowProvider(WorkflowProviderProtocol, ABC):
    """Base class for YAML-based workflow providers.

    This abstract base class implements the common workflow provider pattern
    used across verticals (Research, DevOps, DataAnalysis, etc.), eliminating
    ~200 lines of duplicated boilerplate per vertical.

    Subclasses only need to implement:
        - _get_escape_hatches_module(): Return module path for CONDITIONS/TRANSFORMS

    Optionally override:
        - _get_workflows_directory(): Return Path to YAML workflow files
        - AUTO_WORKFLOW_PATTERNS: Class attribute for auto-trigger patterns
        - TASK_TYPE_MAPPINGS: Class attribute for task type mappings
        - get_auto_workflows(): Return automatic workflow triggers
        - get_workflow_for_task_type(): Map task types to workflow names

    Features:
        - Lazy loading of YAML workflows with caching
        - Automatic escape hatches registration from vertical-specific modules
        - Standard and streaming workflow execution
        - Error handling with graceful degradation
        - Configurable auto-workflow triggers via class attributes

    Example:
        class DevOpsWorkflowProvider(BaseYAMLWorkflowProvider):
            '''Provides DevOps-specific workflows.'''

            def _get_escape_hatches_module(self) -> str:
                return "victor.devops.escape_hatches"

            # Option 1: Use class attributes (new pattern)
            AUTO_WORKFLOW_PATTERNS = [
                (r"deploy\\s+infrastructure", "deploy_infrastructure"),
                (r"container(ize)?", "container_setup"),
            ]

            # Option 2: Override method (old pattern, still supported)
            def get_auto_workflows(self) -> List[Tuple[str, str]]:
                return [
                    (r"deploy\\s+infrastructure", "deploy_infrastructure"),
                    (r"container(ize)?", "container_setup"),
                ]

        provider = DevOpsWorkflowProvider()
        workflows = provider.get_workflows()  # Lazy-loads YAML files
    """

    # Class attributes for auto-workflow configuration
    # Subclasses can override these to provide auto-trigger patterns
    AUTO_WORKFLOW_PATTERNS: List[Tuple[str, str]] = []
    """List of (regex_pattern, workflow_name) tuples for auto-triggering workflows.

    Example:
        AUTO_WORKFLOW_PATTERNS = [
            (r"deep\\s+research", "deep_research"),
            (r"fact\\s*check", "fact_check"),
        ]
    """

    TASK_TYPE_MAPPINGS: Dict[str, str] = {}
    """Dict mapping task types to workflow names.

    Example:
        TASK_TYPE_MAPPINGS = {
            "research": "deep_research",
            "fact_check": "fact_check",
        }
    """

    def __init__(self) -> None:
        """Initialize the workflow provider with lazy loading support."""
        self._workflows: Optional[Dict[str, WorkflowDefinition]] = None
        self._config: Optional["YAMLWorkflowConfig"] = None
        self._compiler: Optional["UnifiedWorkflowCompiler"] = None

    @abstractmethod
    def _get_escape_hatches_module(self) -> str:
        """Return the fully qualified module path for escape hatches.

        This method must be implemented by subclasses to specify where
        the CONDITIONS and TRANSFORMS dictionaries are defined.

        Returns:
            Fully qualified module path string, e.g., "victor.research.escape_hatches"

        Example:
            def _get_escape_hatches_module(self) -> str:
                return "victor.research.escape_hatches"
        """
        ...

    def _get_capability_provider_module(self) -> Optional[str]:
        """Return the module path for the capability provider.

        This optional method allows subclasses to specify their capability
        provider module for automatic registration. This enables workflows
        to access vertical-specific capabilities without manual configuration.

        Returns:
            Fully qualified module path string or None if not applicable

        Example:
            def _get_capability_provider_module(self) -> Optional[str]:
                return "victor.research.capabilities"

        Note:
            Returning None indicates this vertical doesn't have a capability
            provider or doesn't want automatic registration.
        """
        return None

    def get_capability_provider(self) -> Optional[Any]:
        """Get the capability provider for this vertical.

        This convenience method loads and returns the capability provider
        if the vertical has defined one via _get_capability_provider_module().

        Returns:
            Capability provider instance or None if not configured

        Raises:
            ImportError: If the capability provider module cannot be imported
            AttributeError: If the module doesn't have the expected provider

        Example:
            provider = ResearchWorkflowProvider()
            capabilities = provider.get_capability_provider()
            if capabilities:
                print(f"Capabilities: {list(capabilities.get_capabilities().keys())}")
        """
        module_path = self._get_capability_provider_module()
        if not module_path:
            return None

        module, resolved_path = import_module_with_fallback(module_path)

        if module is None:
            logger.warning(
                "Failed to import capability provider from %s (candidates=%s)",
                module_path,
                module_import_candidates(module_path),
            )
            return None

        try:
            # Look for concrete capability provider classes (not abstract base)
            for attr_name in dir(module):
                # Skip private attributes and the abstract base
                if attr_name.startswith("_"):
                    continue
                if attr_name == "BaseCapabilityProvider":
                    continue

                attr = getattr(module, attr_name)
                # Check if it's a concrete class (not abstract) with "CapabilityProvider" in the name
                if (
                    isinstance(attr, type)
                    and "CapabilityProvider" in attr_name
                    and not attr_name.startswith("Base")
                    and hasattr(attr, "get_capabilities")
                    and hasattr(attr, "get_capability_metadata")
                ):
                    try:
                        # Try to instantiate - this will fail for abstract classes
                        instance = attr()
                        # Verify it has the required methods
                        if hasattr(instance, "get_capabilities") and hasattr(
                            instance, "get_capability_metadata"
                        ):
                            return instance
                    except TypeError:
                        # Abstract class or can't be instantiated, skip
                        continue

            logger.warning(
                "No concrete capability provider class found in %s (resolved as %s)",
                module_path,
                resolved_path,
            )
            return None
        except Exception as e:
            logger.warning(
                "Failed to instantiate capability provider from %s (resolved as %s): %s",
                module_path,
                resolved_path,
                e,
            )
            return None

    def _get_workflows_directory(self) -> Path:
        """Return the directory containing YAML workflow files.

        Override this method if workflows are in a non-standard location.
        By default, returns the parent directory of the escape hatches module,
        with "workflows" appended (e.g., victor/research/workflows/).

        Returns:
            Path to the directory containing *.yaml workflow files

        Example:
            def _get_workflows_directory(self) -> Path:
                return Path("/custom/path/to/workflows")
        """
        module_path = self._get_escape_hatches_module()
        module, resolved_path = import_module_with_fallback(module_path)
        if module is None:
            logger.warning(
                "Failed to determine workflows directory: cannot import %s (candidates=%s)",
                module_path,
                module_import_candidates(module_path),
            )
            return Path(__file__).parent

        module_file = getattr(module, "__file__", None)
        if module_file:
            return Path(module_file).parent / "workflows"

        module_paths = getattr(module, "__path__", None)
        if module_paths:
            try:
                return Path(module_paths[0]) / "workflows"
            except Exception:
                pass

        logger.warning(
            "Failed to determine workflows directory for %s (resolved as %s)",
            module_path,
            resolved_path,
        )
        return Path(__file__).parent

    def _load_escape_hatches(self) -> Tuple[
        Dict[str, Callable[[Dict[str, Any]], str]],
        Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
    ]:
        """Dynamically load escape hatches from the vertical-specific module.

        Returns:
            Tuple of (CONDITIONS dict, TRANSFORMS dict) from the escape hatches module

        Raises:
            ImportError: If the escape hatches module cannot be imported
            AttributeError: If CONDITIONS or TRANSFORMS are not defined
        """
        module_path = self._get_escape_hatches_module()
        module, resolved_path = import_module_with_fallback(module_path)
        if module is None:
            logger.warning(
                "Failed to import escape hatches from %s (candidates=%s)",
                module_path,
                module_import_candidates(module_path),
            )
            return {}, {}

        try:
            conditions = getattr(module, "CONDITIONS", {})
            transforms = getattr(module, "TRANSFORMS", {})
            logger.debug(
                "Loaded escape hatches from %s (resolved as %s): %d conditions, %d transforms",
                module_path,
                resolved_path,
                len(conditions),
                len(transforms),
            )
            return conditions, transforms
        except AttributeError as e:
            logger.warning(
                "Missing CONDITIONS/TRANSFORMS in %s (resolved as %s): %s",
                module_path,
                resolved_path,
                e,
            )
            return {}, {}

    def _get_config(self) -> "YAMLWorkflowConfig":
        """Get YAML workflow config with escape hatches registered.

        Creates a YAMLWorkflowConfig instance with the vertical-specific
        conditions and transforms loaded from the escape hatches module.

        Returns:
            YAMLWorkflowConfig instance configured for this vertical
        """
        if self._config is None:
            from victor.workflows.yaml_loader import YAMLWorkflowConfig

            conditions, transforms = self._load_escape_hatches()

            # Merge globally-registered generic escape hatches beneath the
            # provider-specific ones so they are available to every YAML workflow
            # while provider-specific hatches still take precedence (provider wins
            # on name conflicts). Best-effort: never fail config creation on this.
            try:
                from victor.framework.escape_hatch_registry import (
                    get_escape_hatch_registry,
                )
                from victor.workflows.escape_hatches import (
                    ensure_global_escape_hatches_registered,
                )

                ensure_global_escape_hatches_registered()
                (
                    global_conditions,
                    global_transforms,
                ) = get_escape_hatch_registry().get_registry_for_vertical("", include_global=True)
                conditions = {**global_conditions, **conditions}
                transforms = {**global_transforms, **transforms}
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Skipped merging global escape hatches: %s", exc)

            workflows_dir = self._get_workflows_directory()

            self._config = YAMLWorkflowConfig(
                base_dir=workflows_dir,
                condition_registry=conditions,
                transform_registry=transforms,
            )
            logger.debug(f"Created YAML config with base_dir={workflows_dir}")

        return self._config

    def _load_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Lazy load all YAML workflows from the workflows directory.

        Uses escape hatches for complex conditions that can't be expressed in YAML.
        Caches loaded workflows for subsequent access.

        Returns:
            Dict mapping workflow names to WorkflowDefinition instances
        """
        if self._workflows is None:
            try:
                from victor.workflows.yaml_loader import load_workflows_from_directory

                workflows_dir = self._get_workflows_directory()
                config = self._get_config()

                self._workflows = load_workflows_from_directory(
                    workflows_dir,
                    pattern="*.yaml",
                    config=config,
                )
                logger.debug(f"Loaded {len(self._workflows)} YAML workflows from {workflows_dir}")
            except Exception as e:
                logger.warning(f"Failed to load YAML workflows: {e}")
                self._workflows = {}

        return self._workflows

    def get_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Get all workflow definitions for this vertical.

        Returns:
            Dict mapping workflow names to WorkflowDefinition instances
        """
        return self._load_workflows()

    def get_workflow(self, name: str) -> Optional[WorkflowDefinition]:
        """Get a specific workflow by name.

        Args:
            name: The workflow name to retrieve

        Returns:
            WorkflowDefinition if found, None otherwise
        """
        return self._load_workflows().get(name)

    def get_workflow_names(self) -> List[str]:
        """Get list of all available workflow names.

        Returns:
            List of workflow name strings
        """
        return list(self._load_workflows().keys())

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Get automatic workflow triggers based on query patterns.

        This default implementation reads from the AUTO_WORKFLOW_PATTERNS
        class attribute. Subclasses can either:
        1. Set AUTO_WORKFLOW_PATTERNS class attribute (recommended)
        2. Override this method for dynamic behavior

        Returns:
            List of (regex_pattern, workflow_name) tuples

        Example:
            # Using class attribute (recommended)
            class MyProvider(BaseYAMLWorkflowProvider):
                AUTO_WORKFLOW_PATTERNS = [
                    (r"deep\\s+research", "deep_research"),
                ]

            # Or override method (for dynamic behavior)
            def get_auto_workflows(self) -> List[Tuple[str, str]]:
                patterns = super().get_auto_workflows()
                if some_condition:
                    patterns.append((r"custom", "custom_workflow"))
                return patterns
        """
        # Return copy to prevent accidental modification of class attribute
        return list(self.AUTO_WORKFLOW_PATTERNS)

    def get_workflow_for_task_type(self, task_type: str) -> Optional[str]:
        """Get recommended workflow for a task type.

        This default implementation reads from the TASK_TYPE_MAPPINGS
        class attribute. Subclasses can either:
        1. Set TASK_TYPE_MAPPINGS class attribute (recommended)
        2. Override this method for dynamic behavior

        Args:
            task_type: Type of task (e.g., "research", "deploy", "eda")

        Returns:
            Workflow name string or None if no mapping exists

        Example:
            # Using class attribute (recommended)
            class MyProvider(BaseYAMLWorkflowProvider):
                TASK_TYPE_MAPPINGS = {
                    "research": "deep_research",
                    "fact_check": "fact_check",
                }

            # Or override method (for dynamic behavior)
            def get_workflow_for_task_type(self, task_type: str) -> Optional[str]:
                workflow = super().get_workflow_for_task_type(task_type)
                if workflow:
                    return workflow
                # Custom logic
                return "default_workflow"
        """
        return self.TASK_TYPE_MAPPINGS.get(task_type.lower())

    # =========================================================================
    # UnifiedWorkflowCompiler Integration (Recommended Pattern)
    # =========================================================================

    def get_compiler(self) -> "UnifiedWorkflowCompiler":
        """Get or create the unified compiler with caching.

        Returns the UnifiedWorkflowCompiler instance for this provider,
        creating it lazily on first access. The compiler provides consistent
        caching across all workflow compilations.

        **Architecture Note**: UnifiedWorkflowCompiler is the canonical compiler API.
        Vertical providers use it for:
        - Escape hatch integration (condition_registry, transform_registry)
        - Two-level caching (definition + execution)
        - Cache management APIs (get_cache_stats, clear_cache)

        For DI-friendly compile-only usage without caching or execution,
        see WorkflowCompiler in victor.workflows.compiler.

        Returns:
            UnifiedWorkflowCompiler configured with escape hatches

        Example:
            provider = ResearchWorkflowProvider()
            compiler = provider.get_compiler()
            stats = compiler.get_cache_stats()
        """
        if self._compiler is None:
            from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

            self._compiler = UnifiedWorkflowCompiler(enable_caching=True)
        return self._compiler

    def _get_workflow_path(self, workflow_name: str) -> Path:
        """Get the path to a specific workflow YAML file.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Path to the workflow YAML file

        Raises:
            ValueError: If workflow not found
        """
        workflows_dir = self._get_workflows_directory()

        # Try exact filename match first
        exact_path = workflows_dir / f"{workflow_name}.yaml"
        if exact_path.exists():
            return exact_path

        # Try finding in loaded workflows
        workflow = self.get_workflow(workflow_name)
        if workflow is None:
            raise ValueError(f"Workflow not found: {workflow_name}")

        # Default to workflows directory with .yaml extension
        return workflows_dir / f"{workflow_name}.yaml"

    def compile_workflow(self, workflow_name: str) -> "CachedCompiledGraph":
        """Compile a workflow using the unified compiler.

        This is the recommended method for compiling workflows. It uses
        the UnifiedWorkflowCompiler for consistent caching behavior.

        Args:
            workflow_name: Name of the workflow to compile

        Returns:
            CachedCompiledGraph ready for execution with invoke() and stream()

        Raises:
            ValueError: If workflow not found

        Example:
            provider = ResearchWorkflowProvider()
            compiled = provider.compile_workflow("deep_research")
            result = await compiled.invoke({"query": "AI trends"})
        """
        conditions, transforms = self._load_escape_hatches()
        workflow_path = self._get_workflow_path(workflow_name)

        return self.get_compiler().compile_yaml(
            workflow_path,
            workflow_name,
            condition_registry=conditions,
            transform_registry=transforms,
        )

    async def run_compiled_workflow(
        self,
        workflow_name: str,
        context: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> "GraphExecutionResult":
        """Execute a workflow using the unified compiler.

        This method compiles the workflow using the UnifiedWorkflowCompiler
        and executes it. Results benefit from consistent caching.

        Args:
            workflow_name: Name of the workflow to execute
            context: Initial context data for the workflow
            thread_id: Thread ID for checkpointing

        Returns:
            GraphExecutionResult with final state

        Raises:
            ValueError: If workflow not found

        Example:
            provider = ResearchWorkflowProvider()
            result = await provider.run_compiled_workflow(
                "fact_check",
                {"claim": "The Earth is round"}
            )
            print(result.state)
        """
        compiled = self.compile_workflow(workflow_name)
        return await compiled.invoke(context or {}, thread_id=thread_id)

    async def stream_compiled_workflow(
        self,
        workflow_name: str,
        context: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[tuple]:
        """Stream workflow execution using the unified compiler.

        This method compiles the workflow using the UnifiedWorkflowCompiler
        and streams its execution. Yields (node_id, state) tuples after
        each node completes.

        Args:
            workflow_name: Name of the workflow to execute
            context: Initial context data for the workflow
            thread_id: Thread ID for checkpointing

        Yields:
            Tuple of (node_id, state) after each node execution

        Raises:
            ValueError: If workflow not found

        Example:
            provider = ResearchWorkflowProvider()
            async for node_id, state in provider.stream_compiled_workflow(
                "deep_research",
                {"query": "AI trends"}
            ):
                print(f"Completed: {node_id}")
        """
        compiled = self.compile_workflow(workflow_name)
        async for node_id, state in compiled.stream(context or {}, thread_id=thread_id):
            yield node_id, state

    def __repr__(self) -> str:
        """Return a string representation of this provider."""
        class_name = self.__class__.__name__
        workflow_count = len(self._load_workflows())
        return f"{class_name}(workflows={workflow_count})"


__all__ = [
    "BaseYAMLWorkflowProvider",
]
