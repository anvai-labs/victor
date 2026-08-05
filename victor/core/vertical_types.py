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

"""Core vertical types for cross-vertical abstraction.

This module defines fundamental types used across multiple verticals,
placed here in core to:
1. Avoid circular imports between core and verticals
2. Enable framework components to work with vertical abstractions
3. Provide a single source of truth for shared type definitions

These types are re-exported from `victor.core.verticals.base` and
`victor.core.verticals.protocols` for backward compatibility.

Type Categories:
    - Stage Types: StageDefinition for workflow stages
    - Task Types: TaskTypeHint for task-specific prompt hints
    - Middleware Types: MiddlewarePriority, MiddlewareResult
    - Tool Types: TieredToolConfig for intelligent tool selection

Note:
    For vertical-specific protocols (MiddlewareProtocol, SafetyExtensionProtocol,
    etc.), see `victor.core.verticals.protocols`. Only data types are defined here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Set, TYPE_CHECKING, Union

from victor.core.grounding_texts import (
    GROUNDING_RULES as CANONICAL_GROUNDING_RULES,
    GROUNDING_RULES_EXTENDED as CANONICAL_GROUNDING_RULES_EXTENDED,
)
from victor_contracts.core.types import (
    StageDefinition,
    StageDefinitionLike,
    Tier,
    normalize_stage_definition,
    normalize_stage_definitions,
)
from victor_contracts.verticals.protocols.promoted_types import (
    MiddlewarePriority,
    MiddlewareResult,
    TaskTypeHintData as TaskTypeHint,
)

if TYPE_CHECKING:
    from victor.framework.tools import ToolSet


# =============================================================================
# Stage Types
# =============================================================================

# Canonical stage contract now lives in victor-contracts. This module re-exports the
# shared definition for in-repo compatibility while StageBuilder and older
# framework shims continue to use the familiar import path.


# =============================================================================
# Task Type Hints
# =============================================================================


# Canonical task hint contract now lives in victor-contracts.


# =============================================================================
# Standard Task Hints (Common patterns across verticals)
# =============================================================================


class StandardTaskHints:
    """Standard task hints shared across verticals.

    Provides common task type hints that apply to multiple verticals,
    reducing duplication while allowing vertical-specific customization.

    Example:
        # Get a standard hint
        general_hint = StandardTaskHints.get("general")

        # Merge with vertical-specific hints
        vertical_hints = StandardTaskHints.merge_with(my_vertical_hints)

        # Get all standard hints for a vertical type
        hints = StandardTaskHints.for_vertical("coding")
    """

    # Base hint templates that can be customized
    GENERAL: TaskTypeHint = TaskTypeHint(
        task_type="general",
        hint="[GENERAL] Moderate exploration. 3-6 tool calls. Answer concisely.",
        tool_budget=8,
        priority_tools=["read", "grep", "ls"],
    )

    SEARCH: TaskTypeHint = TaskTypeHint(
        task_type="search",
        hint="[SEARCH] Use grep/ls for exploration. Summarize after 2-4 calls.",
        tool_budget=6,
        priority_tools=["grep", "ls", "read"],
    )

    CREATE: TaskTypeHint = TaskTypeHint(
        task_type="create",
        hint="[CREATE] Read 1-2 relevant files for context, then create. Follow existing patterns.",
        tool_budget=5,
        priority_tools=["read", "write"],
    )

    EDIT: TaskTypeHint = TaskTypeHint(
        task_type="edit",
        hint="[EDIT] Read target file first, then modify. Focused changes only.",
        tool_budget=5,
        priority_tools=["read", "edit"],
    )

    ANALYZE: TaskTypeHint = TaskTypeHint(
        task_type="analyze",
        hint="[ANALYZE] Examine content carefully. Read related files. Structured findings.",
        tool_budget=12,
        priority_tools=["read", "grep"],
    )

    # Standard hints as a dictionary
    _STANDARD_HINTS: Dict[str, TaskTypeHint] = {}

    @classmethod
    def _init_standard_hints(cls) -> None:
        """Initialize standard hints dictionary (lazy)."""
        if not cls._STANDARD_HINTS:
            cls._STANDARD_HINTS = {
                "general": cls.GENERAL,
                "search": cls.SEARCH,
                "create": cls.CREATE,
                "edit": cls.EDIT,
                "analyze": cls.ANALYZE,
            }

    @classmethod
    def get(cls, task_type: str) -> Optional[TaskTypeHint]:
        """Get a standard hint by task type.

        Args:
            task_type: Task type name

        Returns:
            TaskTypeHint or None if not found
        """
        cls._init_standard_hints()
        return cls._STANDARD_HINTS.get(task_type.lower())

    @classmethod
    def all(cls) -> Dict[str, TaskTypeHint]:
        """Get all standard hints.

        Returns:
            Dict of task type to TaskTypeHint
        """
        cls._init_standard_hints()
        return cls._STANDARD_HINTS.copy()

    @classmethod
    def merge_with(cls, vertical_hints: Dict[str, TaskTypeHint]) -> Dict[str, TaskTypeHint]:
        """Merge standard hints with vertical-specific hints.

        Vertical-specific hints override standard hints with the same key.

        Args:
            vertical_hints: Vertical-specific task type hints

        Returns:
            Merged dict with standard + vertical hints
        """
        cls._init_standard_hints()
        result = cls._STANDARD_HINTS.copy()
        result.update(vertical_hints)
        return result


# =============================================================================
# Standard Grounding Rules
# =============================================================================


class StandardGroundingRules:
    """Standard grounding rules shared across verticals.

    Provides base grounding rule templates that verticals can use
    or extend with domain-specific rules.

    External verticals can register their own addendums via
    ``register_addendum()`` without modifying this module (OCP).
    """

    # Base grounding rule - applies to all verticals
    BASE: str = CANONICAL_GROUNDING_RULES

    # Extended grounding for local models
    EXTENDED: str = CANONICAL_GROUNDING_RULES_EXTENDED

    # Registry for vertical-specific addendums (OCP: open for extension).
    # Verticals register addendums via register_addendum() in their
    # VictorPlugin.register() call — no hardcoded strings in core.
    _grounding_addendums: Dict[str, str] = {}

    @classmethod
    def register_addendum(cls, vertical_name: str, text: str) -> None:
        """Register a grounding addendum for a vertical.

        External verticals call this to extend grounding rules
        without modifying this module.

        Args:
            vertical_name: Vertical name (e.g. "research", "devops")
            text: Addendum text to append after base grounding rules
        """
        cls._grounding_addendums[vertical_name] = text

    @classmethod
    def unregister_addendum(cls, vertical_name: str) -> None:
        """Remove a previously registered addendum.

        Args:
            vertical_name: Vertical name to remove
        """
        cls._grounding_addendums.pop(vertical_name, None)

    @classmethod
    def get_base(cls, extended: bool = False) -> str:
        """Get base grounding rules.

        Args:
            extended: Whether to use extended rules (for local models)

        Returns:
            Grounding rules string
        """
        return cls.EXTENDED if extended else cls.BASE

    @classmethod
    def for_vertical(cls, vertical: str, extended: bool = False) -> str:
        """Get grounding rules for a specific vertical.

        Looks up addendums from the registry populated via
        ``register_addendum()``.

        Args:
            vertical: Vertical name
            extended: Whether to use extended rules

        Returns:
            Grounding rules string with vertical addendum
        """
        base = cls.get_base(extended)
        addendum = cls._grounding_addendums.get(vertical, "")
        if addendum:
            return f"{base}\n{addendum}"
        return base

    # NOTE: No _register_defaults() — verticals register their own
    # addendums via VictorPlugin.register(context) calls at startup.


# =============================================================================
# Middleware Types
# =============================================================================

# Canonical middleware contracts now live in victor-contracts.


# =============================================================================
# Tiered Tool Configuration
# =============================================================================


@dataclass
class TieredToolConfig:
    """Tiered tool configuration for intelligent tool selection.

    Implements a three-tier system for context-efficient tool management:
    1. Mandatory (always included): Essential tools for any task
    2. Vertical Core (always included for this vertical): Domain-specific core tools
    3. Semantic/Contextual (selected dynamically): Additional tools based on task

    Each tier can specify read-only vs read-write tools to enable
    intelligent filtering based on task intent (analysis vs modification).

    Migration Note:
        The `semantic_pool` and `stage_tools` fields are being deprecated in favor
        of using @tool decorator metadata:
        - `semantic_pool`: Will be derived from ToolMetadataRegistry.get_all_tool_names()
          minus mandatory/vertical_core. Most tools should be candidates.
        - `stage_tools`: Will be derived from @tool(stages=[...]) decorator metadata.
          Use ToolMetadataRegistry.get_tools_by_stage() instead.

    Example:
        # Preferred (new style)
        TieredToolConfig(
            mandatory={"read", "ls", "grep"},            # Essential for any task
            vertical_core={"web", "fetch", "overview"},  # Research-specific core
            readonly_only_for_analysis=True,             # Hide write tools for analysis
        )

    Attributes:
        mandatory: Tools always included regardless of task type
        vertical_core: Tools always included for this vertical
        semantic_pool: DEPRECATED - Tools selected via semantic matching
        stage_tools: DEPRECATED - Tools available at specific stages
        readonly_only_for_analysis: If True, hide write/execute tools for analysis tasks
    """

    _deprecation_warned: ClassVar[Set[str]] = set()

    basic_tools: List[str] = field(default_factory=list)
    standard_tools: List[str] = field(default_factory=list)
    advanced_tools: List[str] = field(default_factory=list)
    mandatory: Set[str] = field(default_factory=set)
    vertical_core: Set[str] = field(default_factory=set)
    semantic_pool: Set[str] = field(default_factory=set)  # DEPRECATED: derive from registry
    stage_tools: Dict[str, Set[str]] = field(
        default_factory=dict
    )  # DEPRECATED: use @tool(stages=[])
    readonly_only_for_analysis: bool = True

    def __post_init__(self) -> None:
        explicit_semantic_pool = bool(self.semantic_pool)
        explicit_stage_tools = bool(self.stage_tools)

        normalized_mandatory = set(self.mandatory) or set(self.basic_tools)
        normalized_vertical_core = set(self.vertical_core) or set(self.standard_tools)
        normalized_semantic_pool = set(self.semantic_pool) or set(self.advanced_tools)
        normalized_stage_tools = {
            stage_name: set(tool_names) for stage_name, tool_names in self.stage_tools.items()
        }

        self.mandatory = normalized_mandatory
        self.vertical_core = normalized_vertical_core
        self.semantic_pool = normalized_semantic_pool
        self.stage_tools = normalized_stage_tools

        if not self.basic_tools:
            self.basic_tools = sorted(normalized_mandatory)
        if not self.standard_tools:
            self.standard_tools = sorted(normalized_vertical_core)
        if not self.advanced_tools:
            self.advanced_tools = sorted(normalized_semantic_pool)

        if explicit_semantic_pool and "semantic_pool" not in TieredToolConfig._deprecation_warned:
            warnings.warn(
                "TieredToolConfig.semantic_pool is deprecated. "
                "Use get_effective_semantic_pool() instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            TieredToolConfig._deprecation_warned.add("semantic_pool")
        if explicit_stage_tools and "stage_tools" not in TieredToolConfig._deprecation_warned:
            warnings.warn(
                "TieredToolConfig.stage_tools is deprecated. "
                "Use get_tools_for_stage_from_registry() instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            TieredToolConfig._deprecation_warned.add("stage_tools")

    @classmethod
    def reset_deprecation_warnings(cls) -> None:
        """Reset deprecation warning tracking to allow re-emission."""
        cls._deprecation_warned.clear()

    def get_base_tools(self) -> Set[str]:
        """Get tools always included (mandatory + vertical core)."""
        return self.mandatory | self.vertical_core

    def get_all_tools(self) -> Set[str]:
        """Get all tools in the configuration."""
        all_tools = self.mandatory | self.vertical_core | self.semantic_pool
        for stage_set in self.stage_tools.values():
            all_tools |= stage_set
        return all_tools

    def get_tools_for_stage(self, stage: str) -> Set[str]:
        """Get tools for a specific stage.

        Args:
            stage: Stage name (e.g., "INITIAL", "SEARCHING", "WRITING")

        Returns:
            Set of tool names for the stage (base + stage-specific)
        """
        base = self.get_base_tools()
        stage_specific = self.stage_tools.get(stage, set())
        return base | stage_specific

    def get_semantic_pool_from_registry(self) -> Set[str]:
        """Get semantic pool dynamically from ToolMetadataRegistry.

        This method derives the semantic pool from all registered tools
        minus the mandatory and vertical_core tools. Use this instead of
        the static semantic_pool field for new implementations.

        Returns:
            Set of tool names for semantic selection
        """
        from victor.tools.metadata_registry import ToolMetadataRegistry

        registry = ToolMetadataRegistry.get_instance()
        all_tools = registry.get_all_tool_names()
        # Semantic pool = all tools - base tools (mandatory + vertical_core)
        base = self.get_base_tools()
        return all_tools - base

    def get_effective_semantic_pool(self) -> Set[str]:
        """Get effective semantic pool, preferring registry over static.

        Returns:
            semantic_pool if explicitly set, otherwise derives from registry
        """
        if self.semantic_pool:
            return self.semantic_pool
        return self.get_semantic_pool_from_registry()

    def get_tools_for_tier(self, tier: Union[Tier, str]) -> List[str]:
        """Get tools available at a specific compatibility tier."""

        tier = Tier(tier) if isinstance(tier, str) else tier

        if tier == Tier.BASIC:
            return self.basic_tools.copy()
        if tier == Tier.STANDARD:
            return self.basic_tools + self.standard_tools
        if tier == Tier.ADVANCED:
            return self.basic_tools + self.standard_tools + self.advanced_tools

        return self.basic_tools.copy()

    def get_max_tier_for_tools(self, available_tools: List[str]) -> Tier:
        """Determine the highest SDK compatibility tier for a tool set."""

        available_set = set(available_tools)

        if available_set.issuperset(
            set(self.basic_tools + self.standard_tools + self.advanced_tools)
        ):
            return Tier.ADVANCED
        if available_set.issuperset(set(self.basic_tools + self.standard_tools)):
            return Tier.STANDARD
        return Tier.BASIC

    def get_tools_for_stage_from_registry(self, stage: str) -> Set[str]:
        """Get tools for a stage using @tool decorator metadata.

        Args:
            stage: Stage name (e.g., "INITIAL", "READING", "EXECUTION")

        Returns:
            Base tools plus stage-specific tools from registry
        """
        from victor.tools.metadata_registry import ToolMetadataRegistry

        registry = ToolMetadataRegistry.get_instance()
        base = self.get_base_tools()
        registry_stage_tools = registry.get_tools_by_stage(stage)
        return base | registry_stage_tools


# =============================================================================
# Vertical Config Base Types
# =============================================================================


@dataclass
class VerticalConfigBase:
    """Base configuration for a vertical.

    This is a simplified base type that can be extended by VerticalConfig
    in victor.core.verticals.base. It provides the core structure without
    requiring framework dependencies.

    Note:
        For the full VerticalConfig with ToolSet support, use
        `victor.core.verticals.base.VerticalConfig` instead.

    Attributes:
        system_prompt: System prompt text
        stages: Stage definitions
        provider_hints: Hints for provider selection
        evaluation_criteria: Criteria for evaluating agent performance
        metadata: Additional vertical-specific metadata
    """

    system_prompt: str = ""
    stages: Dict[str, StageDefinition] = field(default_factory=dict)
    provider_hints: Dict[str, Any] = field(default_factory=dict)
    evaluation_criteria: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Tiered Tool Template (Factory for Reducing Duplication)
# =============================================================================


class TieredToolTemplate:
    """Factory for creating TieredToolConfig with common patterns.

    This template reduces duplication by providing:
    1. Standard mandatory tools used by all verticals
    2. Pre-built configurations for common verticals
    3. Easy customization via factory methods

    Example usage:
        # Simple: use pre-built vertical config
        config = TieredToolTemplate.for_vertical("coding")

        # Custom: specify only vertical-specific tools
        config = TieredToolTemplate.create(
            vertical_core={"edit", "write", "shell", "git"},
            readonly_only_for_analysis=False,
        )

        # Override mandatory tools (rare)
        config = TieredToolTemplate.create(
            mandatory={"read", "ls"},  # Custom mandatory set
            vertical_core={"web_search", "web_fetch"},
        )
    """

    # Standard mandatory tools - essential for any task across all verticals
    DEFAULT_MANDATORY: Set[str] = {"read", "ls", "grep"}

    # Legacy dicts emptied — verticals MUST register via register_vertical_tools().
    # Kept as empty dicts for structural backward compatibility.
    _LEGACY_CORES: Dict[str, Set[str]] = {}
    _LEGACY_READONLY: Dict[str, bool] = {}

    # Backward-compat aliases (empty, kept for import compatibility)
    VERTICAL_CORES = _LEGACY_CORES
    VERTICAL_READONLY_DEFAULTS = _LEGACY_READONLY

    # Dynamic registry for vertical tool configs (OCP extension point)
    # Verticals register via register_vertical_tools() at activation time.
    # Takes precedence over VERTICAL_CORES/VERTICAL_READONLY_DEFAULTS.
    _registered_verticals: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_vertical_tools(
        cls,
        vertical: str,
        core_tools: Set[str],
        readonly_for_analysis: bool = True,
    ) -> None:
        """Register tool configuration for a vertical dynamically.

        This enables new verticals to register their tool configs without
        modifying core code (OCP compliance).

        Args:
            vertical: Vertical name
            core_tools: Set of core tool names for this vertical
            readonly_for_analysis: Whether to restrict writes in analysis mode
        """
        cls._registered_verticals[vertical] = {
            "core_tools": core_tools,
            "readonly_for_analysis": readonly_for_analysis,
        }

        # Propagate to ToolTierRegistry so the tool access controller
        # and other consumers can discover the tier centrally.
        try:
            from victor.core.tool_tier_registry import ToolTierRegistry

            config = cls.create(
                vertical_core=core_tools,
                readonly_only_for_analysis=readonly_for_analysis,
            )
            registry = ToolTierRegistry.get_instance()
            registry.register(
                name=vertical,
                config=config,
                parent="base",
                description=f"Tiered configuration for {vertical} vertical",
                overwrite=True,
            )
        except Exception:
            pass  # Registry not available yet during early bootstrap

    @classmethod
    def create(
        cls,
        vertical_core: Set[str],
        mandatory: Optional[Set[str]] = None,
        readonly_only_for_analysis: bool = True,
        semantic_pool: Optional[Set[str]] = None,
        stage_tools: Optional[Dict[str, Set[str]]] = None,
    ) -> TieredToolConfig:
        """Create a TieredToolConfig with standard mandatory tools.

        Args:
            vertical_core: Domain-specific core tools for the vertical
            mandatory: Override mandatory tools (uses DEFAULT_MANDATORY if None)
            readonly_only_for_analysis: Whether to hide write tools for analysis
            semantic_pool: DEPRECATED - tools for semantic selection
            stage_tools: DEPRECATED - stage-specific tools

        Returns:
            Configured TieredToolConfig
        """
        return TieredToolConfig(
            mandatory=(mandatory if mandatory is not None else cls.DEFAULT_MANDATORY.copy()),
            vertical_core=vertical_core,
            semantic_pool=semantic_pool or set(),
            stage_tools=stage_tools or {},
            readonly_only_for_analysis=readonly_only_for_analysis,
        )

    @classmethod
    def for_vertical(cls, vertical: str) -> Optional[TieredToolConfig]:
        """Get TieredToolConfig for a vertical.

        Checks dynamic registry first, then falls back to built-in defaults.

        Args:
            vertical: Vertical name

        Returns:
            Configured TieredToolConfig or None if vertical not known
        """
        # Dynamic registry only (OCP extension point)
        if vertical in cls._registered_verticals:
            reg = cls._registered_verticals[vertical]
            return cls.create(
                vertical_core=set(reg["core_tools"]),
                readonly_only_for_analysis=reg.get("readonly_for_analysis", True),
            )

        return None

    @classmethod
    def unregister_vertical_tools(cls, vertical: str) -> None:
        """Remove a vertical's tool configuration from the dynamic registry.

        Primarily for test cleanup to ensure isolation between tests.

        Args:
            vertical: Vertical name to unregister
        """
        cls._registered_verticals.pop(vertical, None)

        # Propagate removal to ToolTierRegistry
        try:
            from victor.core.tool_tier_registry import ToolTierRegistry

            registry = ToolTierRegistry.get_instance()
            registry.unregister(vertical)
        except Exception:
            pass  # Registry not available or entry doesn't exist

    @classmethod
    def list_verticals(cls) -> List[str]:
        """List all dynamically registered verticals."""
        return list(cls._registered_verticals.keys())


# =============================================================================
# Stage Builder (Factory for Reducing Duplication)
# =============================================================================


class StageBuilder:
    """Builder class for creating StageDefinition instances with fluent API.

    Reduces boilerplate code when defining stages across verticals. Instead of
    directly instantiating StageDefinition dataclass, use this builder for a
    cleaner, more readable syntax.

    Also provides factory methods for standard/common stages that are shared
    across verticals (INITIAL, COMPLETION, etc.).

    Example:
        builder = StageBuilder()
        stages = {
            "INITIAL": builder.standard_initial(),
            "PLANNING": builder.stage("PLANNING")
                .description("Planning the implementation")
                .tools({ToolNames.READ, ToolNames.GREP})
                .keywords(["plan", "design"])
                .next_stages({"READING", "EXECUTION"})
                .build(),
            "COMPLETION": builder.standard_completion(),
        }

    Or for complete workflow definition:
        builder = StageBuilder()
        stages = (
            builder.workflow("MyVertical")
            .initial()
            .stage("PLANNING")
                .description("Planning")
                .tools({ToolNames.READ})
                .keywords(["plan"])
            .stage("EXECUTION")
                .description("Executing")
                .tools({ToolNames.WRITE})
                .keywords(["execute"])
            .completion()
            .build()
        )
    """

    # Track workflow stages being built
    _stages: Dict[str, StageDefinition]
    _last_stage_name: Optional[str]

    def __init__(self) -> None:
        """Initialize the stage builder."""
        self._stages = {}
        self._last_stage_name = None
        self._current_name = ""
        self._current_description = ""
        self._current_tools: Set[str] = set()
        self._current_keywords: List[str] = []
        self._current_next_stages: Set[str] = set()

    # =========================================================================
    # Fluent API Methods (for stage() chaining)
    # =========================================================================

    def stage(self, name: str) -> "StageBuilder":
        """Start building a new stage definition.

        Args:
            name: Stage name (e.g., "PLANNING", "EXECUTION")

        Returns:
            Self for method chaining
        """
        self._current_name = name
        self._current_description = ""
        self._current_tools = set()
        self._current_keywords = []
        self._current_next_stages = set()
        return self

    def description(self, description: str) -> "StageBuilder":
        """Set stage description.

        Args:
            description: Human-readable description

        Returns:
            Self for method chaining
        """
        self._current_description = description
        return self

    def tools(self, tools: Set[str]) -> "StageBuilder":
        """Set tools available at this stage.

        Args:
            tools: Set of tool names (canonical names from ToolNames)

        Returns:
            Self for method chaining
        """
        self._current_tools = tools
        return self

    def keywords(self, keywords: List[str]) -> "StageBuilder":
        """Set keywords that suggest this stage.

        Args:
            keywords: List of keywords for stage detection

        Returns:
            Self for method chaining
        """
        self._current_keywords = keywords
        return self

    def next_stages(self, next_stages: Set[str]) -> "StageBuilder":
        """Set valid next stages for transitions.

        Args:
            next_stages: Set of stage names that can follow this stage

        Returns:
            Self for method chaining
        """
        self._current_next_stages = next_stages
        return self

    def build(self) -> StageDefinition:
        """Build the current stage definition.

        Returns:
            StageDefinition instance with configured properties
        """
        return StageDefinition(
            name=self._current_name,
            description=self._current_description,
            tools=self._current_tools,
            keywords=self._current_keywords,
            next_stages=self._current_next_stages,
        )

    def add(self, name: str, stage_def: StageDefinition) -> "StageBuilder":
        """Add a pre-built StageDefinition to the collection.

        Args:
            name: Stage name key
            stage_def: StageDefinition instance

        Returns:
            Self for method chaining
        """
        self._stages[name] = stage_def
        self._last_stage_name = name
        return self

    # =========================================================================
    # Standard Stage Factories
    # =========================================================================

    def standard_initial(self) -> StageDefinition:
        """Create a standard INITIAL stage definition.

        Returns:
            StageDefinition for INITIAL stage
        """
        return StageDefinition(
            name="INITIAL",
            description="Understanding the request",
            tools=set(),  # Empty - verticals should customize
            keywords=["what", "how", "explain", "understand"],
            next_stages=set(),  # Verticals should customize
        )

    def standard_completion(self) -> StageDefinition:
        """Create a standard COMPLETION stage definition.

        Returns:
            StageDefinition for COMPLETION stage
        """
        return StageDefinition(
            name="COMPLETION",
            description="Task complete",
            tools=set(),
            keywords=["done", "complete", "finish"],
            next_stages=set(),  # No stages after completion
        )

    # =========================================================================
    # Workflow Builder API
    # =========================================================================

    def workflow(self, name: str) -> "StageBuilder":
        """Start building a complete workflow.

        Resets internal state and prepares for building a series of stages.

        Args:
            name: Workflow name (for documentation/debugging)

        Returns:
            Self for method chaining
        """
        self._stages = {}
        self._last_stage_name = None
        return self

    def initial(self) -> "StageBuilder":
        """Add standard INITIAL stage and set it as last stage.

        Returns:
            Self for method chaining
        """
        initial_stage = self.standard_initial()
        self._stages["INITIAL"] = initial_stage
        self._last_stage_name = "INITIAL"
        return self

    def completion(self) -> "StageBuilder":
        """Add standard COMPLETION stage.

        Returns:
            Self for method chaining
        """
        completion_stage = self.standard_completion()
        self._stages["COMPLETION"] = completion_stage
        return self

    def build_workflow(self) -> Dict[str, StageDefinition]:
        """Build and return the complete workflow stages dictionary.

        Returns:
            Dict mapping stage names to StageDefinition instances
        """
        return self._stages.copy()


__all__ = [
    # Stage Types
    "StageDefinition",
    "StageBuilder",
    # Task Types
    "TaskTypeHint",
    "StandardTaskHints",
    "StandardGroundingRules",
    # Middleware Types
    "MiddlewarePriority",
    "MiddlewareResult",
    # Tool Configuration
    "TieredToolConfig",
    "TieredToolTemplate",
    # Config Base
    "VerticalConfigBase",
]
