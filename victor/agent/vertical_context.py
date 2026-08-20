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

"""Vertical context container for orchestrator state.

This module provides a structured container for all vertical-related state,
replacing scattered private attributes with a proper, protocol-compliant
data structure.

Design Philosophy:
- Single source of truth for vertical state
- Protocol-based access for SOLID compliance
- Backward compatibility with existing code
- Type-safe access to all vertical configuration

Capability Config Storage Pattern
----------------------------------
The `capability_configs` dict provides centralized storage for vertical
capability configurations, replacing direct orchestrator attribute assignment.
This pattern follows SOLID principles by avoiding tight coupling between
verticals and the orchestrator implementation.

OLD Pattern (avoid - creates tight coupling):
    # In vertical integration code
    orchestrator.rag_config = {"chunk_size": 512}
    orchestrator.source_verification_config = {"strict": True}
    orchestrator.code_style = {"max_line_length": 100}

NEW Pattern (preferred - decoupled via context):
    # In vertical integration code
    context.set_capability_config("rag_config", {"chunk_size": 1024})
    context.set_capability_config("source_verification_config", {"strict": True})
    context.set_capability_config("code_style", {"max_line_length": 100})

    # Or bulk apply
    context.apply_capability_configs({
        "rag_config": {"chunk_size": 512},
        "source_verification_config": {"strict": True},
        "code_style": {"max_line_length": 100},
    })

    # Retrieve elsewhere
    rag_config = context.get_capability_config("rag_config", {})
    if rag_config.get("strict"):
        # Apply strict mode
        ...

Benefits:
- Verticals don't need to know orchestrator's internal structure
- Easy to add new configs without modifying orchestrator class
- Type-safe via protocol methods
- Clear separation of concerns

Usage:
    from victor.agent.vertical_context import VerticalContext

    # Create context
    context = VerticalContext(name="coding")

    # Apply vertical configuration
    context.apply_stages(stages)
    context.apply_middleware(middleware_list)
    context.apply_safety_patterns(patterns)

    # Store capability configs
    context.set_capability_config("rag_config", {"chunk_size": 1024})
    context.apply_capability_configs({
        "code_style": {"max_line_length": 100},
        "test_framework": "pytest",
    })

    # Query context
    if context.has_middleware:
        for mw in context.middleware:
            await mw.before_tool_call(...)

    # Retrieve capability configs
    code_style = context.get_capability_config("code_style", {})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
)

if TYPE_CHECKING:
    from victor.core.verticals.base import VerticalConfig
    from victor.core.verticals.protocols import (
        MiddlewareProtocol,
        ModeConfig,
        SafetyPattern,
        TaskTypeHint,
        ToolDependency,
    )
    from victor.core.vertical_types import TieredToolConfig


# =============================================================================
# Protocols for vertical context access
# =============================================================================


# Re-export from canonical location for backward compatibility.
from victor.core.shared_types import (  # noqa: F401
    MutableVerticalContextProtocol,
    VerticalContextProtocol,
)

# =============================================================================
# Vertical Context Data Class
# =============================================================================


@dataclass
class VerticalContext:
    """Container for all vertical-related state.

    This dataclass consolidates all vertical configuration that was
    previously scattered across private attributes on the orchestrator.

    Replaces:
        - _vertical_name
        - _vertical_config
        - _vertical_stages
        - _vertical_middleware
        - _vertical_safety_patterns
        - _vertical_task_hints
        - _vertical_mode_configs
        - _vertical_default_mode
        - _vertical_default_budget
        - _vertical_tool_dependencies
        - _vertical_tool_sequences
        - _framework_system_prompt

    Attributes:
        name: Vertical name (e.g., "coding", "devops", "research")
        config: Full vertical configuration object
        stages: Stage definitions from vertical
        middleware: Middleware implementations
        safety_patterns: Safety patterns from vertical
        task_hints: Task type hints for prompt building
        mode_configs: Mode configurations
        default_mode: Default mode name
        default_budget: Default tool budget
        tool_dependencies: Tool dependency definitions
        tool_sequences: Common tool sequences
        system_prompt: Custom system prompt from vertical
        enabled_tools: Set of enabled tool names from vertical
    """

    # Core vertical identity
    name: Optional[str] = None
    config: Optional["VerticalConfig"] = None

    # Stage configuration
    stages: Dict[str, Any] = field(default_factory=dict)

    # Middleware chain
    middleware: List[Any] = field(default_factory=list)

    # Safety extensions
    safety_patterns: List[Any] = field(default_factory=list)

    # Prompt contributions
    task_hints: Dict[str, Any] = field(default_factory=dict)
    system_prompt: Optional[str] = None
    prompt_sections: List[str] = field(default_factory=list)

    # Mode configuration
    mode_configs: Dict[str, Any] = field(default_factory=dict)
    default_mode: str = "default"
    default_budget: int = 10

    # Tool dependencies
    tool_dependencies: List[Any] = field(default_factory=list)
    tool_sequences: List[List[str]] = field(default_factory=list)

    # Tool filter
    enabled_tools: Set[str] = field(default_factory=set)

    # Capability negotiation results (Phase 4)
    capability_negotiation_results: Optional[Dict[str, Any]] = None

    # New framework integrations (workflows, RL, teams)
    workflows: Dict[str, Any] = field(default_factory=dict)
    rl_config: Optional[Any] = None
    rl_hooks: Optional[Any] = None
    team_specs: Dict[str, Any] = field(default_factory=dict)

    # Tiered tool configuration (Phase 1: Gap fix)
    tiered_config: Optional["TieredToolConfig"] = None

    # Enrichment strategy (Phase 1: Gap fix)
    enrichment_strategy: Optional[Any] = None

    # Tool selection strategy (SOLID: vertical-specific tool selection)
    tool_selection_strategy: Optional[Any] = None

    # Capability configurations (SOLID: centralized config storage)
    # Replaces direct orchestrator attribute assignment patterns like:
    # orchestrator.rag_config = {...}
    # orchestrator.source_verification_config = {...}
    capability_configs: Dict[str, Any] = field(default_factory=dict)

    # ==========================================================================
    # Property Accessors
    # ==========================================================================

    @property
    def vertical_name(self) -> Optional[str]:
        """Get the vertical name."""
        return self.name

    @property
    def has_vertical(self) -> bool:
        """Check if a vertical is configured."""
        return self.name is not None

    @property
    def has_middleware(self) -> bool:
        """Check if middleware is configured."""
        return len(self.middleware) > 0

    @property
    def has_safety_patterns(self) -> bool:
        """Check if safety patterns are configured."""
        return len(self.safety_patterns) > 0

    @property
    def has_mode_configs(self) -> bool:
        """Check if mode configs are configured."""
        return len(self.mode_configs) > 0

    @property
    def has_tool_dependencies(self) -> bool:
        """Check if tool dependencies are configured."""
        return len(self.tool_dependencies) > 0

    @property
    def has_custom_prompt(self) -> bool:
        """Check if a custom system prompt is set."""
        return self.system_prompt is not None

    @property
    def has_workflows(self) -> bool:
        """Check if workflows are configured."""
        return len(self.workflows) > 0

    @property
    def has_rl_config(self) -> bool:
        """Check if RL config is configured."""
        return self.rl_config is not None

    @property
    def has_team_specs(self) -> bool:
        """Check if team specs are configured."""
        return len(self.team_specs) > 0

    @property
    def has_tiered_config(self) -> bool:
        """Check if tiered tool config is configured."""
        return self.tiered_config is not None

    @property
    def has_enrichment_strategy(self) -> bool:
        """Check if enrichment strategy is configured."""
        return self.enrichment_strategy is not None

    @property
    def has_tool_selection_strategy(self) -> bool:
        """Check if tool selection strategy is configured."""
        return self.tool_selection_strategy is not None

    @property
    def has_capability_configs(self) -> bool:
        """Check if capability configs are stored."""
        return len(self.capability_configs) > 0

    # ==========================================================================
    # Mutation Methods (implements MutableVerticalContextProtocol)
    # ==========================================================================

    def apply_vertical(
        self,
        name: str,
        config: Optional["VerticalConfig"] = None,
    ) -> None:
        """Apply a vertical to this context.

        Args:
            name: Vertical name
            config: Optional vertical configuration
        """
        self.name = name
        self.config = config

    def apply_stages(self, stages: Dict[str, Any]) -> None:
        """Apply stage configuration.

        Args:
            stages: Stage definitions dict
        """
        self.stages = stages

    def apply_middleware(self, middleware: List[Any]) -> None:
        """Apply middleware list.

        Args:
            middleware: List of MiddlewareProtocol implementations
        """
        self.middleware = middleware

    def apply_safety_patterns(self, patterns: List[Any]) -> None:
        """Apply safety patterns.

        Args:
            patterns: List of SafetyPattern instances
        """
        self.safety_patterns = patterns

    def apply_task_hints(self, hints: Dict[str, Any]) -> None:
        """Apply task type hints.

        Args:
            hints: Dict mapping task types to TaskTypeHint
        """
        self.task_hints = hints

    def apply_mode_configs(
        self,
        configs: Dict[str, Any],
        default_mode: str = "default",
        default_budget: int = 10,
    ) -> None:
        """Apply mode configurations.

        Args:
            configs: Dict mapping mode names to ModeConfig
            default_mode: Default mode name
            default_budget: Default tool budget
        """
        self.mode_configs = configs
        self.default_mode = default_mode
        self.default_budget = default_budget

    def apply_tool_dependencies(
        self,
        dependencies: List[Any],
        sequences: Optional[List[List[str]]] = None,
    ) -> None:
        """Apply tool dependencies.

        Args:
            dependencies: List of ToolDependency instances
            sequences: List of tool sequences
        """
        self.tool_dependencies = dependencies
        self.tool_sequences = sequences or []

    def apply_system_prompt(self, prompt: str) -> None:
        """Apply custom system prompt.

        Args:
            prompt: System prompt text
        """
        self.system_prompt = prompt

    def apply_enabled_tools(self, tools: Set[str]) -> None:
        """Apply enabled tools filter.

        Args:
            tools: Set of tool names to enable
        """
        self.enabled_tools = tools

    def add_prompt_section(self, section: str) -> None:
        """Add a prompt section.

        Args:
            section: Prompt section text
        """
        self.prompt_sections.append(section)

    def apply_workflows(self, workflows: Dict[str, Any]) -> None:
        """Apply workflow definitions.

        Args:
            workflows: Dict mapping workflow names to definitions
        """
        self.workflows = workflows

    def apply_rl_config(self, config: Any) -> None:
        """Apply RL configuration.

        Args:
            config: RL configuration object
        """
        self.rl_config = config

    def apply_rl_hooks(self, hooks: Any) -> None:
        """Apply RL hooks.

        Args:
            hooks: RL hooks object
        """
        self.rl_hooks = hooks

    def apply_team_specs(self, specs: Dict[str, Any]) -> None:
        """Apply team specifications.

        Args:
            specs: Dict mapping team names to specifications
        """
        self.team_specs = specs

    def apply_tiered_config(self, config: "TieredToolConfig") -> None:
        """Apply tiered tool configuration.

        The tiered config defines mandatory, vertical_core, and semantic_pool
        tool sets for intelligent tool filtering by ToolAccessController.

        Args:
            config: TieredToolConfig from the active vertical
        """
        self.tiered_config = config

    def apply_enrichment_strategy(self, strategy: Any) -> None:
        """Apply enrichment strategy for prompt optimization.

        The strategy enables auto prompt optimization by enriching prompts
        with vertical-specific context (e.g., code symbols, web citations).

        Args:
            strategy: EnrichmentStrategyProtocol implementation from the vertical
        """
        self.enrichment_strategy = strategy

    def apply_tool_selection_strategy(self, strategy: Any) -> None:
        """Apply tool selection strategy for vertical-specific tool prioritization.

        The strategy enables verticals to customize tool selection based on
        domain knowledge, task types, and context.

        Args:
            strategy: ToolSelectionStrategyProtocol implementation from the vertical
        """
        self.tool_selection_strategy = strategy

    def set_capability_config(self, name: str, config: Any) -> None:
        """Store a capability configuration.

        Replaces direct orchestrator attribute assignment pattern:
        - OLD: orchestrator.rag_config = {...}
        - NEW: context.set_capability_config("rag_config", {...})

        Args:
            name: Config name (e.g., "rag_config", "code_style")
            config: Configuration value
        """
        self.capability_configs[name] = config

    def get_capability_config(self, name: str, default: Any = None) -> Any:
        """Retrieve a capability configuration.

        Args:
            name: Config name
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        return self.capability_configs.get(name, default)

    def apply_capability_configs(self, configs: Dict[str, Any]) -> None:
        """Apply multiple capability configurations at once.

        Args:
            configs: Dict mapping config names to configuration values
        """
        self.capability_configs.update(configs)

    # ==========================================================================
    # Query Methods
    # ==========================================================================

    def get_mode_config(self, mode_name: str) -> Optional[Any]:
        """Get mode configuration by name.

        Args:
            mode_name: Mode name to look up

        Returns:
            ModeConfig or None if not found
        """
        return self.mode_configs.get(mode_name)

    def get_tool_budget_for_mode(self, mode_name: Optional[str] = None) -> int:
        """Get tool budget for a mode.

        Args:
            mode_name: Mode name (uses default if None)

        Returns:
            Tool budget for the mode
        """
        if mode_name is None:
            mode_name = self.default_mode

        config = self.mode_configs.get(mode_name)
        if config and hasattr(config, "tool_budget"):
            return config.tool_budget

        return self.default_budget

    def get_stage_tools(self, stage: str) -> Set[str]:
        """Get tools available for a stage.

        Args:
            stage: Stage name

        Returns:
            Set of tool names for the stage
        """
        stage_def = self.stages.get(stage)
        if stage_def and hasattr(stage_def, "tools"):
            return set(stage_def.tools)
        return set()

    def get_task_hint(self, task_type: str) -> Optional[Any]:
        """Get task hint by type.

        Args:
            task_type: Task type to look up

        Returns:
            TaskTypeHint or None if not found
        """
        return self.task_hints.get(task_type)

    def get_workflow(self, name: str) -> Optional[Any]:
        """Get workflow by name.

        Args:
            name: Workflow name to look up

        Returns:
            Workflow definition or None if not found
        """
        return self.workflows.get(name)

    def get_team_spec(self, name: str) -> Optional[Any]:
        """Get team specification by name.

        Args:
            name: Team spec name to look up

        Returns:
            Team specification or None if not found
        """
        return self.team_specs.get(name)

    def list_workflows(self) -> List[str]:
        """List available workflow names.

        Returns:
            List of workflow names
        """
        return list(self.workflows.keys())

    def list_team_specs(self) -> List[str]:
        """List available team spec names.

        Returns:
            List of team spec names
        """
        return list(self.team_specs.keys())

    def get_full_system_prompt(self) -> str:
        """Get the full system prompt including sections.

        Returns:
            Combined system prompt text
        """
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        parts.extend(self.prompt_sections)
        return "\n\n".join(parts)

    @staticmethod
    def get_vertical_tiered_config() -> Any:
        """Get TieredToolConfig from active vertical canonical API.

        Returns:
            TieredToolConfig or None
        """
        try:
            from victor.core.verticals.vertical_loader import get_vertical_loader

            loader = get_vertical_loader()
            if loader.active_vertical:
                getter = getattr(loader.active_vertical, "get_tiered_tool_config", None)
                if callable(getter):
                    return getter()
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(f"Could not get tiered config from vertical: {e}")
        return None

    # ==========================================================================
    # Serialization
    # ==========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "has_config": self.config is not None,
            "stages": list(self.stages.keys()),
            "middleware_count": len(self.middleware),
            "safety_pattern_count": len(self.safety_patterns),
            "task_hints": list(self.task_hints.keys()),
            "mode_configs": list(self.mode_configs.keys()),
            "default_mode": self.default_mode,
            "default_budget": self.default_budget,
            "enabled_tools": list(self.enabled_tools),
            "has_system_prompt": self.has_custom_prompt,
            # New framework integrations
            "workflows": list(self.workflows.keys()),
            "has_rl_config": self.rl_config is not None,
            "has_rl_hooks": self.rl_hooks is not None,
            "team_specs": list(self.team_specs.keys()),
            # Tiered tool config
            "has_tiered_config": self.tiered_config is not None,
            # Enrichment strategy
            "has_enrichment_strategy": self.enrichment_strategy is not None,
            # Tool selection strategy
            "has_tool_selection_strategy": self.tool_selection_strategy is not None,
        }

    @classmethod
    def empty(cls) -> "VerticalContext":
        """Create an empty vertical context.

        Returns:
            Empty VerticalContext instance
        """
        return cls()

    def create_child_context(
        self,
        enabled_tools: Optional[Set[str]] = None,
    ) -> "VerticalContext":
        """Create a child context that inherits shared config from this parent.

        Copies immutable vertical config (name, config, system_prompt, stages,
        middleware, safety_patterns, rl_config, tiered_config, etc.) to the
        child. The child gets its own mutable state (enabled_tools,
        capability_configs, prompt_sections).

        This is the flyweight pattern: shared immutable config, per-agent
        mutable state. Used by multi-agent teams to avoid redundant copies.

        Args:
            enabled_tools: Optional tool set override for the child agent.
                If None, inherits parent's enabled_tools.

        Returns:
            New VerticalContext with inherited config.
        """
        child = VerticalContext()

        # Immutable identity (shared by reference)
        child.name = self.name
        child.config = self.config

        # Config set once at integration time (shared by reference)
        child.system_prompt = self.system_prompt
        child.default_mode = self.default_mode
        child.default_budget = self.default_budget
        child.rl_config = self.rl_config
        child.rl_hooks = self.rl_hooks
        child.tiered_config = self.tiered_config
        child.enrichment_strategy = self.enrichment_strategy
        child.tool_selection_strategy = self.tool_selection_strategy

        # Copy collections (shallow copy — child gets own list/dict/set)
        child.stages = dict(self.stages)
        child.middleware = list(self.middleware)
        child.safety_patterns = list(self.safety_patterns)
        child.task_hints = dict(self.task_hints)
        child.mode_configs = dict(self.mode_configs)
        child.tool_dependencies = list(self.tool_dependencies)
        child.tool_sequences = [list(s) for s in self.tool_sequences]
        child.workflows = dict(self.workflows)
        child.team_specs = dict(self.team_specs)

        # Per-agent mutable state (fresh or overridden)
        child.enabled_tools = set(enabled_tools) if enabled_tools else set(self.enabled_tools)
        child.prompt_sections = []  # Fresh — child builds its own
        child.capability_configs = {}  # Fresh — child gets own capabilities

        return child


# =============================================================================
# Factory Functions
# =============================================================================


def create_vertical_context(
    name: Optional[str] = None,
    config: Optional["VerticalConfig"] = None,
) -> VerticalContext:
    """Create a vertical context.

    Args:
        name: Optional vertical name
        config: Optional vertical configuration

    Returns:
        Configured VerticalContext
    """
    context = VerticalContext()
    if name:
        context.apply_vertical(name, config)
    return context


__all__ = [
    "VerticalContext",
    "VerticalContextProtocol",
    "MutableVerticalContextProtocol",
    "create_vertical_context",
]
