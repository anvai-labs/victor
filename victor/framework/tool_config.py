"""Tool Configuration Externalization Module.

This module provides a clean, externalized approach to tool configuration
that separates tool setup from orchestrator internals.

Design Patterns:
- Strategy Pattern: Pluggable tool configuration strategies
- Builder Pattern: Fluent tool configuration
- Repository Pattern: Tool catalog access

Phase 7.5: Externalize tool configuration from _internal.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Union,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator
    from victor.framework.tools import ToolSet

# Import canonical ToolCategory and registry from tools.py (single source of truth)
from victor.framework.tools import ToolCategory, get_category_registry

# Import capability helpers for protocol-based access
from victor.framework.capability_runtime import (
    check_capability as _check_capability,
    invoke_capability as _invoke_capability,
)

# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class ToolConfiguratorProtocol(Protocol):
    """Protocol for tool configuration strategies."""

    def configure(
        self,
        orchestrator: "AgentOrchestrator",
        tools: Union["ToolSet", List[str]],
    ) -> None:
        """Apply tool configuration to orchestrator."""
        ...

    def get_enabled_tools(self) -> Set[str]:
        """Get set of currently enabled tool names."""
        ...


@runtime_checkable
class ToolFilterProtocol(Protocol):
    """Protocol for tool filtering strategies."""

    def filter(
        self,
        available_tools: Set[str],
        context: Dict[str, Any],
    ) -> Set[str]:
        """Filter available tools based on context."""
        ...


# =============================================================================
# Configuration Enums
# =============================================================================


class ToolConfigMode(str, Enum):
    """Mode for tool configuration."""

    REPLACE = "replace"  # Replace all tools with specified set
    EXTEND = "extend"  # Add to existing tools
    RESTRICT = "restrict"  # Only allow specified tools from existing
    FILTER = "filter"  # Apply filter to existing tools


# Note: ToolCategory is imported from tools.py (single source of truth)
# Do not define ToolCategory here - use the import above


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ToolConfigEntry:
    """Configuration entry for a single tool."""

    name: str
    enabled: bool = True
    category: ToolCategory = ToolCategory.CUSTOM
    priority: int = 0  # Higher = more likely to be selected
    cost_tier: str = "low"
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "category": self.category.value,
            "priority": self.priority,
            "cost_tier": self.cost_tier,
            "constraints": self.constraints,
            "metadata": self.metadata,
        }


@dataclass
class ToolConfigResult:
    """Result of tool configuration operation."""

    success: bool
    enabled_tools: Set[str]
    disabled_tools: Set[str]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Tool Configurator
# =============================================================================


class ToolConfigurator:
    """Configurator for applying tool settings to an orchestrator.

    This class provides a clean interface for configuring tools on an
    AgentOrchestrator, separating tool setup logic from the orchestrator
    implementation.

    Example:
        configurator = ToolConfigurator()

        # Simple configuration
        result = configurator.configure_from_toolset(
            orchestrator,
            ToolSet.default(),
        )

        # With mode
        result = configurator.configure(
            orchestrator,
            tools=["read", "write", "edit"],
            mode=ToolConfigMode.RESTRICT,
        )

        # Advanced configuration
        config = (
            ToolConfigBuilder()
            .enable_category(ToolCategory.CORE)
            .enable_category(ToolCategory.GIT)
            .disable_tools(["shell"])
            .build()
        )
        result = configurator.apply_config(orchestrator, config)
    """

    def __init__(self) -> None:
        """Initialize configurator."""
        self._tool_catalog: Dict[str, ToolConfigEntry] = {}
        self._filters: List[ToolFilterProtocol] = []
        self._hooks: Dict[str, List[Callable]] = {
            "pre_configure": [],
            "post_configure": [],
        }

    # -------------------------------------------------------------------------
    # Configuration Methods
    # -------------------------------------------------------------------------

    def configure_from_toolset(
        self,
        orchestrator: "AgentOrchestrator",
        toolset: "ToolSet",
    ) -> ToolConfigResult:
        """Configure orchestrator from a ToolSet.

        Args:
            orchestrator: Target orchestrator
            toolset: ToolSet configuration

        Returns:
            Configuration result
        """

        enabled_tools = set(toolset.get_tool_names())
        return self._apply_tools(orchestrator, enabled_tools, ToolConfigMode.REPLACE)

    def configure(
        self,
        orchestrator: "AgentOrchestrator",
        tools: Union["ToolSet", List[str], Set[str]],
        mode: ToolConfigMode = ToolConfigMode.REPLACE,
    ) -> ToolConfigResult:
        """Configure tools on orchestrator.

        Args:
            orchestrator: Target orchestrator
            tools: Tools to configure (ToolSet, list, or set)
            mode: Configuration mode

        Returns:
            Configuration result
        """
        from victor.framework.tools import ToolSet

        # Convert to set of tool names
        if isinstance(tools, ToolSet):
            enabled_tools = set(tools.get_tool_names())
        elif isinstance(tools, (list, set)):
            enabled_tools = set(tools)
        else:
            enabled_tools = set()

        return self._apply_tools(orchestrator, enabled_tools, mode)

    def apply_config(
        self,
        orchestrator: "AgentOrchestrator",
        config: "ToolConfig",
    ) -> ToolConfigResult:
        """Apply a ToolConfig to orchestrator.

        Args:
            orchestrator: Target orchestrator
            config: Tool configuration

        Returns:
            Configuration result
        """
        return self._apply_tools(
            orchestrator,
            config.enabled_tools,
            config.mode,
            excluded=config.disabled_tools,
        )

    def _apply_tools(
        self,
        orchestrator: "AgentOrchestrator",
        enabled_tools: Set[str],
        mode: ToolConfigMode,
        excluded: Optional[Set[str]] = None,
    ) -> ToolConfigResult:
        """Internal method to apply tool configuration.

        Args:
            orchestrator: Target orchestrator
            enabled_tools: Tools to enable
            mode: Configuration mode
            excluded: Tools to exclude

        Returns:
            Configuration result
        """
        excluded = excluded or set()
        errors: List[str] = []
        warnings: List[str] = []

        # Run pre-configure hooks
        for hook in self._hooks.get("pre_configure", []):
            try:
                hook(orchestrator, enabled_tools, mode)
            except Exception as e:
                warnings.append(f"Pre-configure hook failed: {e}")

        # Get available tools from orchestrator
        available_tools = self._get_available_tools(orchestrator)

        # Apply mode
        if mode == ToolConfigMode.REPLACE:
            final_tools = enabled_tools - excluded
        elif mode == ToolConfigMode.EXTEND:
            current = self._get_current_tools(orchestrator)
            final_tools = (current | enabled_tools) - excluded
        elif mode == ToolConfigMode.RESTRICT:
            current = self._get_current_tools(orchestrator)
            final_tools = (current & enabled_tools) - excluded
        elif mode == ToolConfigMode.FILTER:
            # Apply filters
            final_tools = enabled_tools
            for tool_filter in self._filters:
                try:
                    final_tools = tool_filter.filter(
                        final_tools,
                        {"orchestrator": orchestrator},
                    )
                except Exception as e:
                    warnings.append(f"Filter failed: {e}")
            final_tools = final_tools - excluded
        else:
            final_tools = enabled_tools - excluded

        # Validate tools exist
        invalid_tools = final_tools - available_tools
        if invalid_tools:
            warnings.append(f"Unknown tools ignored: {invalid_tools}")
            final_tools = final_tools & available_tools

        # Apply to orchestrator
        self._set_orchestrator_tools(orchestrator, final_tools)

        # Calculate disabled tools
        disabled_tools = available_tools - final_tools

        # Run post-configure hooks
        for hook in self._hooks.get("post_configure", []):
            try:
                hook(orchestrator, final_tools, disabled_tools)
            except Exception as e:
                warnings.append(f"Post-configure hook failed: {e}")

        return ToolConfigResult(
            success=len(errors) == 0,
            enabled_tools=final_tools,
            disabled_tools=disabled_tools,
            errors=errors,
            warnings=warnings,
        )

    # -------------------------------------------------------------------------
    # Orchestrator Interaction
    # -------------------------------------------------------------------------

    def _get_available_tools(self, orchestrator: "AgentOrchestrator") -> Set[str]:
        """Get all available tools from orchestrator.

        Args:
            orchestrator: Target orchestrator

        Returns:
            Set of available tool names
        """
        # Check for tools attribute
        tools = getattr(orchestrator, "tools", None)
        if tools is None:
            return set()

        # Handle different tool container types:
        # 1. ToolRegistry (has list_all() or keys() method)
        # 2. dict (has keys() method)
        # 3. list/set of tool names
        if hasattr(tools, "list_all"):
            # ToolRegistry or similar registry - preferred method
            return set(tools.list_all())
        elif hasattr(tools, "keys"):
            # dict-like or registry with keys()
            return set(tools.keys())
        elif isinstance(tools, (list, set, frozenset)):
            # Direct collection of tool names
            return set(tools)

        return set()

    def _get_current_tools(self, orchestrator: "AgentOrchestrator") -> Set[str]:
        """Get currently enabled tools.

        Args:
            orchestrator: Target orchestrator

        Returns:
            Set of enabled tool names
        """
        # Use capability-based check first (protocol-first approach)
        if hasattr(orchestrator, "get_enabled_tools") and callable(orchestrator.get_enabled_tools):
            return orchestrator.get_enabled_tools()

        # Fall back to all available
        return self._get_available_tools(orchestrator)

    def _set_orchestrator_tools(
        self,
        orchestrator: "AgentOrchestrator",
        tools: Set[str],
    ) -> None:
        """Set enabled tools on orchestrator.

        Args:
            orchestrator: Target orchestrator
            tools: Tools to enable
        """
        # Use capability-based approach (protocol-first, fallback to hasattr)
        if _check_capability(orchestrator, "enabled_tools"):
            _invoke_capability(orchestrator, "enabled_tools", tools)
        elif hasattr(orchestrator, "set_enabled_tools") and callable(
            orchestrator.set_enabled_tools
        ):
            orchestrator.set_enabled_tools(tools)
        else:
            # Fallback: log warning as this indicates protocol non-compliance
            import logging

            logging.getLogger(__name__).warning(
                "Orchestrator does not implement enabled_tools capability; "
                "tool configuration may not be applied properly"
            )

    # -------------------------------------------------------------------------
    # Hooks and Filters
    # -------------------------------------------------------------------------

    def add_filter(self, tool_filter: ToolFilterProtocol) -> None:
        """Add a tool filter.

        Args:
            tool_filter: Filter to add
        """
        self._filters.append(tool_filter)

    def remove_filter(self, tool_filter: ToolFilterProtocol) -> bool:
        """Remove a tool filter.

        Args:
            tool_filter: Filter to remove

        Returns:
            True if filter was removed, False if not found
        """
        if tool_filter in self._filters:
            self._filters.remove(tool_filter)
            return True
        return False

    def add_hook(
        self,
        event: str,
        callback: Callable,
    ) -> Callable[[], None]:
        """Add a configuration hook.

        Args:
            event: Hook event ("pre_configure" or "post_configure")
            callback: Callback function

        Returns:
            Unsubscribe function
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

        def unsubscribe() -> None:
            if callback in self._hooks.get(event, []):
                self._hooks[event].remove(callback)

        return unsubscribe


# =============================================================================
# Tool Config
# =============================================================================


@dataclass
class ToolConfig:
    """Immutable tool configuration."""

    enabled_tools: Set[str] = field(default_factory=set)
    disabled_tools: Set[str] = field(default_factory=set)
    mode: ToolConfigMode = ToolConfigMode.REPLACE
    categories: Set[ToolCategory] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_enabled(self, *tools: str) -> "ToolConfig":
        """Create new config with additional enabled tools."""
        return ToolConfig(
            enabled_tools=self.enabled_tools | set(tools),
            disabled_tools=self.disabled_tools,
            mode=self.mode,
            categories=self.categories,
            metadata=dict(self.metadata),
        )

    def with_disabled(self, *tools: str) -> "ToolConfig":
        """Create new config with additional disabled tools."""
        return ToolConfig(
            enabled_tools=self.enabled_tools,
            disabled_tools=self.disabled_tools | set(tools),
            mode=self.mode,
            categories=self.categories,
            metadata=dict(self.metadata),
        )


# =============================================================================
# Tool Config Builder
# =============================================================================


class ToolConfigBuilder:
    """Builder for creating ToolConfig instances.

    Example:
        config = (
            ToolConfigBuilder()
            .mode(ToolConfigMode.RESTRICT)
            .enable_category(ToolCategory.CORE)
            .enable_category(ToolCategory.GIT)
            .enable_tools("custom_tool", "another_tool")
            .disable_tools("shell")
            .build()
        )
    """

    @staticmethod
    def get_category_tools(category: ToolCategory) -> Set[str]:
        """Get tools for a category using the registry.

        This replaces static CATEGORY_TOOLS with dynamic registry lookup.
        Supports custom categories registered by plugins/verticals.

        Args:
            category: Tool category to look up

        Returns:
            Set of tool names in the category
        """
        return get_category_registry().get_tools(category.value)

    def __init__(self) -> None:
        """Initialize builder."""
        self._enabled: Set[str] = set()
        self._disabled: Set[str] = set()
        self._mode: ToolConfigMode = ToolConfigMode.REPLACE
        self._categories: Set[ToolCategory] = set()
        self._metadata: Dict[str, Any] = {}

    def mode(self, mode: ToolConfigMode) -> "ToolConfigBuilder":
        """Set configuration mode.

        Args:
            mode: Configuration mode

        Returns:
            Self for chaining
        """
        self._mode = mode
        return self

    def enable_tools(self, *tools: str) -> "ToolConfigBuilder":
        """Enable specific tools.

        Args:
            *tools: Tool names to enable

        Returns:
            Self for chaining
        """
        self._enabled.update(tools)
        return self

    def disable_tools(self, *tools: str) -> "ToolConfigBuilder":
        """Disable specific tools.

        Args:
            *tools: Tool names to disable

        Returns:
            Self for chaining
        """
        self._disabled.update(tools)
        return self

    def enable_category(self, category: ToolCategory) -> "ToolConfigBuilder":
        """Enable all tools in a category.

        Args:
            category: Tool category

        Returns:
            Self for chaining
        """
        self._categories.add(category)
        category_tools = self.get_category_tools(category)
        self._enabled.update(category_tools)
        return self

    def disable_category(self, category: ToolCategory) -> "ToolConfigBuilder":
        """Disable all tools in a category.

        Args:
            category: Tool category

        Returns:
            Self for chaining
        """
        category_tools = self.get_category_tools(category)
        self._disabled.update(category_tools)
        return self

    def metadata(self, key: str, value: Any) -> "ToolConfigBuilder":
        """Add metadata.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            Self for chaining
        """
        self._metadata[key] = value
        return self

    def build(self) -> ToolConfig:
        """Build the ToolConfig.

        Returns:
            Configured ToolConfig instance
        """
        return ToolConfig(
            enabled_tools=self._enabled.copy(),
            disabled_tools=self._disabled.copy(),
            mode=self._mode,
            categories=self._categories.copy(),
            metadata=dict(self._metadata),
        )


# =============================================================================
# Utility Functions
# =============================================================================


def get_tool_configurator() -> ToolConfigurator:
    """Get a ToolConfigurator instance.

    Returns:
        ToolConfigurator instance
    """
    return ToolConfigurator()


def configure_tools_from_toolset(
    orchestrator: "AgentOrchestrator",
    toolset: "ToolSet",
) -> ToolConfigResult:
    """Convenience function to configure tools from a ToolSet.

    Args:
        orchestrator: Target orchestrator
        toolset: ToolSet configuration

    Returns:
        Configuration result
    """
    configurator = get_tool_configurator()
    return configurator.configure_from_toolset(orchestrator, toolset)


def configure_tools(
    orchestrator: "AgentOrchestrator",
    tools: Union["ToolSet", List[str], Set[str]],
    mode: ToolConfigMode = ToolConfigMode.REPLACE,
) -> ToolConfigResult:
    """Convenience function to configure tools.

    Args:
        orchestrator: Target orchestrator
        tools: Tools to configure
        mode: Configuration mode

    Returns:
        Configuration result
    """
    configurator = get_tool_configurator()
    return configurator.configure(orchestrator, tools, mode)


# =============================================================================
# Built-in Filters
# =============================================================================


class AirgappedFilter:
    """Filter that removes network-dependent tools."""

    NETWORK_TOOLS = {
        "web",
        "web_search",
        "web_fetch",
        "http_request",
        "slack",
        "teams",
        "jira",
    }

    def filter(
        self,
        available_tools: Set[str],
        context: Dict[str, Any],
    ) -> Set[str]:
        """Remove network tools.

        Args:
            available_tools: Available tools
            context: Filter context

        Returns:
            Filtered tools
        """
        from victor.tools.tool_names import get_canonical_name

        blocked = {get_canonical_name(name) for name in self.NETWORK_TOOLS}
        return {name for name in available_tools if get_canonical_name(name) not in blocked}


class CostTierFilter:
    """Filter that limits tools by cost tier."""

    def __init__(self, max_tier: str = "medium") -> None:
        """Initialize filter.

        Args:
            max_tier: Maximum cost tier to allow
        """
        self.max_tier = max_tier
        self.tier_order = ["free", "low", "medium", "high"]

    def filter(
        self,
        available_tools: Set[str],
        context: Dict[str, Any],
    ) -> Set[str]:
        """Filter by cost tier.

        Args:
            available_tools: Available tools
            context: Filter context (should contain tool metadata)

        Returns:
            Filtered tools
        """
        # For now, return all tools (tier info needs to be passed in context)
        # This is a placeholder for when tool metadata is available
        return available_tools


class SecurityFilter:
    """Filter that removes potentially dangerous tools."""

    DANGEROUS_TOOLS = {
        "shell",
        "bash",
        "database",
        "rm",
    }

    def __init__(self, allow_dangerous: bool = False) -> None:
        """Initialize filter.

        Args:
            allow_dangerous: Whether to allow dangerous tools
        """
        self.allow_dangerous = allow_dangerous

    def filter(
        self,
        available_tools: Set[str],
        context: Dict[str, Any],
    ) -> Set[str]:
        """Filter dangerous tools.

        Args:
            available_tools: Available tools
            context: Filter context

        Returns:
            Filtered tools
        """
        if self.allow_dangerous:
            return available_tools
        return available_tools - self.DANGEROUS_TOOLS
