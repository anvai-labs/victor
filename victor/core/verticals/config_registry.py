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

"""Centralized registry for vertical configuration templates.

Provides pre-built configurations for common vertical patterns,
eliminating duplication across vertical assistants.

This module now also manages dynamic vertical configuration, replacing
hardcoded dictionaries like _VERTICAL_CANONICALIZE_SETTINGS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor_contracts.verticals.manifest import ExtensionManifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerticalBehaviorConfig:
    """Configuration for a specific vertical's behavior.

    This dataclass encapsulates all configuration options that control
    vertical behavior, replacing hardcoded dictionaries scattered across
    the codebase.

    Attributes:
        canonicalize_tool_names: Whether to normalize tool names (default: True)
        tool_dependency_strategy: How to load tool dependencies
        strict_mode: If True, all extension load failures raise exceptions
        load_priority: Higher values load first in dependency resolution
        lazy_load: If True, extensions loaded on first access
    """

    canonicalize_tool_names: bool = True
    tool_dependency_strategy: str = "auto"
    strict_mode: bool = False
    load_priority: int = 0
    lazy_load: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Validate tool_dependency_strategy
        valid_strategies = {"auto", "entry_point", "factory", "none"}
        if self.tool_dependency_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid tool_dependency_strategy: {self.tool_dependency_strategy}. "
                f"Must be one of {valid_strategies}"
            )

        # Validate load_priority is non-negative
        if self.load_priority < 0:
            raise ValueError(f"load_priority must be non-negative, got {self.load_priority}")


class VerticalConfigRegistry:
    """Registry of pre-built vertical configuration templates.

    Implements Registry pattern for OCP compliance - new configurations
    can be registered without modifying existing code.

    Example:
        # Get pre-built coding provider hints
        hints = VerticalConfigRegistry.get_provider_hints("coding")

        # Register custom configuration
        VerticalConfigRegistry.register_provider_hints("my_vertical", {...})
    """

    # Provider hints — only the "default" fallback lives in core.
    # Vertical-specific hints are registered dynamically by external vertical
    # packages via register_vertical_config() during VictorPlugin.register().
    _provider_hints: Dict[str, Dict[str, Any]] = {
        "default": {
            "preferred_providers": ["anthropic", "openai"],
            "min_context_window": 100000,
            "requires_tool_calling": True,
        },
    }

    # Evaluation criteria — only the "default" fallback lives in core.
    # Vertical-specific criteria are registered by external packages.
    _evaluation_criteria: Dict[str, List[str]] = {
        "default": [
            "Task completion accuracy",
            "Tool usage efficiency",
            "Response relevance",
            "Error handling",
        ],
    }

    # Dynamic registry for vertical-provided configs (OCP extension point)
    _registered_provider_hints: Dict[str, Dict[str, Any]] = {}
    _registered_eval_criteria: Dict[str, List[str]] = {}

    @classmethod
    def register_vertical_config(
        cls,
        vertical: str,
        provider_hints: Optional[Dict[str, Any]] = None,
        evaluation_criteria: Optional[List[str]] = None,
    ) -> None:
        """Register configuration for a vertical dynamically.

        Enables new verticals to provide their own provider hints and
        evaluation criteria without modifying core code.

        Args:
            vertical: Vertical name
            provider_hints: LLM provider selection hints
            evaluation_criteria: Quality evaluation criteria
        """
        if provider_hints is not None:
            cls._registered_provider_hints[vertical] = provider_hints
        if evaluation_criteria is not None:
            cls._registered_eval_criteria[vertical] = evaluation_criteria

    @classmethod
    def get_provider_hints(cls, vertical_name: str) -> Dict[str, Any]:
        """Get provider hints for a vertical.

        Checks dynamic registry first, then falls back to built-in defaults.

        Args:
            vertical_name: Name of vertical (e.g., "coding", "research")

        Returns:
            Provider hints dictionary (copy to prevent mutation)
        """
        # Check dynamic registry first
        if vertical_name in cls._registered_provider_hints:
            return cls._registered_provider_hints[vertical_name].copy()
        # Fall back to built-in defaults
        if vertical_name not in cls._provider_hints:
            return cls._provider_hints["default"].copy()
        return cls._provider_hints[vertical_name].copy()

    @classmethod
    def get_evaluation_criteria(cls, vertical_name: str) -> List[str]:
        """Get evaluation criteria for a vertical.

        Checks dynamic registry first, then falls back to built-in defaults.

        Args:
            vertical_name: Name of vertical (e.g., "coding", "research")

        Returns:
            List of evaluation criteria (copy to prevent mutation)
        """
        # Check dynamic registry first
        if vertical_name in cls._registered_eval_criteria:
            return cls._registered_eval_criteria[vertical_name].copy()
        # Fall back to built-in defaults
        if vertical_name not in cls._evaluation_criteria:
            return cls._evaluation_criteria["default"].copy()
        return cls._evaluation_criteria[vertical_name].copy()

    @classmethod
    def register_provider_hints(cls, key: str, hints: Dict[str, Any]) -> None:
        """Register custom provider hints (for extensibility).

        Args:
            key: Unique identifier for this configuration
            hints: Provider hints dictionary
        """
        cls._provider_hints[key] = hints

    @classmethod
    def register_evaluation_criteria(cls, key: str, criteria: List[str]) -> None:
        """Register custom evaluation criteria (for extensibility).

        Args:
            key: Unique identifier for this configuration
            criteria: List of evaluation criteria
        """
        cls._evaluation_criteria[key] = criteria


# Vertical Behavior Configuration Registry
# Replaces hardcoded _VERTICAL_CANONICALIZE_SETTINGS and similar dicts


class VerticalBehaviorConfigRegistry:
    """Registry for vertical behavior configurations.

    This singleton registry manages behavior configuration for all verticals,
    providing a centralized location for dynamic configuration while
    maintaining backward compatibility.

    Usage:
        # Register configuration for a vertical
        config = VerticalBehaviorConfig(canonicalize_tool_names=False)
        VerticalBehaviorConfigRegistry.register("devops", config)

        # Get configuration for a vertical
        config = VerticalBehaviorConfigRegistry.get("devops")
        if config.canonicalize_tool_names:
            # Use canonicalized names
    """

    _configs: Dict[str, VerticalBehaviorConfig] = {}
    _default_config: VerticalBehaviorConfig = VerticalBehaviorConfig()

    @classmethod
    def register(cls, name: str, config: VerticalBehaviorConfig) -> None:
        """Register configuration for a vertical.

        Args:
            name: Vertical name (e.g., "coding", "devops", "research")
            config: VerticalBehaviorConfig instance

        Raises:
            ValueError: If name is empty or config is invalid
        """
        if not name:
            raise ValueError("Vertical name cannot be empty")

        if not isinstance(config, VerticalBehaviorConfig):
            raise ValueError(f"Config must be VerticalBehaviorConfig instance, got {type(config)}")

        cls._configs[name] = config
        logger.debug(f"Registered behavior configuration for vertical '{name}'")

    @classmethod
    def get(cls, name: str) -> VerticalBehaviorConfig:
        """Get configuration for a vertical, or defaults.

        Args:
            name: Vertical name

        Returns:
            VerticalBehaviorConfig instance (registered config or defaults)
        """
        return cls._configs.get(name, cls._default_config)

    @classmethod
    def has_config(cls, name: str) -> bool:
        """Check if a vertical has explicit behavior configuration registered.

        Args:
            name: Vertical name

        Returns:
            True if explicit config exists, False otherwise
        """
        return name in cls._configs

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove configuration for a vertical.

        Args:
            name: Vertical name to unregister

        Note:
            This is primarily useful for testing
        """
        if name in cls._configs:
            del cls._configs[name]
            logger.debug(f"Unregistered behavior configuration for vertical '{name}'")

    @classmethod
    def clear(cls) -> None:
        """Clear all registered behavior configurations.

        Note:
            This is primarily useful for testing
        """
        cls._configs.clear()
        logger.debug("Cleared all vertical behavior configurations")

    @classmethod
    def from_manifest(cls, manifest: "ExtensionManifest") -> VerticalBehaviorConfig:
        """Create configuration from an ExtensionManifest.

        Extracts configuration fields from a manifest and creates a
        VerticalBehaviorConfig instance. This is the primary way verticals
        should declare their behavior configuration.

        Args:
            manifest: ExtensionManifest instance

        Returns:
            VerticalBehaviorConfig instance

        Examples:
            >>> manifest = ExtensionManifest(
            ...     name="devops",
            ...     canonicalize_tool_names=False,
            ...     tool_dependency_strategy="entry_point",
            ... )
            >>> config = VerticalBehaviorConfigRegistry.from_manifest(manifest)
        """
        return VerticalBehaviorConfig(
            canonicalize_tool_names=manifest.canonicalize_tool_names,
            tool_dependency_strategy=manifest.tool_dependency_strategy,
            strict_mode=manifest.strict_mode,
            load_priority=manifest.load_priority,
            lazy_load=manifest.lazy_load,
        )

    @classmethod
    def get_or_create_from_manifest(
        cls, name: str, manifest: Optional["ExtensionManifest"]
    ) -> VerticalBehaviorConfig:
        """Get existing config or create from manifest.

        This is a convenience method that checks if explicit configuration
        exists for a vertical, and if not, creates it from the manifest.

        Args:
            name: Vertical name
            manifest: ExtensionManifest instance (if available)

        Returns:
            VerticalBehaviorConfig instance
        """
        # Check if explicit config exists
        if cls.has_config(name):
            return cls.get(name)

        # Create from manifest if available
        if manifest is not None:
            return cls.from_manifest(manifest)

        # Return defaults
        return cls._default_config

    @classmethod
    def list_configured_verticals(cls) -> list[str]:
        """List all verticals with explicit behavior configuration.

        Returns:
            List of vertical names that have explicit config
        """
        return list(cls._configs.keys())


# Convenience functions for common operations


def get_canonicalization_setting(vertical_name: str) -> bool:
    """Get tool name canonicalization setting for a vertical.

    This is a convenience function that replaces the old
    _VERTICAL_CANONICALIZE_SETTINGS dictionary lookup.

    Args:
        vertical_name: Name of the vertical

    Returns:
        True if tool names should be canonicalized, False otherwise

    Examples:
        >>> get_canonicalization_setting("coding")
        True
        >>> get_canonicalization_setting("devops")
        False
    """
    config = VerticalBehaviorConfigRegistry.get(vertical_name)
    return config.canonicalize_tool_names


def get_tool_dependency_strategy(vertical_name: str) -> str:
    """Get tool dependency strategy for a vertical.

    Args:
        vertical_name: Name of the vertical

    Returns:
        Strategy string: "auto", "entry_point", "factory", or "none"
    """
    config = VerticalBehaviorConfigRegistry.get(vertical_name)
    return config.tool_dependency_strategy


def is_strict_mode_enabled(vertical_name: str) -> bool:
    """Check if strict mode is enabled for a vertical.

    Args:
        vertical_name: Name of the vertical

    Returns:
        True if strict mode is enabled, False otherwise
    """
    config = VerticalBehaviorConfigRegistry.get(vertical_name)
    return config.strict_mode
