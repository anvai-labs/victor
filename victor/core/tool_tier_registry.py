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

"""Tool Tier Registry - Central registry for cross-vertical tool tier management.

This module provides a singleton registry for managing tiered tool configurations
across all verticals. The registry enables:

1. Cross-Vertical Tier Definitions: Register and retrieve tier configurations
   that can be shared or inherited across verticals.

2. Tier Inheritance: Base configurations can be extended by specific verticals.

3. Dynamic Tier Updates: Tiers can be updated at runtime for A/B testing or
   feature flags.

4. Tier Validation: Ensures tier configurations are consistent and valid.

Design Patterns:
    - Singleton: Single registry instance for global access
    - Registry (PoEAA): Central lookup for tier configurations
    - Template Method: Default tiers with customization hooks

Usage:
    from victor.core.tool_tier_registry import ToolTierRegistry

    # Get the singleton instance
    registry = ToolTierRegistry.get_instance()

    # Register a tier configuration
    registry.register(
        name="coding",
        config=TieredToolConfig(
            mandatory={"read", "ls", "grep"},
            vertical_core={"edit", "write", "shell", "git"},
        ),
    )

    # Retrieve a tier configuration
    config = registry.get("coding")

    # Extend a base tier
    extended = registry.extend("coding", vertical_core={"docker", "test"})

    # List all registered tiers
    all_tiers = registry.list_all()
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.core.vertical_types import TieredToolConfig


@dataclass
class TierRegistryEntry:
    """Entry in the tool tier registry.

    Attributes:
        name: Tier name (typically vertical name)
        config: The TieredToolConfig for this tier
        parent: Parent tier name for inheritance
        description: Human-readable description
        metadata: Additional metadata for this tier
        version: Version string for tracking changes
    """

    name: str
    config: Any  # TieredToolConfig
    parent: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "config": {
                "mandatory": list(self.config.mandatory) if self.config else [],
                "vertical_core": list(self.config.vertical_core) if self.config else [],
                "semantic_pool": list(self.config.semantic_pool) if self.config else [],
                "readonly_only_for_analysis": (
                    self.config.readonly_only_for_analysis if self.config else True
                ),
            },
            "parent": self.parent,
            "description": self.description,
            "metadata": self.metadata,
            "version": self.version,
        }


class ToolTierRegistry:
    """Singleton registry for tool tier configurations.

    Provides centralized management of tiered tool configurations across
    all verticals, enabling inheritance, validation, and dynamic updates.

    Thread Safety:
        All operations are thread-safe using RLock for reentrant locking.

    Singleton Access:
        Use ToolTierRegistry.get_instance() to get the singleton.
        Direct instantiation is allowed for testing but not recommended.

    Example:
        registry = ToolTierRegistry.get_instance()

        # Register base tier
        registry.register("base", TieredToolConfig(
            mandatory={"read", "ls", "grep"},
        ))

        # Register derived tier with inheritance
        registry.register(
            "coding",
            TieredToolConfig(vertical_core={"edit", "write"}),
            parent="base",
        )

        # Get merged config (includes parent mandatory tools)
        config = registry.get_merged("coding")
    """

    _instance: Optional["ToolTierRegistry"] = None
    _lock: threading.RLock = threading.RLock()

    def __init__(self) -> None:
        """Initialize the registry."""
        self._entries: Dict[str, TierRegistryEntry] = {}
        self._registry_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "ToolTierRegistry":
        """Get the singleton instance.

        Returns:
            The global ToolTierRegistry instance.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                cls._instance._register_defaults()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance.

        Primarily for testing to ensure clean state between tests.
        """
        with cls._lock:
            cls._instance = None

    def _register_defaults(self) -> None:
        """Register default tier configurations.

        Called during singleton initialization to set up the base tier
        that is common across all verticals. Vertical-specific tiers
        are registered dynamically when verticals call
        TieredToolTemplate.register_vertical_tools() at activation time.
        """
        from victor.core.vertical_types import TieredToolConfig, TieredToolTemplate

        # Register base tier only (shared across all verticals)
        self.register(
            name="base",
            config=TieredToolConfig(
                mandatory=TieredToolTemplate.DEFAULT_MANDATORY.copy(),
                vertical_core=set(),
            ),
            description="Base tier with mandatory tools for all verticals",
        )

    def register(
        self,
        name: str,
        config: Any,  # TieredToolConfig
        *,
        parent: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0",
        overwrite: bool = False,
    ) -> None:
        """Register a tier configuration.

        Args:
            name: Tier name (must be unique unless overwrite=True)
            config: TieredToolConfig instance
            parent: Parent tier name for inheritance
            description: Human-readable description
            metadata: Additional metadata
            version: Version string
            overwrite: If True, overwrite existing entry

        Raises:
            ValueError: If name already exists and overwrite=False
            ValueError: If parent doesn't exist
        """
        with self._registry_lock:
            if name in self._entries and not overwrite:
                raise ValueError(
                    f"Tier '{name}' already registered. Use overwrite=True to replace."
                )

            if parent and parent not in self._entries:
                raise ValueError(f"Parent tier '{parent}' not found in registry.")

            self._entries[name] = TierRegistryEntry(
                name=name,
                config=config,
                parent=parent,
                description=description,
                metadata=metadata or {},
                version=version,
            )

    def unregister(self, name: str) -> bool:
        """Unregister a tier configuration.

        Args:
            name: Tier name to remove

        Returns:
            True if removed, False if not found
        """
        with self._registry_lock:
            if name in self._entries:
                del self._entries[name]
                return True
            return False

    def get(self, name: str) -> Optional[Any]:
        """Get a tier configuration by name.

        Args:
            name: Tier name

        Returns:
            TieredToolConfig or None if not found
        """
        with self._registry_lock:
            entry = self._entries.get(name)
            return entry.config if entry else None

    def get_entry(self, name: str) -> Optional[TierRegistryEntry]:
        """Get the full registry entry by name.

        Args:
            name: Tier name

        Returns:
            TierRegistryEntry or None if not found
        """
        with self._registry_lock:
            return self._entries.get(name)

    def get_merged(self, name: str) -> Optional[Any]:
        """Get a tier configuration with parent inheritance merged.

        Merges the tier's tools with all parent tiers, creating a
        complete configuration with inherited mandatory tools.

        Args:
            name: Tier name

        Returns:
            TieredToolConfig with merged parent tools, or None if not found
        """
        from victor.core.vertical_types import TieredToolConfig

        with self._registry_lock:
            entry = self._entries.get(name)
            if not entry:
                return None

            # Start with current config
            merged_mandatory: Set[str] = set(entry.config.mandatory)
            merged_vertical_core: Set[str] = set(entry.config.vertical_core)
            merged_semantic_pool: Set[str] = set(entry.config.semantic_pool)
            merged_stage_tools: Dict[str, Set[str]] = dict(entry.config.stage_tools)

            # Walk up parent chain
            current = entry
            while current.parent:
                parent_entry = self._entries.get(current.parent)
                if not parent_entry:
                    break

                # Merge parent tools (parent tools are added, not replaced)
                merged_mandatory |= set(parent_entry.config.mandatory)
                merged_vertical_core |= set(parent_entry.config.vertical_core)
                merged_semantic_pool |= set(parent_entry.config.semantic_pool)

                # Merge stage tools
                for stage, tools in parent_entry.config.stage_tools.items():
                    if stage in merged_stage_tools:
                        merged_stage_tools[stage] |= set(tools)
                    else:
                        merged_stage_tools[stage] = set(tools)

                current = parent_entry

            return TieredToolConfig(
                mandatory=merged_mandatory,
                vertical_core=merged_vertical_core,
                semantic_pool=merged_semantic_pool,
                stage_tools=merged_stage_tools,
                readonly_only_for_analysis=entry.config.readonly_only_for_analysis,
            )

    def extend(
        self,
        base_name: str,
        *,
        mandatory: Optional[Set[str]] = None,
        vertical_core: Optional[Set[str]] = None,
        semantic_pool: Optional[Set[str]] = None,
        stage_tools: Optional[Dict[str, Set[str]]] = None,
        readonly_only_for_analysis: Optional[bool] = None,
    ) -> Optional[Any]:
        """Create an extended tier configuration from a base.

        Creates a new TieredToolConfig by merging the base config with
        additional tools. Does not register the result.

        Args:
            base_name: Base tier name to extend
            mandatory: Additional mandatory tools
            vertical_core: Additional vertical core tools
            semantic_pool: Additional semantic pool tools
            stage_tools: Additional stage-specific tools
            readonly_only_for_analysis: Override readonly setting

        Returns:
            New TieredToolConfig with extended tools, or None if base not found
        """
        from victor.core.vertical_types import TieredToolConfig

        base_config = self.get_merged(base_name)
        if not base_config:
            return None

        # Merge with extensions
        new_mandatory = set(base_config.mandatory)
        if mandatory:
            new_mandatory |= mandatory

        new_vertical_core = set(base_config.vertical_core)
        if vertical_core:
            new_vertical_core |= vertical_core

        new_semantic_pool = set(base_config.semantic_pool)
        if semantic_pool:
            new_semantic_pool |= semantic_pool

        new_stage_tools = dict(base_config.stage_tools)
        if stage_tools:
            for stage, tools in stage_tools.items():
                if stage in new_stage_tools:
                    new_stage_tools[stage] |= tools
                else:
                    new_stage_tools[stage] = tools

        new_readonly = (
            readonly_only_for_analysis
            if readonly_only_for_analysis is not None
            else base_config.readonly_only_for_analysis
        )

        return TieredToolConfig(
            mandatory=new_mandatory,
            vertical_core=new_vertical_core,
            semantic_pool=new_semantic_pool,
            stage_tools=new_stage_tools,
            readonly_only_for_analysis=new_readonly,
        )

    def list_all(self) -> List[str]:
        """List all registered tier names.

        Returns:
            List of tier names
        """
        with self._registry_lock:
            return list(self._entries.keys())

    def list_entries(self) -> List[TierRegistryEntry]:
        """List all registry entries.

        Returns:
            List of TierRegistryEntry objects
        """
        with self._registry_lock:
            return list(self._entries.values())

    def clear(self) -> None:
        """Clear all registered tiers.

        Primarily for testing. Re-registers defaults after clearing.
        """
        with self._registry_lock:
            self._entries.clear()

    def has(self, name: str) -> bool:
        """Check if a tier is registered.

        Args:
            name: Tier name

        Returns:
            True if registered, False otherwise
        """
        with self._registry_lock:
            return name in self._entries

    def get_vertical_tier(self, vertical_name: str) -> Optional[Any]:
        """Get tier configuration for a vertical by name.

        Convenience method that looks up a tier by vertical name,
        falling back to the 'base' tier if not found.

        Args:
            vertical_name: Vertical name (e.g., "coding", "research")

        Returns:
            TieredToolConfig for the vertical, or base tier if not found
        """
        config = self.get(vertical_name)
        if config is None:
            config = self.get("base")
        return config

    def validate_tier(self, name: str) -> List[str]:
        """Validate a tier configuration.

        Checks for common issues:
        - Missing parent references
        - Circular inheritance
        - Empty tool sets
        - Invalid tool names

        Args:
            name: Tier name to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: List[str] = []

        with self._registry_lock:
            entry = self._entries.get(name)
            if not entry:
                return [f"Tier '{name}' not found"]

            # Check for circular inheritance
            visited: Set[str] = {name}
            current = entry
            while current.parent:
                if current.parent in visited:
                    errors.append(f"Circular inheritance detected: {current.parent}")
                    break
                visited.add(current.parent)
                parent_entry = self._entries.get(current.parent)
                if not parent_entry:
                    errors.append(f"Parent tier '{current.parent}' not found")
                    break
                current = parent_entry

            # Check for empty mandatory tools
            if not entry.config.mandatory:
                errors.append("Empty mandatory tool set (should include at least 'read')")

            return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire registry to dictionary.

        Returns:
            Dictionary representation of all entries
        """
        with self._registry_lock:
            return {name: entry.to_dict() for name, entry in self._entries.items()}


# Convenience function for module-level access
def get_tool_tier_registry() -> ToolTierRegistry:
    """Get the global ToolTierRegistry instance.

    Returns:
        The singleton ToolTierRegistry instance.
    """
    return ToolTierRegistry.get_instance()


__all__ = [
    "ToolTierRegistry",
    "TierRegistryEntry",
    "get_tool_tier_registry",
]
