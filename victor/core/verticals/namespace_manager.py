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

"""Plugin namespace manager for isolating vertical plugins.

This module provides namespace-based isolation for vertical plugins to prevent
naming collisions. Each vertical can have its own isolated namespace for
plugins, allowing multiple versions or implementations to coexist.

Design Principles:
    - Namespace isolation to prevent collisions
    - Multiple versions can coexist
    - Backward compatible with non-namespaced plugins
    - Clear hierarchy: default < contrib < external
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class NamespaceType(Enum):
    """Predefined namespace types with priority ordering."""

    DEFAULT = "default"  # Fallback namespace
    CONTRIBUTED = "contrib"  # Built-in contrib verticals (deprecated)
    EXTERNAL = "external"  # Third-party vertical packages
    EXPERIMENTAL = "experimental"  # Experimental features


@dataclass
class NamespaceConfig:
    """Configuration for a plugin namespace.

    Attributes:
        name: Namespace name (e.g., "default", "external")
        priority: Higher priority namespaces override lower ones (0-100)
        allow_override: Whether plugins can override others in this namespace
        description: Human-readable description
        allowed_prefixes: Optional list of allowed name prefixes
    """

    name: str
    priority: int = 50
    allow_override: bool = True
    description: str = ""
    allowed_prefixes: Optional[List[str]] = None


# Predefined namespace configurations
NAMESPACES: Dict[str, NamespaceConfig] = {
    "default": NamespaceConfig(
        name="default",
        priority=0,
        allow_override=True,
        description="Default namespace for plugins without explicit namespace",
    ),
    "contrib": NamespaceConfig(
        name="contrib",
        priority=50,
        allow_override=True,
        description="Built-in contrib verticals (deprecated, use external)",
    ),
    "external": NamespaceConfig(
        name="external",
        priority=100,
        allow_override=True,
        description="Third-party external vertical packages",
    ),
    "experimental": NamespaceConfig(
        name="experimental",
        priority=25,
        allow_override=False,
        description="Experimental features under development",
    ),
}


@dataclass
class NamespacedPluginKey:
    """A namespaced plugin key that prevents collisions.

    Attributes:
        namespace: Namespace name
        plugin_name: Name of the plugin
        version: Optional version string
        full_key: Full composite key (namespace:name:version)
    """

    namespace: str
    plugin_name: str
    version: Optional[str] = None

    def __post_init__(self) -> None:
        """Generate the full composite key."""
        if self.version:
            self.full_key = f"{self.namespace}:{self.plugin_name}:{self.version}"
        else:
            self.full_key = f"{self.namespace}:{self.plugin_name}"

    def __hash__(self) -> int:
        return hash(self.full_key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NamespacedPluginKey):
            return False
        return self.full_key == other.full_key

    def __str__(self) -> str:
        return self.full_key


class PluginNamespaceManager:
    """Manager for plugin namespace isolation.

    This class provides namespace-based isolation for vertical plugins,
    preventing naming collisions and allowing multiple versions to coexist.

    Example:
        manager = PluginNamespaceManager.get_instance()

        # Register a plugin in external namespace
        key = manager.make_key("external", "my_tool", "1.0.0")

        # Resolve plugin considering namespace priorities
        plugin = manager.resolve("my_tool", available_namespaces=["external", "default"])
    """

    _instance: Optional["PluginNamespaceManager"] = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        """Initialize the namespace manager."""
        self._plugins: Dict[str, NamespacedPluginKey] = {}
        self._plugin_objects: Dict[str, Any] = {}
        self._lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "PluginNamespaceManager":
        """Get singleton namespace manager instance.

        Returns:
            PluginNamespaceManager singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def make_key(
        self,
        namespace: str,
        plugin_name: str,
        version: Optional[str] = None,
    ) -> NamespacedPluginKey:
        """Create a namespaced plugin key.

        Args:
            namespace: Namespace name (e.g., "external", "contrib")
            plugin_name: Name of the plugin
            version: Optional version string

        Returns:
            NamespacedPluginKey object
        """
        return NamespacedPluginKey(namespace, plugin_name, version)

    def register_plugin(
        self,
        namespace: str,
        plugin_name: str,
        plugin: Any,
        version: Optional[str] = None,
    ) -> None:
        """Register a plugin in a namespace.

        Args:
            namespace: Namespace name
            plugin_name: Name of the plugin
            plugin: The plugin object/function
            version: Optional version string
        """
        key = self.make_key(namespace, plugin_name, version)

        with self._lock:
            # Check if plugin already exists
            if key.full_key in self._plugins:
                logger.warning(f"Plugin '{key.full_key}' already registered, overwriting")

            self._plugins[key.full_key] = key
            self._plugin_objects[key.full_key] = plugin
            logger.debug(f"Registered plugin '{key.full_key}'")

    def resolve(
        self,
        plugin_name: str,
        available_namespaces: List[str],
        default_namespace: str = "default",
    ) -> Optional[Any]:
        """Resolve a plugin by name considering namespace priorities.

        Searches for the plugin in available namespaces ordered by priority.
        Higher priority namespaces override lower priority ones.

        Args:
            plugin_name: Name of the plugin to resolve
            available_namespaces: List of namespaces to search
            default_namespace: Fallback namespace if not in available_namespaces

        Returns:
            The resolved plugin object, or None if not found
        """
        # Sort namespaces by priority (highest first)
        namespace_configs = sorted(
            [
                (name, NAMESPACES.get(name, NamespaceConfig(name=name)))
                for name in available_namespaces
            ],
            key=lambda x: (
                -x[1].priority,
                x[0],
            ),  # Sort by priority desc, then name asc
        )

        for namespace, config in namespace_configs:
            # Check for versioned keys first
            for version, key in self._find_keys_by_plugin(namespace, plugin_name):
                if key.full_key in self._plugins:
                    logger.debug(
                        f"Resolved plugin '{plugin_name}' from namespace '{namespace}' "
                        f"(priority={config.priority})"
                    )
                    # Return the plugin object
                    return self._plugin_objects.get(key.full_key)

            # Check for unversioned keys
            unversioned_key = f"{namespace}:{plugin_name}"
            if unversioned_key in self._plugins:
                logger.debug(
                    f"Resolved plugin '{plugin_name}' from namespace '{namespace}' "
                    f"(priority={config.priority})"
                )
                return self._plugin_objects.get(unversioned_key)

        # Not found
        logger.debug(f"Plugin '{plugin_name}' not found in namespaces: {available_namespaces}")
        return None

    def _find_keys_by_plugin(
        self, namespace: str, plugin_name: str
    ) -> List[Tuple[str, NamespacedPluginKey]]:
        """Find all keys for a plugin in a namespace.

        Args:
            namespace: Namespace name
            plugin_name: Plugin name to search for

        Returns:
            List of (version, key) tuples sorted by version (newest first)
        """
        matching = []
        prefix = f"{namespace}:{plugin_name}:"

        for full_key, key in self._plugins.items():
            if full_key.startswith(prefix):
                # Extract version
                version = full_key[len(prefix) :]
                matching.append((version, key))

        # Sort by version (newest first) - assumes semantic versioning
        try:
            matching.sort(key=lambda x: x[0], reverse=True)
        except (ValueError, AttributeError):
            # If versions aren't sortable, keep original order
            pass

        return matching

    def list_plugins(self, namespace: Optional[str] = None) -> List[str]:
        """List all registered plugins.

        Args:
            namespace: If specified, only list plugins in this namespace

        Returns:
            List of plugin full keys
        """
        with self._lock:
            if namespace:
                prefix = f"{namespace}:"
                return [k for k in self._plugins.keys() if k.startswith(prefix)]
            else:
                return list(self._plugins.keys())

    def unregister_plugin(
        self,
        namespace: str,
        plugin_name: str,
        version: Optional[str] = None,
    ) -> bool:
        """Unregister a plugin.

        Args:
            namespace: Namespace name
            plugin_name: Name of the plugin
            version: Optional version string

        Returns:
            True if plugin was unregistered, False if not found
        """
        key = self.make_key(namespace, plugin_name, version)

        with self._lock:
            if key.full_key in self._plugins:
                del self._plugins[key.full_key]
                if key.full_key in self._plugin_objects:
                    del self._plugin_objects[key.full_key]
                logger.debug(f"Unregistered plugin '{key.full_key}'")
                return True
            return False

    def clear(self, namespace: Optional[str] = None) -> int:
        """Clear plugins.

        Args:
            namespace: If specified, only clear plugins in this namespace

        Returns:
            Number of plugins cleared
        """
        with self._lock:
            if namespace:
                prefix = f"{namespace}:"
                to_delete = [k for k in self._plugins.keys() if k.startswith(prefix)]
                for k in to_delete:
                    del self._plugins[k]
                    if k in self._plugin_objects:
                        del self._plugin_objects[k]
                return len(to_delete)
            else:
                count = len(self._plugins)
                self._plugins.clear()
                self._plugin_objects.clear()
                return count

    def get_namespace_config(self, namespace: str) -> Optional[NamespaceConfig]:
        """Get configuration for a namespace.

        Args:
            namespace: Namespace name

        Returns:
            NamespaceConfig if found, None otherwise
        """
        return NAMESPACES.get(namespace)

    def register_namespace(self, config: NamespaceConfig) -> None:
        """Register a custom namespace configuration.

        Args:
            config: NamespaceConfig to register

        Example:
            config = NamespaceConfig(
                name="custom",
                priority=75,
                description="My custom namespace"
            )
            manager.register_namespace(config)
        """
        NAMESPACES[config.name] = config
        logger.debug(f"Registered namespace '{config.name}' with priority {config.priority}")

    def list_namespaces(self) -> List[str]:
        """List all registered namespace names.

        Returns:
            List of namespace names
        """
        return list(NAMESPACES.keys())


# Convenience functions


def get_namespace_manager() -> PluginNamespaceManager:
    """Get the singleton namespace manager instance.

    Returns:
        PluginNamespaceManager instance
    """
    return PluginNamespaceManager.get_instance()


def make_plugin_key(
    namespace: str,
    plugin_name: str,
    version: Optional[str] = None,
) -> NamespacedPluginKey:
    """Create a namespaced plugin key.

    Convenience function for PluginNamespaceManager.make_key().

    Args:
        namespace: Namespace name
        plugin_name: Name of the plugin
        version: Optional version string

    Returns:
        NamespacedPluginKey object
    """
    manager = get_namespace_manager()
    return manager.make_key(namespace, plugin_name, version)


__all__ = [
    "NamespaceType",
    "NamespaceConfig",
    "NamespacedPluginKey",
    "PluginNamespaceManager",
    "get_namespace_manager",
    "make_plugin_key",
    "NAMESPACES",
]
