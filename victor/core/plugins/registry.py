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

"""Plugin registry for Victor.

Handles discovery and registration of Victor plugins via entry points.
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

from victor.core.aot_manifest import AOTManifestManager
from victor.core.container import ServiceContainer, get_container
from victor.core.plugins.protocol import VictorPlugin
from victor.core.registry import SingletonRegistry
from victor.framework.entry_point_registry import get_entry_point_values

if TYPE_CHECKING:
    from victor.core.plugins.context import HostPluginContext

logger = logging.getLogger(__name__)


def call_lifecycle_hook(plugin: VictorPlugin, hook: str, **kwargs: Any) -> Any:
    """Call an optional lifecycle hook on a plugin if it implements it.

    Uses hasattr-based dispatch for backward compatibility with plugins
    that don't implement newer lifecycle methods.

    Args:
        plugin: The plugin instance.
        hook: Method name (e.g., 'on_activate', 'on_deactivate', 'health_check').
        **kwargs: Arguments to pass to the hook.

    Returns:
        Hook return value, or None if not implemented.
    """
    method = getattr(plugin, hook, None)
    if method is not None and callable(method):
        try:
            return method(**kwargs)
        except Exception as e:
            logger.warning("Lifecycle hook '%s' failed on plugin '%s': %s", hook, plugin.name, e)
    return None


async def call_lifecycle_hook_async(plugin: VictorPlugin, hook: str, **kwargs: Any) -> Any:
    """Async-aware lifecycle hook dispatch.

    Prefers ``{hook}_async`` if the plugin implements it, otherwise
    falls back to the synchronous ``{hook}`` method.

    Args:
        plugin: The plugin instance.
        hook: Base method name (e.g., 'on_activate').
        **kwargs: Arguments to pass to the hook.

    Returns:
        Hook return value, or None if not implemented.
    """
    # Try async variant first
    async_method = getattr(plugin, f"{hook}_async", None)
    if async_method is not None and callable(async_method):
        try:
            return await async_method(**kwargs)
        except Exception as e:
            logger.warning(
                "Async lifecycle hook '%s_async' failed on plugin '%s': %s",
                hook,
                plugin.name,
                e,
            )
            return None

    # Fall back to sync hook
    return call_lifecycle_hook(plugin, hook, **kwargs)


class _ExternalPluginAdapter:
    """Wraps an ExternalPluginManager RegisteredPlugin as a VictorPlugin.

    This adapter allows subprocess-based external plugins (discovered via
    plugin.json manifests) to appear in the unified PluginRegistry alongside
    entry-point-based Python plugins.
    """

    def __init__(self, registered_plugin: Any) -> None:
        self._plugin = registered_plugin

    @property
    def name(self) -> str:
        return self._plugin.plugin_id

    def register(self, context: Any) -> None:
        """External plugins register tools via their tool definitions."""
        if not hasattr(context, "register_tool"):
            return
        for tool_def in self._plugin.manifest.tools:
            logger.debug(
                "External plugin %s provides tool: %s",
                self.name,
                tool_def.name,
            )

    def get_cli_app(self) -> None:
        return None

    def on_activate(self) -> None:
        pass

    def on_deactivate(self) -> None:
        pass

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": self._plugin.enabled,
            "adapter": "external_plugin",
            "kind": self._plugin.kind.value,
            "version": self._plugin.version,
            "tools": len(self._plugin.manifest.tools),
        }


# CONSOLIDATION: plugin-vertical unification — see memory plugin_vertical_consolidation.md
# PluginRegistry is the single authority for "installed Victor plugins + their verticals".
# VerticalLoader, VerticalRegistryManager, UnifiedVerticalRegistry, and bootstrap
# capability discovery all read from it via get_vertical_classes() / list_plugins().
class PluginRegistry(SingletonRegistry["PluginRegistry"]):
    """Registry for discovering and managing Victor plugins.

    Discovers plugins via the 'victor.plugins' entry point group
    and optionally from external manifest-based plugins.
    """

    ENTRY_POINT_GROUP = "victor.plugins"
    AOT_GROUPS = ["victor.plugins"]

    def __init__(self) -> None:
        """Initialize registry."""
        super().__init__()
        self._plugins: Dict[str, VictorPlugin] = {}
        self._context: Optional[HostPluginContext] = None
        self._discovered = False
        self._vertical_classes: Optional[Dict[str, Type[Any]]] = None

    def discover(self, force: bool = False) -> List[VictorPlugin]:
        """Discover and load plugins from entry points.

        Uses a three-tier discovery strategy:
        1. AOT manifest (O(1) file read, fastest)
        2. Shared entry-point registry helpers (single-pass scan, process cache)
        3. Registry invalidation on forced refresh

        Args:
            force: Whether to force re-discovery.

        Returns:
            List of discovered plugins.
        """
        if self._discovered and not force:
            return list(self._plugins.values())

        # Invalidate derived caches on forced refresh.
        if force:
            self._vertical_classes = None

        # Try AOT manifest fast-path (skip when forcing refresh)
        aot_entries = None
        if not force:
            aot_entries = self._try_aot_fast_path()

        if aot_entries is not None:
            ep_entries = aot_entries.get(self.ENTRY_POINT_GROUP, {})
        else:
            ep_entries = get_entry_point_values(self.ENTRY_POINT_GROUP, force=force)

        for name, value in ep_entries.items():
            try:
                plugin_cls = self._load_plugin_from_value(name, value)
                logger.info(f"Loading plugin from entry point: {name}")

                # If it's a class, instantiate it
                if isinstance(plugin_cls, type):
                    plugin = plugin_cls()
                else:
                    plugin = plugin_cls

                if isinstance(plugin, VictorPlugin):
                    self._plugins[plugin.name] = plugin
                    logger.info(f"Discovered plugin: {plugin.name} (from {value})")
                else:
                    logger.warning(
                        f"Entry point {name} did not provide a valid VictorPlugin: {plugin_cls}"
                    )
            except Exception as e:
                logger.error(f"Failed to load plugin from entry point {name}: {e}")

        # Update AOT manifest after slow-path discovery
        if aot_entries is None:
            self._update_aot_manifest()

        # Discover external manifest-based plugins and bridge them in.
        self._discover_external_plugins()

        self._discovered = True
        return list(self._plugins.values())

    def _discover_external_plugins(self) -> None:
        """Discover external plugins via plugin.json manifests.

        Wraps each discovered external plugin as a VictorPlugin adapter
        so it appears alongside entry-point-based plugins.
        """
        try:
            from victor.core.plugins.external import ExternalPluginManager

            manager = ExternalPluginManager()
            plugins = manager.discover_plugins()
            for plugin in plugins:
                if plugin.plugin_id in self._plugins:
                    continue  # Entry-point plugins take precedence
                adapter = _ExternalPluginAdapter(plugin)
                self._plugins[adapter.name] = adapter
                logger.info(
                    "Discovered external plugin: %s (%s, %d tools)",
                    plugin.plugin_id,
                    plugin.kind.value,
                    len(plugin.manifest.tools),
                )
        except ImportError:
            logger.debug("External plugin support not available")
        except Exception as e:
            logger.debug("External plugin discovery failed: %s", e)

    def _try_aot_fast_path(self) -> Optional[Dict[str, Dict[str, str]]]:
        """Try to load entry points from AOT manifest.

        Returns:
            Dict mapping group names to their entry point dicts
            (name -> "module:attr"), or None if no valid manifest.
        """
        try:
            manager = AOTManifestManager()
            manifest = manager.load_manifest()
            if manifest is None:
                return None

            result: Dict[str, Dict[str, str]] = {}
            for group, entries in manifest.entries.items():
                group_dict: Dict[str, str] = {}
                for entry in entries:
                    value = f"{entry.module}:{entry.attr}" if entry.attr else entry.module
                    group_dict[entry.name] = value
                result[group] = group_dict

            logger.debug("AOT manifest hit: loaded %d groups", len(result))
            return result
        except Exception as e:
            logger.debug("AOT manifest fast-path failed: %s", e)
            return None

    def _update_aot_manifest(self) -> None:
        """Update AOT manifest after a slow-path scan."""
        try:
            manager = AOTManifestManager()
            manifest = manager.build_manifest(self.AOT_GROUPS)
            manager.save_manifest(manifest)
            logger.debug("Updated AOT manifest with %d groups", len(manifest.entries))
        except Exception as e:
            logger.debug("Failed to update AOT manifest: %s", e)

    def _load_plugin_from_value(self, name: str, value: str) -> Any:
        """Load a plugin object from an entry point value string.

        Args:
            name: Entry point name.
            value: Entry point value in "module:attr" or "module.attr" format.

        Returns:
            Loaded class or object.
        """
        if ":" in value:
            module_name, attr_name = value.split(":", 1)
        else:
            module_name, attr_name = value.rsplit(".", 1)

        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    def register_all(self, container: Optional[ServiceContainer] = None) -> None:
        """Register all discovered plugins into the container.

        Args:
            container: Service container to register into. Defaults to global.
        """
        from victor.core.plugins.context import HostPluginContext

        container = container or get_container()
        self._context = HostPluginContext(container)

        for plugin in self.discover():
            try:
                plugin.register(self._context)
                logger.debug(f"Registered plugin: {plugin.name}")
            except Exception as e:
                logger.error(f"Failed to register plugin {plugin.name}: {e}", exc_info=True)

    @property
    def context(self) -> Optional[HostPluginContext]:
        """Return the registration context containing plugin commands."""
        return self._context

    def get_plugin(self, name: str) -> Optional[VictorPlugin]:
        """Get a plugin by name.

        Args:
            name: Plugin identifier.

        Returns:
            VictorPlugin or None if not found.
        """
        return self._plugins.get(name)

    def check_plugin_health(self) -> Dict[str, Dict[str, Any]]:
        """Run health checks on all registered plugins.

        Returns:
            Dict mapping plugin names to their health status dicts.
            Plugins without health_check get {'healthy': True, 'reason': 'no health_check'}.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for name, plugin in self._plugins.items():
            status = call_lifecycle_hook(plugin, "health_check")
            if status is None:
                status = {"healthy": True, "reason": "no health_check"}
            results[name] = status
        return results

    def list_plugins(self) -> List[VictorPlugin]:
        """List all discovered plugins.

        Returns:
            List of plugins.
        """
        return list(self._plugins.values())

    @property
    def is_discovered(self) -> bool:
        """Whether discovery has been run."""
        return self._discovered

    def list_all_with_type(self) -> List[Dict[str, Any]]:
        """Return a unified view of all plugins with type classification.

        Each entry is a dict with keys: name, type, version, enabled.
        The ``type`` field classifies the plugin as:
        - ``"external"`` — Subprocess plugin wrapped via _ExternalPluginAdapter
        - ``"plugin"`` — Regular VictorPlugin (entry-point-based)

        Returns:
            Sorted list of plugin info dicts.
        """
        if not self._discovered:
            return []

        entries: List[Dict[str, Any]] = []
        for name, plugin in self._plugins.items():
            if isinstance(plugin, _ExternalPluginAdapter):
                plugin_type = "external"
                version = getattr(plugin, "_plugin", None)
                version = getattr(version, "version", "0.0.0") if version else "0.0.0"
                enabled = getattr(getattr(plugin, "_plugin", None), "enabled", True)
            else:
                plugin_type = "plugin"
                version = getattr(plugin, "version", "0.0.0")
                enabled = True

            entries.append(
                {
                    "name": name,
                    "type": plugin_type,
                    "version": str(version),
                    "enabled": enabled,
                }
            )

        return sorted(entries, key=lambda e: e["name"])

    # CONSOLIDATION: plugin-vertical unification — see memory plugin_vertical_consolidation.md
    def get_vertical_classes(self) -> Dict[str, Type[Any]]:
        """Return the vertical classes each discovered plugin registers.

        The single authority for "which plugin provides which vertical". Uses a
        capture-only PluginContext (no side effects) so VerticalLoader,
        VerticalRegistryManager, and bootstrap capability discovery can read
        the same materialized result without repeating entry-point scans or
        re-instantiating plugins.

        Returns:
            Mapping from vertical name to the SDK ``VerticalBase`` subclass
            the plugin registered.
        """
        if self._vertical_classes is not None:
            return dict(self._vertical_classes)

        # Ensure plugins are instantiated; avoid forcing a re-scan.
        if not self._discovered:
            self.discover()

        try:
            from victor_contracts.discovery import collect_verticals_from_candidate
        except ImportError:
            logger.debug("victor_contracts.discovery not importable; skipping vertical capture")
            self._vertical_classes = {}
            return {}

        result: Dict[str, Type[Any]] = {}
        for plugin_name, plugin in self._plugins.items():
            if isinstance(plugin, _ExternalPluginAdapter):
                # External subprocess plugins don't provide SDK verticals.
                continue
            try:
                for vertical_name, vertical_cls in collect_verticals_from_candidate(plugin).items():
                    # First plugin to register wins; emit a warning on collision.
                    if vertical_name in result and result[vertical_name] is not vertical_cls:
                        logger.warning(
                            "Vertical name collision on '%s': keeping %s, ignoring %s",
                            vertical_name,
                            result[vertical_name].__name__,
                            vertical_cls.__name__,
                        )
                        continue
                    result[vertical_name] = vertical_cls
            except Exception as exc:
                logger.debug(
                    "Failed to collect verticals from plugin '%s': %s",
                    plugin_name,
                    exc,
                )

        self._vertical_classes = result
        return dict(result)
