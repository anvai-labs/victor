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

"""Dynamic capability loading for plugins and runtime extension.

This module provides the CapabilityLoader class for discovering, loading,
and managing capabilities at runtime. It supports:

- Loading capabilities from Python modules dynamically
- Registering/unregistering capabilities at runtime
- Hot-reloading modules for plugin development
- Integration with CapabilityRegistryMixin

Design Pattern: Module Loader + Registry
========================================
- CapabilityLoader manages module discovery and loading
- Integrates with existing CapabilityRegistryMixin for registration
- Supports hot-reload via module cache invalidation

Phase 4.4: Dynamic Capability Loading
=====================================
This is part of Victor's architecture roadmap to support dynamic capability
loading for plugins and runtime extension.

Usage:
    from victor.framework import CapabilityLoader

    # Create loader
    loader = CapabilityLoader()

    # Load capabilities from a module
    loader.load_from_module("my_plugin.capabilities")

    # Register dynamically
    loader.register_capability(
        name="custom_safety",
        handler=my_safety_check,
        capability_type=CapabilityType.SAFETY,
        version="1.0",
    )

    # Hot-reload a module
    loader.reload_module("my_plugin.capabilities")

    # Check loaded capabilities
    print(loader.list_capabilities())

    # Apply to orchestrator
    loader.apply_to(orchestrator)
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from victor.framework.module_loader import DynamicModuleLoader
from victor.framework.protocols import (
    CapabilityRegistryProtocol,
    CapabilityType,
    OrchestratorCapability,
)
from victor.framework.strict_mode import ensure_not_private_fallback
from victor_contracts.capabilities import (
    CapabilityEntry as SDKCapabilityEntry,
    CapabilityType as SDKCapabilityType,
    OrchestratorCapability as SDKOrchestratorCapability,
)

logger = logging.getLogger(__name__)


def _normalize_capability_type(
    capability_type: Union[CapabilityType, SDKCapabilityType, str, Any],
) -> CapabilityType:
    """Normalize framework/SDK/string capability types to the framework enum."""

    if isinstance(capability_type, CapabilityType):
        return capability_type
    if isinstance(capability_type, SDKCapabilityType):
        return CapabilityType(capability_type.value)
    if isinstance(capability_type, str):
        return CapabilityType(capability_type)
    raise TypeError(f"Unsupported capability type: {capability_type!r}")


def _coerce_capability(
    capability: Union[OrchestratorCapability, SDKOrchestratorCapability, Any],
) -> OrchestratorCapability:
    """Convert SDK capability declarations into framework capability declarations."""

    if isinstance(capability, OrchestratorCapability):
        return capability
    if isinstance(capability, SDKOrchestratorCapability):
        return OrchestratorCapability(
            name=capability.name,
            capability_type=_normalize_capability_type(capability.capability_type),
            version=capability.version,
            setter=capability.setter,
            getter=capability.getter,
            attribute=capability.attribute,
            description=capability.description,
            required=capability.required,
            deprecated=capability.deprecated,
            deprecated_message=capability.deprecated_message,
        )
    if hasattr(capability, "name") and hasattr(capability, "capability_type"):
        return OrchestratorCapability(
            name=capability.name,
            capability_type=_normalize_capability_type(capability.capability_type),
            version=getattr(capability, "version", "1.0"),
            setter=getattr(capability, "setter", None),
            getter=getattr(capability, "getter", None),
            attribute=getattr(capability, "attribute", None),
            description=getattr(capability, "description", ""),
            required=getattr(capability, "required", False),
            deprecated=getattr(capability, "deprecated", False),
            deprecated_message=getattr(capability, "deprecated_message", ""),
        )
    raise TypeError(f"Unsupported capability object: {capability!r}")


# =============================================================================
# Capability Entry for Dynamic Loading
# =============================================================================


@dataclass
class CapabilityEntry:
    """Entry for a dynamically loaded capability.

    This extends OrchestratorCapability with loader-specific metadata
    for tracking source module and handler functions.

    Attributes:
        capability: The underlying OrchestratorCapability
        handler: The callable that implements the capability (setter)
        getter_handler: Optional getter callable
        source_module: Module name this capability was loaded from
        metadata: Additional loader-specific metadata
    """

    capability: OrchestratorCapability
    handler: Optional[Callable] = None
    getter_handler: Optional[Callable] = None
    source_module: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Get capability name."""
        return self.capability.name

    @property
    def version(self) -> str:
        """Get capability version."""
        return self.capability.version

    @property
    def capability_type(self) -> CapabilityType:
        """Get capability type."""
        return self.capability.capability_type


# =============================================================================
# Capability Load Error
# =============================================================================


class CapabilityLoadError(Exception):
    """Raised when a capability fails to load."""

    def __init__(
        self,
        message: str,
        module_name: Optional[str] = None,
        capability_name: Optional[str] = None,
    ) -> None:
        self.module_name = module_name
        self.capability_name = capability_name
        super().__init__(message)


# =============================================================================
# Capability Loader
# =============================================================================


class CapabilityLoader(DynamicModuleLoader):
    """Dynamic capability loader for plugins and runtime extension.

    This class manages the lifecycle of dynamically loaded capabilities:
    - Discovery from Python modules
    - Registration with type-safe declarations
    - Hot-reload with module cache invalidation
    - Application to orchestrators via CapabilityRegistryMixin

    The loader maintains its own registry of capabilities which can then
    be applied to orchestrators that implement CapabilityRegistryProtocol.

    Inherits from DynamicModuleLoader for shared module loading infrastructure:
    - Module cache invalidation
    - File watching with debounced reloads
    - Submodule tracking

    Thread Safety:
        The loader is NOT thread-safe. Use external synchronization if
        loading/unloading capabilities from multiple threads.

    Example:
        loader = CapabilityLoader()

        # Load from module
        loader.load_from_module("my_plugin.capabilities")

        # Manual registration
        loader.register_capability(
            name="custom_check",
            handler=my_check_function,
            capability_type=CapabilityType.SAFETY,
        )

        # Apply to orchestrator
        loader.apply_to(orchestrator)
    """

    def __init__(
        self,
        auto_discover: bool = False,
        plugin_dirs: Optional[List[Path]] = None,
    ) -> None:
        """Initialize the capability loader.

        Args:
            auto_discover: If True, discover capabilities from plugin_dirs on init
            plugin_dirs: Directories to search for capability modules
        """
        # Initialize base class with watch_dirs
        super().__init__(watch_dirs=plugin_dirs or [])

        self._capabilities: Dict[str, CapabilityEntry] = {}
        self._module_capabilities: Dict[str, Set[str]] = {}  # module -> capability names
        self._plugin_dirs = plugin_dirs or []
        self._watchers: List[Callable[[str, str], None]] = []  # change callbacks

        if auto_discover:
            self.discover_capabilities()

    # =========================================================================
    # Module Loading
    # =========================================================================

    def load_from_module(
        self,
        module_name: str,
        force_reload: bool = False,
    ) -> List[str]:
        """Load capabilities from a Python module.

        The module should define capabilities using one of these patterns:

        1. CAPABILITIES list of OrchestratorCapability/CapabilityEntry
        2. @capability decorator on functions
        3. Classes extending BaseCapa (if applicable)

        Args:
            module_name: Fully qualified module name (e.g., "my_plugin.caps")
            force_reload: If True, reload even if already loaded

        Returns:
            List of capability names that were loaded

        Raises:
            CapabilityLoadError: If module cannot be loaded

        Example:
            # In my_plugin/capabilities.py:
            CAPABILITIES = [
                CapabilityEntry(
                    capability=OrchestratorCapability(
                        name="custom_safety",
                        capability_type=CapabilityType.SAFETY,
                        setter="custom_safety_check",
                    ),
                    handler=my_safety_check,
                )
            ]

            # Or using decorator:
            @capability(
                name="custom_safety",
                capability_type=CapabilityType.SAFETY,
            )
            def my_safety_check(patterns: List[str]) -> None:
                ...
        """
        if module_name in self._loaded_modules and not force_reload:
            logger.debug(f"Module '{module_name}' already loaded, skipping")
            return list(self._module_capabilities.get(module_name, []))

        try:
            # Invalidate cache if reloading
            if force_reload and module_name in sys.modules:
                self._invalidate_module(module_name)

            # Import module
            module = importlib.import_module(module_name)
            self.track_module(module_name, module)
            self._module_capabilities[module_name] = set()

            loaded_names = []

            # Pattern 1: Check for CAPABILITIES list
            if hasattr(module, "CAPABILITIES"):
                capabilities = module.CAPABILITIES
                for entry in capabilities:
                    name = self._register_from_entry(entry, module_name)
                    if name:
                        loaded_names.append(name)

            # Pattern 2: Check for decorated functions
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, "_capability_meta"):
                    meta = attr._capability_meta
                    name = self._register_from_meta(attr, meta, module_name)
                    if name:
                        loaded_names.append(name)

            # Pattern 3: Check for capability classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and hasattr(attr, "get_capability")
                    and attr_name != "BaseCapability"
                ):
                    try:
                        instance = attr()
                        capability = instance.get_capability()
                        handler = getattr(instance, "execute", None)
                        name = self._register_capability_internal(
                            capability=capability,
                            handler=handler,
                            source_module=module_name,
                        )
                        if name:
                            loaded_names.append(name)
                    except Exception as e:
                        logger.warning(f"Failed to instantiate capability class {attr_name}: {e}")

            logger.info(
                f"Loaded {len(loaded_names)} capabilities from '{module_name}': "
                f"{', '.join(loaded_names)}"
            )
            return loaded_names

        except ImportError as e:
            raise CapabilityLoadError(
                f"Failed to import module '{module_name}': {e}",
                module_name=module_name,
            )
        except Exception as e:
            raise CapabilityLoadError(
                f"Error loading capabilities from '{module_name}': {e}",
                module_name=module_name,
            )

    def load_from_path(
        self,
        path: Union[str, Path],
        module_name: Optional[str] = None,
    ) -> List[str]:
        """Load capabilities from a file path.

        Args:
            path: Path to Python file containing capabilities
            module_name: Optional module name (defaults to filename)

        Returns:
            List of loaded capability names

        Raises:
            CapabilityLoadError: If file cannot be loaded
        """
        path = Path(path)
        if not path.exists():
            raise CapabilityLoadError(
                f"Capability file not found: {path}",
                module_name=str(path),
            )

        if module_name is None:
            module_name = f"victor_capability_{path.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise CapabilityLoadError(
                    f"Could not load spec for {path}",
                    module_name=module_name,
                )

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            self.track_module(module_name, module, file_path=path)
            self._module_capabilities[module_name] = set()

            loaded_names = []

            # Check for CAPABILITIES list
            if hasattr(module, "CAPABILITIES"):
                for entry in module.CAPABILITIES:
                    name = self._register_from_entry(entry, module_name)
                    if name:
                        loaded_names.append(name)

            # Check for decorated functions
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, "_capability_meta"):
                    meta = attr._capability_meta
                    name = self._register_from_meta(attr, meta, module_name)
                    if name:
                        loaded_names.append(name)

            logger.info(f"Loaded {len(loaded_names)} capabilities from '{path}'")
            return loaded_names

        except Exception as e:
            raise CapabilityLoadError(
                f"Failed to load capabilities from '{path}': {e}",
                module_name=module_name,
            )

    def _register_from_entry(
        self,
        entry: Union[
            CapabilityEntry,
            SDKCapabilityEntry,
            OrchestratorCapability,
            SDKOrchestratorCapability,
            Dict[str, Any],
        ],
        source_module: str,
    ) -> Optional[str]:
        """Register capability from an entry definition.

        Args:
            entry: Capability entry (various formats supported)
            source_module: Source module name

        Returns:
            Registered capability name or None
        """
        if isinstance(entry, (CapabilityEntry, SDKCapabilityEntry)):
            return self._register_capability_internal(
                capability=_coerce_capability(entry.capability),
                handler=entry.handler,
                getter_handler=entry.getter_handler,
                source_module=source_module,
                metadata=entry.metadata,
            )
        elif isinstance(entry, (OrchestratorCapability, SDKOrchestratorCapability)):
            return self._register_capability_internal(
                capability=_coerce_capability(entry),
                source_module=source_module,
            )
        elif isinstance(entry, dict):
            # Dict format: construct capability from dict
            capability = OrchestratorCapability(
                name=entry["name"],
                capability_type=_normalize_capability_type(
                    entry.get("capability_type", CapabilityType.TOOL)
                ),
                version=entry.get("version", "1.0"),
                setter=entry.get("setter"),
                getter=entry.get("getter"),
                attribute=entry.get("attribute"),
                description=entry.get("description", ""),
            )
            return self._register_capability_internal(
                capability=capability,
                handler=entry.get("handler"),
                getter_handler=entry.get("getter_handler"),
                source_module=source_module,
            )
        else:
            logger.warning(f"Unknown capability entry type: {type(entry)}")
            return None

    def _register_from_meta(
        self,
        func: Callable,
        meta: Dict[str, Any],
        source_module: str,
    ) -> Optional[str]:
        """Register capability from decorator metadata.

        Args:
            func: Decorated function
            meta: Decorator metadata
            source_module: Source module name

        Returns:
            Registered capability name or None
        """
        capability = OrchestratorCapability(
            name=meta["name"],
            capability_type=_normalize_capability_type(
                meta.get("capability_type", CapabilityType.TOOL)
            ),
            version=meta.get("version", "1.0"),
            setter=meta.get("setter", meta["name"]),
            getter=meta.get("getter"),
            description=meta.get("description", func.__doc__ or ""),
        )
        return self._register_capability_internal(
            capability=capability,
            handler=func,
            source_module=source_module,
        )

    def _register_capability_internal(
        self,
        capability: OrchestratorCapability,
        handler: Optional[Callable] = None,
        getter_handler: Optional[Callable] = None,
        source_module: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Internal capability registration.

        Args:
            capability: OrchestratorCapability to register
            handler: Setter handler function
            getter_handler: Getter handler function
            source_module: Source module name
            metadata: Additional metadata

        Returns:
            Capability name if registered, None if skipped
        """
        name = capability.name

        if name in self._capabilities:
            existing = self._capabilities[name]
            logger.warning(
                f"Capability '{name}' already registered from "
                f"'{existing.source_module}', replacing with '{source_module}'"
            )

        entry = CapabilityEntry(
            capability=capability,
            handler=handler,
            getter_handler=getter_handler,
            source_module=source_module,
            metadata=metadata or {},
        )
        self._capabilities[name] = entry

        if source_module:
            self._module_capabilities.setdefault(source_module, set()).add(name)

        logger.debug(
            f"Registered capability '{name}' v{capability.version} "
            f"(type={capability.capability_type.value})"
        )
        return name

    # =========================================================================
    # Manual Registration
    # =========================================================================

    def register_capability(
        self,
        name: str,
        handler: Callable,
        capability_type: CapabilityType = CapabilityType.TOOL,
        version: str = "1.0",
        getter: Optional[Callable] = None,
        description: str = "",
        **kwargs: Any,
    ) -> None:
        """Register a capability manually.

        This provides a programmatic API for registering capabilities
        without loading from a module.

        Args:
            name: Capability name (must be unique)
            handler: Callable that implements the capability (setter)
            capability_type: Type of capability
            version: Version string (MAJOR.MINOR format)
            getter: Optional getter callable
            description: Human-readable description
            **kwargs: Additional capability options

        Example:
            def my_safety_check(patterns: List[str]) -> None:
                for pattern in patterns:
                    # Apply pattern
                    pass

            loader.register_capability(
                name="custom_safety",
                handler=my_safety_check,
                capability_type=CapabilityType.SAFETY,
                version="1.0",
                description="Custom safety pattern checker",
            )
        """
        capability = OrchestratorCapability(
            name=name,
            capability_type=capability_type,
            version=version,
            setter=name,
            getter=name if getter else None,
            description=description,
            **kwargs,
        )

        self._register_capability_internal(
            capability=capability,
            handler=handler,
            getter_handler=getter,
            source_module="<manual>",
        )

    def unregister_capability(self, name: str) -> bool:
        """Unregister a capability.

        Args:
            name: Capability name to unregister

        Returns:
            True if capability was unregistered
        """
        if name not in self._capabilities:
            logger.warning(f"Capability '{name}' not registered")
            return False

        entry = self._capabilities.pop(name)

        # Remove from module tracking
        if entry.source_module and entry.source_module in self._module_capabilities:
            self._module_capabilities[entry.source_module].discard(name)

        logger.debug(f"Unregistered capability '{name}'")
        return True

    # =========================================================================
    # Hot Reload
    # =========================================================================

    def reload_module(self, module_name: str) -> List[str]:
        """Hot-reload a module and re-register its capabilities.

        This performs proper module cache invalidation to ensure
        code changes are picked up.

        Args:
            module_name: Module name to reload

        Returns:
            List of reloaded capability names

        Raises:
            CapabilityLoadError: If module was not previously loaded
        """
        if module_name not in self._loaded_modules:
            raise CapabilityLoadError(
                f"Module '{module_name}' was not loaded by this loader",
                module_name=module_name,
            )

        # Check if this was loaded from a file path
        module_path = self.get_module_path(module_name)

        # Unregister existing capabilities from this module
        if module_name in self._module_capabilities:
            for cap_name in list(self._module_capabilities[module_name]):
                self.unregister_capability(cap_name)

        # Invalidate module cache (use base class method)
        self._invalidate_module(module_name)

        # Notify watchers
        for watcher in self._watchers:
            try:
                watcher(module_name, "reload")
            except Exception as e:
                logger.warning(f"Watcher error during reload: {e}")

        # Reload module - use path if available, otherwise try import
        if module_path and module_path.exists():
            return self.load_from_path(module_path, module_name)
        else:
            return self.load_from_module(module_name, force_reload=True)

    def _invalidate_module(self, module_name: str) -> None:
        """Invalidate Python module cache for hot-reload.

        Delegates to base class invalidate_module() for shared logic.

        Args:
            module_name: Module name to invalidate
        """
        # Use base class implementation
        self.invalidate_module(module_name)

    def unload_module(self, module_name: str) -> bool:
        """Unload a module and all its capabilities.

        Args:
            module_name: Module name to unload

        Returns:
            True if module was unloaded
        """
        if module_name not in self._loaded_modules:
            logger.warning(f"Module '{module_name}' not loaded")
            return False

        # Unregister all capabilities from this module
        if module_name in self._module_capabilities:
            for cap_name in list(self._module_capabilities[module_name]):
                self.unregister_capability(cap_name)
            del self._module_capabilities[module_name]

        # Invalidate module cache (uses base class method via _invalidate_module)
        self._invalidate_module(module_name)

        # Untrack module (cleanup base class tracking)
        self.untrack_module(module_name)

        # Notify watchers
        for watcher in self._watchers:
            try:
                watcher(module_name, "unload")
            except Exception as e:
                logger.warning(f"Watcher error during unload: {e}")

        logger.info(f"Unloaded module '{module_name}'")
        return True

    # =========================================================================
    # Discovery
    # =========================================================================

    def discover_capabilities(self) -> Dict[str, List[str]]:
        """Discover and load capabilities from plugin directories.

        Searches configured plugin directories for capability modules
        (files named *_capabilities.py or capabilities.py).

        Returns:
            Dict mapping module path to list of loaded capability names
        """
        results: Dict[str, List[str]] = {}

        for plugin_dir in self._plugin_dirs:
            plugin_dir = Path(plugin_dir).expanduser()
            if not plugin_dir.exists():
                continue

            # Find capability modules
            for pattern in ["*_capabilities.py", "capabilities.py"]:
                for path in plugin_dir.rglob(pattern):
                    try:
                        loaded = self.load_from_path(path)
                        results[path] = loaded
                    except CapabilityLoadError as e:
                        logger.warning(f"Failed to discover capabilities: {e}")

        logger.info(
            f"Discovered capabilities from {len(results)} modules in "
            f"{len(self._plugin_dirs)} directories"
        )
        return results

    def add_plugin_dir(self, path: Union[str, Path]) -> None:
        """Add a plugin directory for discovery.

        Args:
            path: Directory path to add
        """
        path = Path(path).expanduser()
        if path not in self._plugin_dirs:
            self._plugin_dirs.append(path)
            logger.debug(f"Added plugin directory: {path}")

    # =========================================================================
    # Query
    # =========================================================================

    def list_capabilities(self) -> List[str]:
        """Get list of all registered capability names.

        Returns:
            List of capability names
        """
        return list(self._capabilities.keys())

    def get_capability(self, name: str) -> Optional[CapabilityEntry]:
        """Get a capability entry by name.

        Args:
            name: Capability name

        Returns:
            CapabilityEntry or None if not found
        """
        return self._capabilities.get(name)

    def has_capability(self, name: str) -> bool:
        """Check if a capability is registered.

        Args:
            name: Capability name

        Returns:
            True if registered
        """
        return name in self._capabilities

    def get_capabilities_by_type(self, capability_type: CapabilityType) -> List[CapabilityEntry]:
        """Get all capabilities of a specific type.

        Args:
            capability_type: Type to filter by

        Returns:
            List of matching capability entries
        """
        return [
            entry
            for entry in self._capabilities.values()
            if entry.capability_type == capability_type
        ]

    def get_capabilities_by_module(self, module_name: str) -> List[CapabilityEntry]:
        """Get all capabilities from a specific module.

        Args:
            module_name: Module name

        Returns:
            List of capability entries from that module
        """
        cap_names = self._module_capabilities.get(module_name, set())
        return [self._capabilities[name] for name in cap_names if name in self._capabilities]

    def list_loaded_modules(self) -> List[str]:
        """Get list of all loaded module names.

        Returns:
            List of module names
        """
        return self.get_tracked_modules()

    # =========================================================================
    # Application to Orchestrator
    # =========================================================================

    def apply_to(
        self,
        orchestrator: Any,
        capability_names: Optional[List[str]] = None,
    ) -> List[str]:
        """Apply loaded capabilities to an orchestrator.

        This registers all loaded capabilities with the orchestrator's
        capability registry (if it implements CapabilityRegistryProtocol).

        Args:
            orchestrator: Target orchestrator
            capability_names: Optional list of specific capabilities to apply
                            (applies all if None)

        Returns:
            List of successfully applied capability names

        Raises:
            TypeError: If orchestrator doesn't support capability registration
        """
        applied = []

        # Resolve registration path (public API first, private fallback only when allowed).
        register_dynamic = getattr(orchestrator, "register_dynamic_capability", None)
        private_register = getattr(orchestrator, "_register_capability", None)
        use_public_registration = callable(register_dynamic)
        use_private_registration = callable(private_register)

        if not use_public_registration and not use_private_registration:
            raise TypeError(
                f"Orchestrator {type(orchestrator).__name__} does not expose a "
                "capability registration API (expected register_dynamic_capability)"
            )

        # Strict mode blocks private registration fallback paths.
        if not use_public_registration:
            ensure_not_private_fallback("_register_capability", operation="capability registration")

        names_to_apply = capability_names or list(self._capabilities.keys())

        for name in names_to_apply:
            entry = self._capabilities.get(name)
            if entry is None:
                logger.warning(f"Capability '{name}' not found, skipping")
                continue

            try:
                if use_public_registration:
                    result = register_dynamic(
                        entry.capability,
                        setter_method=entry.handler,
                        getter_method=entry.getter_handler,
                    )
                    if result is False:
                        raise RuntimeError(
                            f"register_dynamic_capability returned False for '{name}'"
                        )
                else:
                    private_register(
                        entry.capability,
                        setter_method=entry.handler,
                        getter_method=entry.getter_handler,
                    )

                applied.append(name)
                logger.debug(f"Applied capability '{name}' to orchestrator")
            except Exception as e:
                logger.error(f"Failed to apply capability '{name}': {e}")

        logger.info(f"Applied {len(applied)}/{len(names_to_apply)} capabilities")
        return applied

    # =========================================================================
    # Watchers for Hot Reload
    # =========================================================================

    def add_watcher(self, callback: Callable[[str, str], None]) -> None:
        """Add a watcher callback for module changes.

        The callback is invoked with (module_name, event_type) where
        event_type is one of: "reload", "unload".

        Args:
            callback: Callback function
        """
        self._watchers.append(callback)

    def remove_watcher(self, callback: Callable[[str, str], None]) -> None:
        """Remove a watcher callback.

        Args:
            callback: Callback function to remove
        """
        if callback in self._watchers:
            self._watchers.remove(callback)

    # =========================================================================
    # File Watching (Optional)
    # =========================================================================

    def _on_module_changed(
        self,
        module_name: str,
        file_path: Path,
        event_type: str,
    ) -> None:
        """Handle a module file change (override from DynamicModuleLoader).

        This method is called when a watched file changes.

        Args:
            module_name: Name of the changed module
            file_path: Path to the changed file
            event_type: Type of change ("modified", "created", "deleted")
        """
        if event_type == "deleted":
            # Don't try to reload deleted modules
            logger.debug(f"Module '{module_name}' file deleted: {file_path}")
            return

        try:
            self.reload_module(module_name)
            # Notify watchers
            for watcher in self._watchers:
                try:
                    watcher(module_name, "reload")
                except Exception as e:
                    logger.warning(f"Watcher error during auto-reload: {e}")
        except Exception as e:
            logger.error(f"Auto-reload failed for '{module_name}': {e}")

    def watch_for_changes(
        self,
        callback: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Start watching plugin directories for file changes.

        When a capability module file changes, it will be automatically
        reloaded. Requires the 'watchdog' package.

        Uses base class DynamicModuleLoader's file watching infrastructure.

        Args:
            callback: Optional callback(module_name) when a module is reloaded

        Returns:
            True if watching started successfully
        """

        # Use base class file watcher with a callback wrapper
        def on_file_change(file_path: str, event_type: str) -> None:
            if callback:
                module_name = self._find_module_for_file(Path(file_path))
                if module_name:
                    callback(module_name)

        return self.setup_file_watcher(
            dirs=[Path(d) for d in self._plugin_dirs],
            on_change=on_file_change,
            recursive=True,
        )

    def stop_watching(self) -> None:
        """Stop watching for file changes."""
        self.stop_file_watcher()

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self) -> "CapabilityLoader":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup."""
        self.stop_watching()


# =============================================================================
# Capability Decorator
# =============================================================================


def capability(
    name: str,
    capability_type: CapabilityType = CapabilityType.TOOL,
    version: str = "1.0",
    description: Optional[str] = None,
    setter: Optional[str] = None,
    getter: Optional[str] = None,
) -> Callable[[Callable], Callable]:
    """Decorator to mark a function as a capability handler.

    This decorator attaches metadata to functions so they can be
    discovered by CapabilityLoader.load_from_module().

    Args:
        name: Capability name
        capability_type: Type of capability
        version: Version string
        description: Human-readable description (defaults to docstring)
        setter: Setter method name (defaults to name)
        getter: Optional getter method name

    Returns:
        Decorator function

    Example:
        @capability(
            name="custom_safety",
            capability_type=CapabilityType.SAFETY,
            version="1.0",
            description="Apply custom safety patterns",
        )
        def apply_custom_safety(patterns: List[str]) -> None:
            for pattern in patterns:
                # Apply pattern logic
                pass
    """

    def decorator(func: Callable) -> Callable:
        func._capability_meta = {
            "name": name,
            "capability_type": capability_type,
            "version": version,
            "description": description or func.__doc__ or "",
            "setter": setter or name,
            "getter": getter,
        }
        return func

    return decorator


# =============================================================================
# Factory Functions
# =============================================================================


def create_capability_loader(
    plugin_dirs: Optional[List[Union[str, Path]]] = None,
    auto_discover: bool = False,
) -> CapabilityLoader:
    """Create a CapabilityLoader with standard configuration.

    Args:
        plugin_dirs: Directories to search for capabilities
        auto_discover: If True, discover capabilities on creation

    Returns:
        Configured CapabilityLoader
    """
    dirs = [Path(d) for d in plugin_dirs] if plugin_dirs else []
    return CapabilityLoader(
        plugin_dirs=dirs,
        auto_discover=auto_discover,
    )


def get_default_capability_loader() -> CapabilityLoader:
    """Get or create the default capability loader.

    This provides a singleton-like pattern for the default loader
    that searches ~/.victor/capabilities/ for capability modules.

    Returns:
        Default CapabilityLoader instance
    """
    global _default_loader

    if _default_loader is None:
        default_dir = Path.home() / ".victor" / "capabilities"
        _default_loader = CapabilityLoader(
            plugin_dirs=[default_dir] if default_dir.exists() else [],
        )

    return _default_loader


_default_loader: Optional[CapabilityLoader] = None


__all__ = [
    "CapabilityLoader",
    "CapabilityEntry",
    "CapabilityLoadError",
    "capability",
    "create_capability_loader",
    "get_default_capability_loader",
]
