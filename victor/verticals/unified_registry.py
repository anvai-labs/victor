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

"""
Unified vertical registry with consolidated discovery and loading.

This registry provides a single point of entry for:
- Vertical discovery from entry points
- Capability negotiation
- Version compatibility checking
- Lifecycle management

CONSOLIDATION: plugin-vertical unification — see memory plugin_vertical_consolidation.md
This registry is a *metadata / compatibility* view over the plugin system;
plugin instantiation and vertical class capture are owned by
``victor.core.plugins.registry.PluginRegistry.get_vertical_classes()``. Keep
version checks and compatibility logic here, but do not re-scan entry points
for discovery.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Set

from victor.core.verticals.adapters import ensure_runtime_vertical
from victor.core.verticals.manifest_contract import get_or_create_vertical_manifest
from victor.framework.entry_point_registry import (
    CAPABILITY_ENTRY_POINT_GROUPS,
    PROTOCOL_ENTRY_POINT_GROUPS,
    get_entry_point_registry,
)

logger = logging.getLogger(__name__)


class VerticalStatus(Enum):
    """Vertical installation status."""

    INSTALLED = "installed"
    MISSING = "missing"
    VERSION_MISMATCH = "version_mismatch"
    INCOMPATIBLE = "incompatible"


@dataclass
class VerticalInfo:
    """Information about an installed vertical."""

    name: str
    version: str
    installed_version: Optional[str]
    status: VerticalStatus
    capabilities: Set[str]
    entry_point: Optional[str]
    metadata: Dict[str, Any]

    def is_available(self) -> bool:
        """Check if vertical is available for use."""
        return self.status == VerticalStatus.INSTALLED


class _RegistryPluginContext:
    """Minimal plugin context that registers verticals into the unified registry."""

    def __init__(self, registry: "UnifiedVerticalRegistry", entry_point: Any) -> None:
        self._registry = registry
        self._entry_point = entry_point

    def register_vertical(self, vertical_class: type[Any]) -> None:
        """Register a plugin vertical into the unified registry."""

        runtime_vertical = ensure_runtime_vertical(vertical_class)
        manifest = get_or_create_vertical_manifest(
            vertical_class
        ) or get_or_create_vertical_manifest(runtime_vertical)

        if manifest is not None:
            name = manifest.name
            version = manifest.version
            capabilities = {extension.value for extension in manifest.provides}
            metadata = {
                "manifest": manifest,
                "vertical_class": vertical_class,
                "runtime_vertical_class": runtime_vertical,
                "plugin_name": getattr(self._entry_point, "name", ""),
            }
        else:
            name = getattr(runtime_vertical, "name", getattr(vertical_class, "__name__", "unknown"))
            version = getattr(runtime_vertical, "version", "1.0.0")
            capabilities = set()
            metadata = {
                "vertical_class": vertical_class,
                "runtime_vertical_class": runtime_vertical,
                "plugin_name": getattr(self._entry_point, "name", ""),
            }

        self._registry.register_vertical(
            name=name,
            version=version,
            capabilities=capabilities,
            entry_point=getattr(self._entry_point, "value", None),
            metadata=metadata,
        )


class UnifiedVerticalRegistry:
    """
    Unified registry for all verticals.

    Features:
    - Consolidated discovery from all entry point groups
    - Version compatibility checking
    - Capability aggregation
    - Status monitoring
    """

    def __init__(self) -> None:
        self._verticals: Dict[str, VerticalInfo] = {}
        self._capability_index: Dict[str, Set[str]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize registry by discovering all verticals.

        This method is idempotent - calling it multiple times is safe.
        """
        if self._initialized:
            return

        # Discover from all entry point groups
        await self._discover_from_entry_points()

        # Build capability index
        self._build_capability_index()

        # Check compatibility
        await self._check_compatibility()

        self._initialized = True

    async def _discover_from_entry_points(self) -> None:
        """Discover verticals from all entry point groups."""
        # Main vertical plugins
        await self._load_from_entry_point_group("victor.plugins")

        # Contract protocol providers
        for group in PROTOCOL_ENTRY_POINT_GROUPS:
            await self._load_from_entry_point_group(group)

        # Capability providers
        for group in CAPABILITY_ENTRY_POINT_GROUPS:
            await self._load_from_entry_point_group(group)

    async def _load_from_entry_point_group(self, group: str) -> None:
        """
        Load verticals from a specific entry point group.

        Args:
            group: Entry point group name
        """
        try:
            registry = get_entry_point_registry()
            group_obj = registry.get_group(group)
            if not group_obj:
                return

            for entry_point_tuple in group_obj.entry_points.values():
                ep = entry_point_tuple[0]
                try:
                    await self._load_entry_point(ep)
                except Exception as e:
                    logger.debug(f"Failed to load entry point {ep.name}: {e}")

        except Exception as e:
            logger.debug(f"Entry point group {group} not found: {e}")

    async def _load_entry_point(self, entry_point: Any) -> None:
        """
        Load a single entry point.

        Args:
            entry_point: EntryPoint object
        """
        try:
            plugin = entry_point.load()

            # Call register if it's a VictorPlugin
            if hasattr(plugin, "register"):
                context = _RegistryPluginContext(self, entry_point)
                result = plugin.register(context)
                if inspect.isawaitable(result):
                    await result

        except Exception as e:
            logger.debug(f"Failed to load entry point {entry_point.name}: {e}")

    def _build_capability_index(self) -> None:
        """Build index from capabilities to verticals."""
        self._capability_index = {}

        for name, info in self._verticals.items():
            for capability in info.capabilities:
                if capability not in self._capability_index:
                    self._capability_index[capability] = set()
                self._capability_index[capability].add(name)

    async def _check_compatibility(self) -> None:
        """Check version compatibility for all verticals."""
        for info in self._verticals.values():
            self._apply_compatibility(info)

    def register_vertical(
        self,
        name: str,
        version: str,
        capabilities: Set[str],
        entry_point: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a vertical with the registry.

        Args:
            name: Vertical name
            version: Vertical version
            capabilities: Set of capabilities provided
            entry_point: Entry point string
            metadata: Additional metadata
        """
        # Check if vertical is actually installed
        installed_version = self._check_installed_version(name)

        if installed_version is None:
            status = VerticalStatus.MISSING
        elif installed_version != version:
            status = VerticalStatus.VERSION_MISMATCH
        else:
            status = VerticalStatus.INSTALLED

        info = VerticalInfo(
            name=name,
            version=version,
            installed_version=installed_version,
            status=status,
            capabilities=capabilities,
            entry_point=entry_point,
            metadata=metadata or {},
        )

        self._verticals[name] = info

        # Update capability index
        for capability in capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = set()
            self._capability_index[capability].add(name)

        if self._initialized:
            self._apply_compatibility(info)

    def _check_installed_version(self, package_name: str) -> Optional[str]:
        """
        Check if a package is installed and return its version.

        Args:
            package_name: Name of the package

        Returns:
            Installed version string, or None if not installed
        """
        try:
            import importlib.metadata as metadata

            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            return None

    def _apply_compatibility(self, info: VerticalInfo) -> None:
        """Apply shared compatibility checks to an installed vertical."""

        if info.status != VerticalStatus.INSTALLED:
            return

        try:
            from victor.core.verticals.compatibility_gate import (
                VerticalCompatibilityGate,
            )

            manifest = self._manifest_from_metadata(info)
            report = VerticalCompatibilityGate().assess_vertical(
                vertical_name=info.name,
                vertical_version=info.version,
                manifest=manifest,
            )
            info.metadata["compatibility"] = report.to_dict()
            if not report.compatible:
                info.status = VerticalStatus.INCOMPATIBLE
        except Exception as exc:
            logger.debug("Compatibility check failed for vertical %s: %s", info.name, exc)

    def _manifest_from_metadata(self, info: VerticalInfo):
        """Rehydrate an ExtensionManifest from registry metadata when possible."""

        try:
            from victor_contracts.verticals.manifest import (
                ExtensionManifest,
                ExtensionType,
            )
        except Exception:
            return None

        manifest = info.metadata.get("manifest")
        if isinstance(manifest, ExtensionManifest):
            return manifest

        vertical_class = info.metadata.get("vertical_class")
        if vertical_class is not None:
            try:
                from victor.core.verticals.manifest_contract import (
                    get_or_create_vertical_manifest,
                )

                return get_or_create_vertical_manifest(vertical_class)
            except Exception:
                return None

        if not isinstance(info.metadata, Mapping):
            return None

        raw_provides = info.metadata.get("provides")
        raw_requires = info.metadata.get("requires")
        raw_min_framework_version = info.metadata.get("min_framework_version")
        raw_api_version = info.metadata.get("api_version", 1)
        raw_required_features = info.metadata.get("requires_features", ())

        if (
            raw_provides is None
            and raw_requires is None
            and raw_min_framework_version is None
            and "api_version" not in info.metadata
            and "requires_features" not in info.metadata
        ):
            return None

        def _coerce_types(values: Any) -> set[ExtensionType]:
            if not isinstance(values, (list, tuple, set, frozenset)):
                return set()
            normalized: set[ExtensionType] = set()
            for value in values:
                try:
                    normalized.add(
                        value if isinstance(value, ExtensionType) else ExtensionType(value)
                    )
                except Exception:
                    continue
            return normalized

        return ExtensionManifest(
            name=info.name,
            version=info.version,
            api_version=int(raw_api_version or 1),
            min_framework_version=(
                str(raw_min_framework_version) if raw_min_framework_version else None
            ),
            provides=_coerce_types(raw_provides),
            requires=_coerce_types(raw_requires),
            requires_features={str(value) for value in raw_required_features or ()},
        )

    def get_vertical(self, name: str) -> Optional[VerticalInfo]:
        """
        Get vertical info by name.

        Args:
            name: Vertical name

        Returns:
            VerticalInfo if found, None otherwise
        """
        return self._verticals.get(name)

    def list_verticals(self, status: Optional[VerticalStatus] = None) -> List[VerticalInfo]:
        """
        List all verticals, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of VerticalInfo objects
        """
        verticals = list(self._verticals.values())

        if status:
            verticals = [v for v in verticals if v.status == status]

        return verticals

    def get_verticals_with_capability(self, capability: str) -> List[VerticalInfo]:
        """
        Get all verticals that provide a capability.

        Args:
            capability: Capability name

        Returns:
            List of VerticalInfo objects
        """
        vertical_names = self._capability_index.get(capability, set())
        return [self._verticals[name] for name in vertical_names if name in self._verticals]

    def is_bundle_available(self, bundle_name: str) -> bool:
        """
        Check if all verticals in a bundle are available.

        Args:
            bundle_name: Name of the bundle

        Returns:
            True if all verticals are installed and compatible
        """
        from victor.verticals.product_bundle import resolve_bundle_dependencies

        required_verticals = resolve_bundle_dependencies(bundle_name)

        for vertical_name in required_verticals:
            info = self.get_vertical(vertical_name)
            if not info or not info.is_available():
                return False

        return True

    def get_missing_verticals_for_bundle(self, bundle_name: str) -> List[str]:
        """
        Get list of missing verticals for a bundle.

        Args:
            bundle_name: Name of the bundle

        Returns:
            List of missing vertical names
        """
        from victor.verticals.product_bundle import resolve_bundle_dependencies

        required_verticals = resolve_bundle_dependencies(bundle_name)
        missing = []

        for vertical_name in required_verticals:
            info = self.get_vertical(vertical_name)
            if not info or not info.is_available():
                missing.append(vertical_name)

        return missing

    def get_capabilities(self) -> Set[str]:
        """
        Get all available capabilities.

        Returns:
            Set of capability names
        """
        return set(self._capability_index.keys())

    def reset(self) -> None:
        """Reset the registry (useful for testing)."""
        self._verticals = {}
        self._capability_index = {}
        self._initialized = False


# Singleton instance
_registry: Optional[UnifiedVerticalRegistry] = None


async def get_registry() -> UnifiedVerticalRegistry:
    """
    Get the global vertical registry singleton.

    Returns:
        UnifiedVerticalRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = UnifiedVerticalRegistry()
        await _registry.initialize()
    return _registry


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _registry
    _registry = None
