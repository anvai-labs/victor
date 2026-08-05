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

"""YAML loader for tool dependency configurations.

This module provides functionality to load, validate, and convert YAML-based
tool dependency configurations into ToolDependencyConfig objects compatible
with BaseToolDependencyProvider.

Design Principles (SOLID):
    - SRP: Loader is separate from schema validation (tool_dependency_schema.py)
    - OCP: Loader can be extended without modification via hooks
    - DIP: Depends on abstractions (ToolDependencyConfig, ToolDependencySpec)

Features:
    - Load YAML files with schema validation
    - Convert validated specs to ToolDependencyConfig
    - Optional tool name canonicalization
    - Factory function for creating providers directly from YAML
    - Support for caching loaded configurations

Example:
    from pathlib import Path
    from victor.core.tool_dependency_loader import (
        load_tool_dependency_yaml,
        create_tool_dependency_provider,
    )

    # Load and create provider in one step
    provider = create_tool_dependency_provider(
        Path("/path/to/tool_dependencies.yaml")
    )

    # Or load config separately for inspection
    config = load_tool_dependency_yaml(
        Path("/path/to/tool_dependencies.yaml"),
        canonicalize=True,
    )
    print(config.required_tools)  # {"read", "write", "edit", "ls", "grep"}
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from importlib.metadata import entry_points

import yaml

from victor.core.yaml_utils import safe_load as yaml_safe_load
from victor_contracts.verticals.tool_dependencies import (
    BaseToolDependencyProvider,
    ToolDependencyConfig,
)
from victor.core.tool_dependency_schema import ToolDependencySpec
from victor.core.tool_types import ToolDependency
from victor.core.verticals.config_registry import (
    get_canonicalization_setting,
    VerticalBehaviorConfigRegistry,
)

logger = logging.getLogger(__name__)


def import_module_with_fallback(module_path: str):
    """Lazy proxy to avoid importing vertical package graph at module import time."""
    from victor.core.verticals.import_resolver import (
        import_module_with_fallback as _resolver,
    )

    return _resolver(module_path)


def module_import_candidates(module_path: str):
    """Lazy proxy for compatibility candidate expansion."""
    from victor.core.verticals.import_resolver import (
        module_import_candidates as _resolver,
    )

    return _resolver(module_path)


def normalize_vertical_name(vertical_name: str) -> str:
    """Lazy proxy for shared vertical-name normalization."""
    from victor.core.verticals.import_resolver import (
        normalize_vertical_name as _resolver,
    )

    return _resolver(vertical_name)


class ToolDependencyLoadError(Exception):
    """Exception raised when loading tool dependency YAML fails.

    Attributes:
        path: Path to the YAML file that failed to load.
        message: Detailed error message.
        cause: Original exception that caused the failure.
    """

    def __init__(
        self,
        path: Path,
        message: str,
        cause: Optional[Exception] = None,
    ):
        """Initialize the error.

        Args:
            path: Path to the YAML file that failed to load.
            message: Detailed error message.
            cause: Original exception that caused the failure.
        """
        self.path = path
        self.message = message
        self.cause = cause
        super().__init__(f"Failed to load {path}: {message}")


class ToolDependencyLoader:
    """Loader for YAML-based tool dependency configurations.

    This class provides methods to load, validate, and convert YAML
    configurations into ToolDependencyConfig objects.

    Attributes:
        canonicalize: Whether to canonicalize tool names during loading.

    Example:
        loader = ToolDependencyLoader(canonicalize=True)
        config = loader.load(Path("tool_dependencies.yaml"))
    """

    def __init__(self, canonicalize: bool = True):
        """Initialize the loader.

        Args:
            canonicalize: Whether to convert tool names to canonical form.
                Defaults to True for consistency with RL Q-values.
        """
        self._canonicalize = canonicalize
        self._cache: Dict[Path, ToolDependencyConfig] = {}

    def load(
        self,
        yaml_path: Path,
        use_cache: bool = True,
    ) -> ToolDependencyConfig:
        """Load a tool dependency YAML file and convert to config.

        Args:
            yaml_path: Path to the YAML configuration file.
            use_cache: Whether to use cached result if available.

        Returns:
            ToolDependencyConfig ready for use with BaseToolDependencyProvider.

        Raises:
            ToolDependencyLoadError: If loading or validation fails.
        """
        yaml_path = yaml_path.resolve()

        # Check cache
        if use_cache and yaml_path in self._cache:
            logger.debug(f"Using cached config for {yaml_path}")
            return self._cache[yaml_path]

        # Load and validate
        spec = self._load_and_validate(yaml_path)

        # Convert to config
        config = self._convert_to_config(spec)

        # Cache result
        self._cache[yaml_path] = config

        logger.info(f"Loaded tool dependency config for '{spec.vertical}' from {yaml_path}")
        return config

    def load_from_string(self, yaml_content: str) -> ToolDependencyConfig:
        """Load a tool dependency config from a YAML string.

        Useful for testing or embedding configs in code.

        Args:
            yaml_content: YAML configuration as a string.

        Returns:
            ToolDependencyConfig ready for use with BaseToolDependencyProvider.

        Raises:
            ToolDependencyLoadError: If parsing or validation fails.
        """
        try:
            data = yaml_safe_load(yaml_content)
            spec = ToolDependencySpec.model_validate(data)
            return self._convert_to_config(spec)
        except yaml.YAMLError as e:
            raise ToolDependencyLoadError(
                Path("<string>"),
                f"YAML parsing error: {e}",
                cause=e,
            )
        except Exception as e:
            raise ToolDependencyLoadError(
                Path("<string>"),
                f"Validation error: {e}",
                cause=e,
            )

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()

    def _load_and_validate(self, yaml_path: Path) -> ToolDependencySpec:
        """Load and validate YAML file.

        Args:
            yaml_path: Path to the YAML file.

        Returns:
            Validated ToolDependencySpec.

        Raises:
            ToolDependencyLoadError: If loading or validation fails.
        """
        if not yaml_path.exists():
            raise ToolDependencyLoadError(
                yaml_path,
                "File not found",
            )

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml_safe_load(f)
        except yaml.YAMLError as e:
            raise ToolDependencyLoadError(
                yaml_path,
                f"YAML parsing error: {e}",
                cause=e,
            )

        if data is None:
            raise ToolDependencyLoadError(
                yaml_path,
                "Empty YAML file",
            )

        try:
            spec = ToolDependencySpec.model_validate(data)
        except Exception as e:
            raise ToolDependencyLoadError(
                yaml_path,
                f"Schema validation error: {e}",
                cause=e,
            )

        return spec

    def _convert_to_config(self, spec: ToolDependencySpec) -> ToolDependencyConfig:
        """Convert validated spec to ToolDependencyConfig.

        Args:
            spec: Validated ToolDependencySpec.

        Returns:
            ToolDependencyConfig for use with BaseToolDependencyProvider.
        """
        # Convert dependencies
        dependencies = self._convert_dependencies(spec.dependencies)

        # Convert transitions
        transitions = self._convert_transitions(spec.transitions)

        # Convert clusters
        clusters = self._convert_clusters(spec.clusters)

        # Convert sequences
        sequences = self._convert_sequences(spec.sequences)

        # Convert tool sets
        required_tools = self._convert_tool_set(spec.required_tools)
        optional_tools = self._convert_tool_set(spec.optional_tools)
        default_sequence = self._convert_tool_list(spec.default_sequence)

        return ToolDependencyConfig(
            dependencies=dependencies,
            transitions=transitions,
            clusters=clusters,
            sequences=sequences,
            required_tools=required_tools,
            optional_tools=optional_tools,
            default_sequence=default_sequence,
        )

    def _convert_dependencies(
        self,
        deps: List[Any],
    ) -> List[ToolDependency]:
        """Convert dependency entries to ToolDependency objects.

        Args:
            deps: List of ToolDependencyEntry objects from spec.

        Returns:
            List of ToolDependency objects.
        """
        result = []
        for dep in deps:
            tool_name = self._canonicalize_name(dep.tool)
            depends_on = self._convert_tool_set(dep.depends_on)
            enables = self._convert_tool_set(dep.enables)

            result.append(
                ToolDependency(
                    tool_name=tool_name,
                    depends_on=depends_on,
                    enables=enables,
                    weight=dep.weight,
                )
            )
        return result

    def _convert_transitions(
        self,
        transitions: Dict[str, List[Any]],
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Convert transition mappings.

        Args:
            transitions: Dict of tool -> list of ToolTransition.

        Returns:
            Dict mapping tool -> list of (next_tool, probability) tuples.
        """
        result: Dict[str, List[Tuple[str, float]]] = {}
        for source, targets in transitions.items():
            source_name = self._canonicalize_name(source)
            result[source_name] = [(self._canonicalize_name(t.tool), t.weight) for t in targets]
        return result

    def _convert_clusters(
        self,
        clusters: Dict[str, List[str]],
    ) -> Dict[str, Set[str]]:
        """Convert cluster definitions.

        Args:
            clusters: Dict of cluster_name -> list of tools.

        Returns:
            Dict mapping cluster_name -> set of tools.
        """
        return {name: self._convert_tool_set(tools) for name, tools in clusters.items()}

    def _convert_sequences(
        self,
        sequences: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """Convert sequence definitions.

        Args:
            sequences: Dict of sequence_name -> list of tools.

        Returns:
            Dict mapping sequence_name -> list of tools.
        """
        return {name: self._convert_tool_list(tools) for name, tools in sequences.items()}

    def _convert_tool_set(self, tools: List[str]) -> Set[str]:
        """Convert tool list to set, optionally canonicalizing names.

        Args:
            tools: List of tool names.

        Returns:
            Set of (optionally canonicalized) tool names.
        """
        if self._canonicalize:
            return {self._canonicalize_name(t) for t in tools}
        return set(tools)

    def _convert_tool_list(self, tools: List[str]) -> List[str]:
        """Convert tool list, optionally canonicalizing names.

        Args:
            tools: List of tool names.

        Returns:
            List of (optionally canonicalized) tool names.
        """
        if self._canonicalize:
            return [self._canonicalize_name(t) for t in tools]
        return list(tools)

    def _canonicalize_name(self, name: str) -> str:
        """Canonicalize a tool name if enabled.

        Args:
            name: Tool name to canonicalize.

        Returns:
            Canonical tool name if canonicalization enabled, else original.
        """
        if not self._canonicalize:
            return name

        try:
            from victor.framework.tool_naming import get_canonical_name

            return get_canonical_name(name)
        except ImportError:
            # If tool_naming not available, return as-is
            logger.warning("Tool naming module not available, skipping canonicalization")
            return name


# Module-level loader instance for convenience
_default_loader = ToolDependencyLoader(canonicalize=True)


def load_tool_dependency_yaml(
    yaml_path: Union[str, Path],
    canonicalize: bool = True,
    use_cache: bool = True,
) -> ToolDependencyConfig:
    """Load a tool dependency YAML file and convert to config.

    Convenience function that uses a module-level loader instance.

    Args:
        yaml_path: Path to the YAML configuration file.
        canonicalize: Whether to convert tool names to canonical form.
        use_cache: Whether to use cached result if available.

    Returns:
        ToolDependencyConfig ready for use with BaseToolDependencyProvider.

    Raises:
        ToolDependencyLoadError: If loading or validation fails.

    Example:
        config = load_tool_dependency_yaml("/path/to/tool_dependencies.yaml")
        print(config.required_tools)  # {"read", "write", "edit", "ls", "grep"}
    """
    path = Path(yaml_path) if isinstance(yaml_path, str) else yaml_path

    if canonicalize != _default_loader._canonicalize:
        # Use a fresh loader if canonicalization setting differs
        loader = ToolDependencyLoader(canonicalize=canonicalize)
        return loader.load(path, use_cache=False)

    return _default_loader.load(path, use_cache=use_cache)


def create_tool_dependency_provider(
    yaml_path: Union[str, Path],
    canonicalize: bool = True,
) -> BaseToolDependencyProvider:
    """Create a tool dependency provider from a YAML configuration file.

    Factory function that loads a YAML file and returns a configured
    BaseToolDependencyProvider instance.

    Args:
        yaml_path: Path to the YAML configuration file.
        canonicalize: Whether to convert tool names to canonical form.

    Returns:
        BaseToolDependencyProvider configured with the YAML settings.

    Raises:
        ToolDependencyLoadError: If loading or validation fails.

    Example:
        provider = create_tool_dependency_provider(
            "/path/to/tool_dependencies.yaml"
        )
        deps = provider.get_dependencies()
        sequences = provider.get_tool_sequences()
    """
    config = load_tool_dependency_yaml(yaml_path, canonicalize=canonicalize)
    return BaseToolDependencyProvider(config=config)


class YAMLToolDependencyProvider(BaseToolDependencyProvider):
    """Tool dependency provider that loads configuration from YAML.

    A convenience class that combines YAML loading with BaseToolDependencyProvider.
    Useful for verticals that want a single class to instantiate.

    Attributes:
        yaml_path: Path to the YAML configuration file.
        vertical: Name of the vertical from the YAML config.

    Example:
        provider = YAMLToolDependencyProvider(
            Path("/path/to/tool_dependencies.yaml")
        )
        print(provider.vertical)  # "coding"
        print(provider.get_required_tools())  # {"read", "write", "edit", "ls", "grep"}
    """

    def __init__(
        self,
        yaml_path: Union[str, Path],
        canonicalize: bool = True,
        additional_dependencies: Optional[List[ToolDependency]] = None,
        additional_sequences: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize the provider from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.
            canonicalize: Whether to convert tool names to canonical form.
            additional_dependencies: Additional dependencies to merge.
            additional_sequences: Additional sequences to merge.

        Raises:
            ToolDependencyLoadError: If loading or validation fails.
        """
        self._yaml_path = Path(yaml_path) if isinstance(yaml_path, str) else yaml_path
        self._canonicalize = canonicalize

        # Load the base config
        config = load_tool_dependency_yaml(
            self._yaml_path,
            canonicalize=canonicalize,
        )

        # Merge additional dependencies if provided
        if additional_dependencies:
            config.dependencies.extend(additional_dependencies)

        # Merge additional sequences if provided
        if additional_sequences:
            config.sequences.update(additional_sequences)

        # Extract vertical name from the spec for reference
        loader = ToolDependencyLoader(canonicalize=False)
        spec = loader._load_and_validate(self._yaml_path)
        self._vertical = spec.vertical

        # Initialize parent
        super().__init__(config=config)

    @property
    def yaml_path(self) -> Path:
        """Get the path to the YAML configuration file."""
        return self._yaml_path

    @property
    def vertical(self) -> str:
        """Get the vertical name from the YAML config."""
        return self._vertical


# Mtime-aware provider cache for automatic cache invalidation
# Format: {path_str: (mtime_ns, provider_instance)}
_provider_cache: Dict[str, Tuple[int, BaseToolDependencyProvider]] = {}
# Vertical provider resolution cache
_vertical_provider_cache: Dict[Tuple[str, Optional[bool]], BaseToolDependencyProvider] = {}
_vertical_provider_cache_lock = threading.Lock()


def get_cached_provider(yaml_path: str) -> BaseToolDependencyProvider:
    """Get a cached tool dependency provider for a YAML file with mtime validation.

    This function caches provider instances with file modification time (mtime)
    validation. If the YAML file is modified, the cache is automatically invalidated
    and a fresh provider is created.

    Args:
        yaml_path: Path to the YAML configuration file (as string for caching).

    Returns:
        Cached BaseToolDependencyProvider instance.

    Example:
        # First call loads from disk
        provider1 = get_cached_provider("/path/to/tool_dependencies.yaml")

        # Second call returns cached instance (if file unchanged)
        provider2 = get_cached_provider("/path/to/tool_dependencies.yaml")
        assert provider1 is provider2

        # If file is modified, cache is invalidated and fresh instance returned
    """
    path = Path(yaml_path).resolve()
    path_str = str(path)

    # Check cache with mtime validation
    if path_str in _provider_cache:
        cached_mtime, provider = _provider_cache[path_str]
        try:
            current_mtime = path.stat().st_mtime_ns
            if current_mtime == cached_mtime:
                logger.debug(f"Cache HIT for {yaml_path} (mtime: {current_mtime})")
                return provider
            else:
                logger.debug(
                    f"Cache INVALIDATED for {yaml_path} "
                    f"(mtime changed: {cached_mtime} -> {current_mtime})"
                )
        except FileNotFoundError:
            logger.debug(f"Cache MISS (file not found): {yaml_path}")
            # File was deleted, invalidate cache entry
            del _provider_cache[path_str]

    # Create new provider (cache miss or invalidated)
    logger.debug(f"Cache MISS for {yaml_path}, creating new provider")
    provider = create_tool_dependency_provider(yaml_path)

    # Cache with current mtime
    try:
        mtime = path.stat().st_mtime_ns
        _provider_cache[path_str] = (mtime, provider)
    except FileNotFoundError:
        # File was deleted during load, don't cache
        pass

    return provider


def invalidate_provider_cache(yaml_path: Optional[str] = None) -> int:
    """Invalidate cached provider instances.

    Useful for testing or forcing a reload of YAML files.

    Args:
        yaml_path: Optional specific path to invalidate. If None, clears all.

    Returns:
        Number of cache entries invalidated.

    Example:
        # Invalidate all caches
        count = invalidate_provider_cache()

        # Invalidate specific file
        count = invalidate_provider_cache("/path/to/tool_dependencies.yaml")
    """
    if yaml_path:
        path_str = str(Path(yaml_path).resolve())
        if path_str in _provider_cache:
            del _provider_cache[path_str]
            logger.debug(f"Invalidated cache for {yaml_path}")
            return 1
        return 0
    else:
        count = len(_provider_cache)
        _provider_cache.clear()
        logger.debug(f"Invalidated all {count} cached providers")
        return count


# NOTE: _VERTICAL_CANONICALIZE_SETTINGS has been removed in favor of
# VerticalBehaviorConfigRegistry. Use get_canonicalization_setting() instead.
# This configuration is now managed via ExtensionManifest in each vertical.

# Resolution telemetry counters for runtime diagnostics.
_TOOL_DEPENDENCY_RESOLUTION_STATS: Dict[str, int] = {
    "total_requests": 0,
    "provider_cache_hits": 0,
    "provider_cache_misses": 0,
    "provider_cache_clears": 0,
    "provider_cache_entries_cleared": 0,
    "entry_point_resolutions": 0,
    "entry_point_load_failures": 0,
    "entry_point_cache_clears": 0,
    "module_factory_resolutions": 0,
    "module_factory_failures": 0,
    "package_resource_resolutions": 0,
    "package_resource_failures": 0,
    "empty_provider_returns": 0,
    "unknown_vertical_errors": 0,
}
_TOOL_DEPENDENCY_STATS_LOCK = threading.Lock()


def _increment_resolution_stat(name: str) -> None:
    """Increment a tool-dependency resolution telemetry counter."""
    with _TOOL_DEPENDENCY_STATS_LOCK:
        _TOOL_DEPENDENCY_RESOLUTION_STATS[name] = _TOOL_DEPENDENCY_RESOLUTION_STATS.get(name, 0) + 1


def get_tool_dependency_resolution_stats() -> Dict[str, int]:
    """Return resolution telemetry counters and entry-point cache stats."""
    with _TOOL_DEPENDENCY_STATS_LOCK:
        stats = dict(_TOOL_DEPENDENCY_RESOLUTION_STATS)
    with _vertical_provider_cache_lock:
        provider_cache_size = len(_vertical_provider_cache)

    cache_info = _cached_tool_dependency_entry_points.cache_info()
    stats.update(
        {
            "entry_point_cache_hits": cache_info.hits,
            "entry_point_cache_misses": cache_info.misses,
            "entry_point_cache_maxsize": cache_info.maxsize or 0,
            "entry_point_cache_currsize": cache_info.currsize,
            "provider_cache_currsize": provider_cache_size,
        }
    )
    return stats


def reset_tool_dependency_resolution_stats(
    clear_entry_point_cache: bool = False,
) -> None:
    """Reset resolution telemetry counters and optionally clear EP cache."""
    with _TOOL_DEPENDENCY_STATS_LOCK:
        for key in _TOOL_DEPENDENCY_RESOLUTION_STATS:
            _TOOL_DEPENDENCY_RESOLUTION_STATS[key] = 0

    if clear_entry_point_cache:
        clear_tool_dependency_entry_point_cache()


@lru_cache(maxsize=1)
def _cached_tool_dependency_entry_points() -> Tuple[Any, ...]:
    """Cache tool dependency entry points for faster repeated resolution."""
    try:
        return tuple(entry_points(group="victor.tool_dependencies"))
    except Exception as e:
        logger.debug("No tool dependency entry points found: %s", e)
        return ()


def clear_tool_dependency_entry_point_cache() -> None:
    """Clear cached tool dependency entry-point lookups."""
    _cached_tool_dependency_entry_points.cache_clear()
    _increment_resolution_stat("entry_point_cache_clears")


def clear_vertical_tool_dependency_provider_cache() -> int:
    """Clear cached vertical tool dependency provider instances.

    Returns:
        Number of cached provider entries removed.
    """
    with _vertical_provider_cache_lock:
        count = len(_vertical_provider_cache)
        _vertical_provider_cache.clear()
    _increment_resolution_stat("provider_cache_clears")
    if count:
        with _TOOL_DEPENDENCY_STATS_LOCK:
            _TOOL_DEPENDENCY_RESOLUTION_STATS["provider_cache_entries_cleared"] += count
    return count


# CONSOLIDATION: plugin-vertical unification — see memory plugin_vertical_consolidation.md
def register_vertical_tool_dependency_provider(
    vertical: str,
    provider: "BaseToolDependencyProvider",
) -> None:
    """Register a tool dependency provider programmatically.

    Used by :class:`HostPluginContext.register_tool_dependency` so a plugin's
    ``register(context)`` call can wire a dependency provider without the
    sidecar ``victor.tool_dependencies`` entry-point group.

    Args:
        vertical: Vertical name (normalized internally).
        provider: Provider instance implementing ``BaseToolDependencyProvider``.
    """
    vertical_name = normalize_vertical_name(vertical)
    with _vertical_provider_cache_lock:
        # Seed both canonicalization variants so downstream callers hit cache.
        for canonicalize in (True, False):
            _vertical_provider_cache[(vertical_name, canonicalize)] = provider


def create_vertical_tool_dependency_provider(
    vertical: str,
    canonicalize: Optional[bool] = None,
) -> BaseToolDependencyProvider:
    """Factory function to create tool dependency providers for verticals using entry points.

    This function loads tool dependency providers from external vertical packages
    via the victor.tool_dependencies entry point group. This decouples the framework
    from hardcoded knowledge about specific verticals.

    Args:
        vertical: Vertical name (coding, devops, research, rag, dataanalysis)
        canonicalize: Whether to canonicalize tool names. If None, uses the
            vertical's default setting.

    Returns:
        Configured tool dependency provider for the vertical, or EmptyToolDependencyProvider
        if the vertical package is not installed.

    Example:
        # Create provider for coding vertical
        provider = create_vertical_tool_dependency_provider("coding")
        deps = provider.get_dependencies()

        # Create provider with explicit canonicalization setting
        provider = create_vertical_tool_dependency_provider("devops", canonicalize=False)

        # Get recommended sequence for edit task
        sequence = provider.get_recommended_sequence("edit")

    Entry Point Registration:
        Vertical packages register their provider factory via entry points:

        [project.entry-points."victor.tool_dependencies"]
        coding = "victor_coding.tool_dependencies:get_provider"
        devops = "victor_devops.tool_dependencies:get_provider"

        # In victor_coding/tool_dependencies.py:
        def get_provider() -> BaseToolDependencyProvider:
            return YAMLToolDependencyProvider(
                yaml_path=Path(__file__).parent / "tool_dependencies.yaml",
                canonicalize=True,
            )
    """
    _increment_resolution_stat("total_requests")
    vertical_name = normalize_vertical_name(vertical)
    effective_canonicalize = (
        canonicalize if canonicalize is not None else get_canonicalization_setting(vertical_name)
    )
    cache_key = (vertical_name, effective_canonicalize)

    with _vertical_provider_cache_lock:
        cached_provider = _vertical_provider_cache.get(cache_key)
    if cached_provider is not None:
        _increment_resolution_stat("provider_cache_hits")
        return cached_provider
    _increment_resolution_stat("provider_cache_misses")

    # Try to load from entry points (external vertical packages).
    for ep in _cached_tool_dependency_entry_points():
        if normalize_vertical_name(ep.name) != vertical_name:
            continue
        try:
            provider_factory = ep.load()
            provider = provider_factory()
            _increment_resolution_stat("entry_point_resolutions")
            with _vertical_provider_cache_lock:
                _vertical_provider_cache[cache_key] = provider
            logger.debug(
                "Loaded tool dependency provider for '%s' from entry point",
                vertical_name,
            )
            return provider
        except Exception as e:
            _increment_resolution_stat("entry_point_load_failures")
            logger.debug(
                "Failed loading tool dependency provider for '%s' from entry point '%s': %s",
                vertical_name,
                ep.name,
                e,
            )

    # Fallback 1: module-level provider factory (external-first resolver).
    # This supports extracted vertical repos even when entry points are unavailable.
    module_path = f"victor.{vertical_name}.tool_dependencies"
    module, resolved_path = import_module_with_fallback(module_path)
    if module is not None and hasattr(module, "get_provider"):
        try:
            provider = module.get_provider()
            _increment_resolution_stat("module_factory_resolutions")
            with _vertical_provider_cache_lock:
                _vertical_provider_cache[cache_key] = provider
            logger.debug(
                "Loaded tool dependency provider for '%s' from module '%s'",
                vertical_name,
                resolved_path or module_path,
            )
            return provider
        except Exception as e:
            _increment_resolution_stat("module_factory_failures")
            logger.debug(
                "Module-level tool dependency provider failed for '%s' from '%s': %s",
                vertical_name,
                resolved_path or module_path,
                e,
            )

    # Check if vertical is known (has explicit configuration or is registered)
    # Note: We no longer restrict to a hardcoded list - any vertical can be used
    # The registry will provide defaults for unknown verticals
    if not VerticalBehaviorConfigRegistry.has_config(vertical_name):
        # This is now informational, not an error - unknown verticals get defaults
        logger.debug(f"Vertical '{vertical_name}' has no explicit configuration, using defaults")

    # Fallback 2: package resource YAML (works for wheel/pip installs).
    checked_packages: List[str] = []
    package_candidates: List[str] = []
    for candidate in module_import_candidates(module_path):
        if "." in candidate:
            package = candidate.rsplit(".", 1)[0]
        else:
            package = candidate
        if package not in package_candidates:
            package_candidates.append(package)

    for package in package_candidates:
        checked_packages.append(package)
        try:
            yaml_resource = files(package).joinpath("tool_dependencies.yaml")
            if yaml_resource.is_file():
                yaml_content = yaml_resource.read_text(encoding="utf-8")
                config = ToolDependencyLoader(canonicalize=effective_canonicalize).load_from_string(
                    yaml_content
                )
                _increment_resolution_stat("package_resource_resolutions")
                logger.debug(
                    "Loaded tool dependency provider for '%s' from package resource '%s:tool_dependencies.yaml'",
                    vertical_name,
                    package,
                )
                provider = BaseToolDependencyProvider(config=config)
                with _vertical_provider_cache_lock:
                    _vertical_provider_cache[cache_key] = provider
                return provider
        except Exception as e:
            _increment_resolution_stat("package_resource_failures")
            logger.debug(
                "Package resource fallback failed for '%s' in '%s': %s",
                vertical_name,
                package,
                e,
            )

    logger.debug(
        "Tool dependencies YAML package resource not found for vertical '%s'. Checked packages: %s",
        vertical_name,
        ", ".join(checked_packages),
    )
    # Return an LSP-compliant empty provider (Null Object pattern)
    from victor.core.tool_types import EmptyToolDependencyProvider

    _increment_resolution_stat("empty_provider_returns")
    provider = EmptyToolDependencyProvider(vertical_name)
    with _vertical_provider_cache_lock:
        _vertical_provider_cache[cache_key] = provider
    return provider


__all__ = [
    "ToolDependencyLoader",
    "ToolDependencyLoadError",
    "YAMLToolDependencyProvider",
    "load_tool_dependency_yaml",
    "create_tool_dependency_provider",
    "get_cached_provider",
    "invalidate_provider_cache",
    "clear_tool_dependency_entry_point_cache",
    "clear_vertical_tool_dependency_provider_cache",
    "get_tool_dependency_resolution_stats",
    "reset_tool_dependency_resolution_stats",
    "create_vertical_tool_dependency_provider",
]
