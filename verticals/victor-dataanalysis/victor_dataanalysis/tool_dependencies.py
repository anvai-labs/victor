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

"""Data Analysis Tool Dependencies - YAML-based tool relationships for analysis workflows.

This module provides the DataAnalysisToolDependencyProvider which loads tool
dependency configuration from YAML (tool_dependencies.yaml) using the new
YAML-based configuration system.

Migration Notes:
    - Migrated from hand-coded Python dictionaries to YAML configuration
    - Uses YAMLToolDependencyProvider from victor.core.tool_dependency_loader
    - YAML file: victor/dataanalysis/tool_dependencies.yaml
    - Backward compatibility maintained via deprecated constant exports

Design Principles:
    - YAML-first: Configuration in declarative YAML, not imperative Python
    - Canonical Names: Uses canonical tool names for RL Q-value consistency
    - Backward Compatibility: Deprecated constants still available for migration
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

from victor.framework.extensions import YAMLToolDependencyProvider

if TYPE_CHECKING:
    from victor_contracts.verticals.protocols import ToolDependency

logger = logging.getLogger(__name__)

# Path to the YAML configuration file
_YAML_PATH = Path(__file__).parent / "tool_dependencies.yaml"


class DataAnalysisToolDependencyProvider(YAMLToolDependencyProvider):
    """Tool dependency provider for data analysis vertical.

    .. deprecated::
        Use ``create_vertical_tool_dependency_provider('dataanalysis')`` instead.
        This class is maintained for backward compatibility.

    Loads configuration from YAML file (tool_dependencies.yaml) using the
    YAMLToolDependencyProvider base class.

    This provider supports:
    - Data profiling workflows
    - Visualization sequences
    - ML pipeline tool transitions
    - Statistical analysis patterns

    Example:
        # Preferred (new code):
        from victor.framework.extensions import create_vertical_tool_dependency_provider
        provider = create_vertical_tool_dependency_provider("dataanalysis")

        # Deprecated (backward compatible):
        >>> provider = DataAnalysisToolDependencyProvider()
        >>> provider.vertical
        'data_analysis'
        >>> provider.get_required_tools()
        {'read', 'write', 'shell'}
    """

    def __init__(self) -> None:
        """Initialize the provider with YAML configuration.

        Note: canonicalize=False to preserve distinct tool names like 'code_search'
        which would otherwise be mapped to 'grep' by the alias system. The DataAnalysis
        vertical uses both 'grep' (keyword) and 'code_search' (semantic) as separate tools.

        .. deprecated::
            Use ``create_vertical_tool_dependency_provider('dataanalysis')`` instead.
        """
        warnings.warn(
            "DataAnalysisToolDependencyProvider is deprecated. "
            "Use create_vertical_tool_dependency_provider('dataanalysis') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            yaml_path=_YAML_PATH,
            canonicalize=False,  # Preserve 'code_search' as distinct from 'grep'
        )
        logger.debug(f"Loaded DataAnalysis tool dependencies from {_YAML_PATH}")


# =============================================================================
# BACKWARD COMPATIBILITY EXPORTS (DEPRECATED)
# =============================================================================
# These constants are deprecated and will be removed in a future version.
# Use DataAnalysisToolDependencyProvider instead to access configuration.


def _get_deprecated_config() -> "DataAnalysisToolDependencyProvider":
    """Lazily load provider for deprecated constant access."""
    if not hasattr(_get_deprecated_config, "_provider"):
        _get_deprecated_config._provider = DataAnalysisToolDependencyProvider()
    return _get_deprecated_config._provider


def _warn_deprecated(name: str) -> None:
    """Emit deprecation warning for legacy constant access."""
    warnings.warn(
        f"{name} is deprecated. Use DataAnalysisToolDependencyProvider instead. "
        "This constant will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3,
    )


class _DeprecatedTransitions:
    """Lazy loader for deprecated DATA_ANALYSIS_TOOL_TRANSITIONS."""

    def __getitem__(self, key: str) -> List[Tuple[str, float]]:
        _warn_deprecated("DATA_ANALYSIS_TOOL_TRANSITIONS")
        return _get_deprecated_config().get_tool_transitions().get(key, [])

    def __iter__(self):
        _warn_deprecated("DATA_ANALYSIS_TOOL_TRANSITIONS")
        return iter(_get_deprecated_config().get_tool_transitions())

    def items(self):
        _warn_deprecated("DATA_ANALYSIS_TOOL_TRANSITIONS")
        return _get_deprecated_config().get_tool_transitions().items()

    def keys(self):
        _warn_deprecated("DATA_ANALYSIS_TOOL_TRANSITIONS")
        return _get_deprecated_config().get_tool_transitions().keys()

    def values(self):
        _warn_deprecated("DATA_ANALYSIS_TOOL_TRANSITIONS")
        return _get_deprecated_config().get_tool_transitions().values()

    def get(self, key: str, default=None):
        _warn_deprecated("DATA_ANALYSIS_TOOL_TRANSITIONS")
        return _get_deprecated_config().get_tool_transitions().get(key, default)


class _DeprecatedClusters:
    """Lazy loader for deprecated DATA_ANALYSIS_TOOL_CLUSTERS."""

    def __getitem__(self, key: str) -> Set[str]:
        _warn_deprecated("DATA_ANALYSIS_TOOL_CLUSTERS")
        return _get_deprecated_config().get_tool_clusters().get(key, set())

    def __iter__(self):
        _warn_deprecated("DATA_ANALYSIS_TOOL_CLUSTERS")
        return iter(_get_deprecated_config().get_tool_clusters())

    def items(self):
        _warn_deprecated("DATA_ANALYSIS_TOOL_CLUSTERS")
        return _get_deprecated_config().get_tool_clusters().items()

    def keys(self):
        _warn_deprecated("DATA_ANALYSIS_TOOL_CLUSTERS")
        return _get_deprecated_config().get_tool_clusters().keys()

    def values(self):
        _warn_deprecated("DATA_ANALYSIS_TOOL_CLUSTERS")
        return _get_deprecated_config().get_tool_clusters().values()

    def get(self, key: str, default=None):
        _warn_deprecated("DATA_ANALYSIS_TOOL_CLUSTERS")
        return _get_deprecated_config().get_tool_clusters().get(key, default)


class _DeprecatedSequences:
    """Lazy loader for deprecated DATA_ANALYSIS_TOOL_SEQUENCES."""

    def __getitem__(self, key: str) -> List[str]:
        _warn_deprecated("DATA_ANALYSIS_TOOL_SEQUENCES")
        return _get_deprecated_config().get_tool_sequences().get(key, [])

    def __iter__(self):
        _warn_deprecated("DATA_ANALYSIS_TOOL_SEQUENCES")
        return iter(_get_deprecated_config().get_tool_sequences())

    def items(self):
        _warn_deprecated("DATA_ANALYSIS_TOOL_SEQUENCES")
        return _get_deprecated_config().get_tool_sequences().items()

    def keys(self):
        _warn_deprecated("DATA_ANALYSIS_TOOL_SEQUENCES")
        return _get_deprecated_config().get_tool_sequences().keys()

    def values(self):
        _warn_deprecated("DATA_ANALYSIS_TOOL_SEQUENCES")
        return _get_deprecated_config().get_tool_sequences().values()

    def get(self, key: str, default=None):
        _warn_deprecated("DATA_ANALYSIS_TOOL_SEQUENCES")
        return _get_deprecated_config().get_tool_sequences().get(key, default)


class _DeprecatedDependencies:
    """Lazy loader for deprecated DATA_ANALYSIS_TOOL_DEPENDENCIES."""

    def __iter__(self):
        _warn_deprecated("DATA_ANALYSIS_TOOL_DEPENDENCIES")
        return iter(_get_deprecated_config().get_dependencies())

    def __len__(self):
        _warn_deprecated("DATA_ANALYSIS_TOOL_DEPENDENCIES")
        return len(_get_deprecated_config().get_dependencies())

    def __getitem__(self, index: int) -> "ToolDependency":
        _warn_deprecated("DATA_ANALYSIS_TOOL_DEPENDENCIES")
        return _get_deprecated_config().get_dependencies()[index]


class _DeprecatedRequiredTools:
    """Lazy loader for deprecated DATA_ANALYSIS_REQUIRED_TOOLS."""

    def __iter__(self):
        _warn_deprecated("DATA_ANALYSIS_REQUIRED_TOOLS")
        return iter(_get_deprecated_config().get_required_tools())

    def __len__(self):
        _warn_deprecated("DATA_ANALYSIS_REQUIRED_TOOLS")
        return len(_get_deprecated_config().get_required_tools())

    def __contains__(self, item):
        _warn_deprecated("DATA_ANALYSIS_REQUIRED_TOOLS")
        return item in _get_deprecated_config().get_required_tools()


class _DeprecatedOptionalTools:
    """Lazy loader for deprecated DATA_ANALYSIS_OPTIONAL_TOOLS."""

    def __iter__(self):
        _warn_deprecated("DATA_ANALYSIS_OPTIONAL_TOOLS")
        return iter(_get_deprecated_config().get_optional_tools())

    def __len__(self):
        _warn_deprecated("DATA_ANALYSIS_OPTIONAL_TOOLS")
        return len(_get_deprecated_config().get_optional_tools())

    def __contains__(self, item):
        _warn_deprecated("DATA_ANALYSIS_OPTIONAL_TOOLS")
        return item in _get_deprecated_config().get_optional_tools()


# Deprecated constants - provide backward compatibility with deprecation warnings
DATA_ANALYSIS_TOOL_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = _DeprecatedTransitions()  # type: ignore
DATA_ANALYSIS_TOOL_CLUSTERS: Dict[str, Set[str]] = _DeprecatedClusters()  # type: ignore
DATA_ANALYSIS_TOOL_SEQUENCES: Dict[str, List[str]] = _DeprecatedSequences()  # type: ignore
DATA_ANALYSIS_TOOL_DEPENDENCIES: List["ToolDependency"] = _DeprecatedDependencies()  # type: ignore
DATA_ANALYSIS_REQUIRED_TOOLS: Set[str] = _DeprecatedRequiredTools()  # type: ignore
DATA_ANALYSIS_OPTIONAL_TOOLS: Set[str] = _DeprecatedOptionalTools()  # type: ignore


__all__ = [
    "DataAnalysisToolDependencyProvider",
    "get_provider",
    # Deprecated exports for backward compatibility
    "DATA_ANALYSIS_TOOL_DEPENDENCIES",
    "DATA_ANALYSIS_TOOL_TRANSITIONS",
    "DATA_ANALYSIS_TOOL_CLUSTERS",
    "DATA_ANALYSIS_TOOL_SEQUENCES",
    "DATA_ANALYSIS_REQUIRED_TOOLS",
    "DATA_ANALYSIS_OPTIONAL_TOOLS",
]


# =============================================================================
# Entry Point Provider Factory
# =============================================================================


def get_provider() -> DataAnalysisToolDependencyProvider:
    """Entry point provider factory for dataanalysis vertical.

    This function is registered as an entry point in pyproject.toml:
        [project.entry-points."victor.tool_dependencies"]
        dataanalysis = "victor_dataanalysis.tool_dependencies:get_provider"

    Returns:
        A configured tool dependency provider for the dataanalysis vertical.

    Example:
        # Framework usage via entry points:
        from importlib.metadata import entry_points
        eps = entry_points(group="victor.tool_dependencies")
        for ep in eps:
            if ep.name == "dataanalysis":
                provider_factory = ep.load()
                provider = provider_factory()
                deps = provider.get_dependencies()
    """
    return DataAnalysisToolDependencyProvider()
