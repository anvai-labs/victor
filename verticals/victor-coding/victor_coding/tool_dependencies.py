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

"""Coding-specific tool dependencies and sequences.

This module defines tool execution patterns and transition probabilities
for intelligent tool selection in software development tasks.

YAML-Based Configuration (Phase 2):
    Tool dependency configuration has been migrated from hand-coded Python
    dictionaries to a YAML file at `victor/coding/tool_dependencies.yaml`.
    This enables:
    - Easier maintenance and modification of tool relationships
    - Consistent schema validation via Pydantic models
    - Reuse of the core YAML loading infrastructure

Backward Compatibility:
    - CodingToolDependencyProvider class is preserved with the same interface
    - Legacy constants (CODING_TOOL_DEPENDENCIES, etc.) are deprecated but
      still available for existing code that imports them directly
    - All existing code using CodingToolDependencyProvider continues to work

Migration Path:
    Before (hand-coded Python):
        from victor_coding.tool_dependencies import CodingToolDependencyProvider
        provider = CodingToolDependencyProvider()

    After (YAML-based, same interface):
        from victor_coding.tool_dependencies import CodingToolDependencyProvider
        provider = CodingToolDependencyProvider()  # Now loads from YAML

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from victor_contracts.verticals.tool_dependencies import (
    YAMLToolDependencyProvider,
    load_tool_dependency_yaml,
)
from victor_contracts.verticals.tool_dependencies import ToolDependencyConfig
from victor_contracts.verticals.protocols import ToolDependency

logger = logging.getLogger(__name__)

# Path to the YAML configuration file
_YAML_CONFIG_PATH = Path(__file__).parent / "tool_dependencies.yaml"


class CodingToolDependencyProvider(YAMLToolDependencyProvider):
    """Tool dependency provider for coding vertical.

    .. deprecated::
        Use ``create_vertical_tool_dependency_provider('coding')`` instead.
        This class is maintained for backward compatibility.

    Extends YAMLToolDependencyProvider with coding-specific tool
    relationships, transitions, and sequences loaded from YAML.

    Provides tool execution patterns that improve intelligent tool
    selection for software development tasks.

    Configuration is loaded from:
        victor/coding/tool_dependencies.yaml

    Example:
        # Preferred (new code):
        from victor.framework.extensions import create_vertical_tool_dependency_provider
        provider = create_vertical_tool_dependency_provider("coding")

        # Deprecated (backward compatible):
        provider = CodingToolDependencyProvider()

        # Get tool dependencies
        deps = provider.get_dependencies()

        # Get recommended sequence for a task type
        sequence = provider.get_recommended_sequence("edit")

        # Get transition weight between tools
        weight = provider.get_transition_weight("read", "edit")

    Migration Notes:
        This class previously used hand-coded Python dictionaries. It has
        been migrated to use YAML configuration while maintaining full
        backward compatibility with the existing interface. For new code,
        use create_vertical_tool_dependency_provider('coding') instead.
    """

    def __init__(
        self,
        additional_dependencies: Optional[List[ToolDependency]] = None,
        additional_sequences: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize the provider.

        Args:
            additional_dependencies: Additional tool dependencies to merge
            additional_sequences: Additional tool sequences to merge

        .. deprecated::
            Use ``create_vertical_tool_dependency_provider('coding')`` instead.
        """
        warnings.warn(
            "CodingToolDependencyProvider is deprecated. "
            "Use create_vertical_tool_dependency_provider('coding') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            yaml_path=_YAML_CONFIG_PATH,
            canonicalize=True,
            additional_dependencies=additional_dependencies,
            additional_sequences=additional_sequences,
        )


def _load_yaml_config() -> ToolDependencyConfig:
    """Load the YAML configuration for lazy initialization of legacy constants.

    Returns:
        ToolDependencyConfig loaded from YAML
    """
    return load_tool_dependency_yaml(_YAML_CONFIG_PATH, canonicalize=True)


# =============================================================================
# Deprecated Legacy Constants (for backward compatibility)
# =============================================================================
# These constants are DEPRECATED and will be removed in a future version.
# Use CodingToolDependencyProvider instead.
#
# WARNING: These are lazily initialized from YAML on first access.
# For new code, use CodingToolDependencyProvider().get_*() methods instead.


class _LazyDeprecatedProperty:
    """Descriptor for lazy initialization of deprecated constants."""

    def __init__(self, attr_name: str, extractor_name: str, deprecation_msg: str):
        self.attr_name = attr_name
        self.extractor_name = extractor_name
        self.deprecation_msg = deprecation_msg
        self._cached_value = None
        self._loaded = False

    def __get__(self, obj, objtype=None):
        if not self._loaded:
            warnings.warn(
                self.deprecation_msg,
                DeprecationWarning,
                stacklevel=2,
            )
            config = _load_yaml_config()
            extractor = getattr(self, f"_extract_{self.extractor_name}")
            self._cached_value = extractor(config)
            self._loaded = True
        return self._cached_value

    def _extract_dependencies(self, config: ToolDependencyConfig) -> List[ToolDependency]:
        return config.dependencies

    def _extract_transitions(
        self, config: ToolDependencyConfig
    ) -> Dict[str, List[Tuple[str, float]]]:
        return config.transitions

    def _extract_clusters(self, config: ToolDependencyConfig) -> Dict[str, Set[str]]:
        return config.clusters

    def _extract_sequences(self, config: ToolDependencyConfig) -> Dict[str, List[str]]:
        return config.sequences

    def _extract_required_tools(self, config: ToolDependencyConfig) -> Set[str]:
        return config.required_tools

    def _extract_optional_tools(self, config: ToolDependencyConfig) -> Set[str]:
        return config.optional_tools


class _DeprecatedConstants:
    """Container for deprecated legacy constants.

    Access any attribute to trigger lazy loading and deprecation warning.
    """

    CODING_TOOL_DEPENDENCIES = _LazyDeprecatedProperty(
        "CODING_TOOL_DEPENDENCIES",
        "dependencies",
        "CODING_TOOL_DEPENDENCIES is deprecated. "
        "Use CodingToolDependencyProvider().get_dependencies() instead.",
    )

    CODING_TOOL_TRANSITIONS = _LazyDeprecatedProperty(
        "CODING_TOOL_TRANSITIONS",
        "transitions",
        "CODING_TOOL_TRANSITIONS is deprecated. "
        "Use CodingToolDependencyProvider().get_tool_transitions() instead.",
    )

    CODING_TOOL_CLUSTERS = _LazyDeprecatedProperty(
        "CODING_TOOL_CLUSTERS",
        "clusters",
        "CODING_TOOL_CLUSTERS is deprecated. "
        "Use CodingToolDependencyProvider().get_tool_clusters() instead.",
    )

    CODING_TOOL_SEQUENCES = _LazyDeprecatedProperty(
        "CODING_TOOL_SEQUENCES",
        "sequences",
        "CODING_TOOL_SEQUENCES is deprecated. "
        "Use CodingToolDependencyProvider().get_tool_sequences() instead.",
    )

    CODING_REQUIRED_TOOLS = _LazyDeprecatedProperty(
        "CODING_REQUIRED_TOOLS",
        "required_tools",
        "CODING_REQUIRED_TOOLS is deprecated. "
        "Use CodingToolDependencyProvider().get_required_tools() instead.",
    )

    CODING_OPTIONAL_TOOLS = _LazyDeprecatedProperty(
        "CODING_OPTIONAL_TOOLS",
        "optional_tools",
        "CODING_OPTIONAL_TOOLS is deprecated. "
        "Use CodingToolDependencyProvider().get_optional_tools() instead.",
    )


# Instantiate the container for module-level access
_deprecated = _DeprecatedConstants()


# Module-level aliases that trigger deprecation warnings on access
def __getattr__(name: str):
    """Module-level __getattr__ to provide deprecated constant access.

    This allows existing code that imports the constants directly to continue
    working while receiving deprecation warnings.
    """
    deprecated_map = {
        "CODING_TOOL_DEPENDENCIES": "CODING_TOOL_DEPENDENCIES",
        "CODING_TOOL_TRANSITIONS": "CODING_TOOL_TRANSITIONS",
        "CODING_TOOL_CLUSTERS": "CODING_TOOL_CLUSTERS",
        "CODING_TOOL_SEQUENCES": "CODING_TOOL_SEQUENCES",
        "CODING_REQUIRED_TOOLS": "CODING_REQUIRED_TOOLS",
        "CODING_OPTIONAL_TOOLS": "CODING_OPTIONAL_TOOLS",
    }

    if name in deprecated_map:
        return getattr(_deprecated, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CodingToolDependencyProvider",
    "get_provider",
    # Deprecated constants (for backward compatibility, accessed via __getattr__)
    "CODING_TOOL_DEPENDENCIES",  # noqa: F822
    "CODING_TOOL_SEQUENCES",  # noqa: F822
    "CODING_TOOL_TRANSITIONS",  # noqa: F822
    "CODING_TOOL_CLUSTERS",  # noqa: F822
    "CODING_REQUIRED_TOOLS",  # noqa: F822
    "CODING_OPTIONAL_TOOLS",  # noqa: F822
]


# =============================================================================
# Entry Point Provider Factory
# =============================================================================


def get_provider() -> YAMLToolDependencyProvider:
    """Entry point provider factory for coding vertical.

    This function is registered as an entry point in pyproject.toml:
        [project.entry-points."victor.tool_dependencies"]
        coding = "victor_coding.tool_dependencies:get_provider"

    Returns:
        A configured tool dependency provider for the coding vertical.

    Example:
        # Framework usage via entry points:
        from importlib.metadata import entry_points
        eps = entry_points(group="victor.tool_dependencies")
        for ep in eps:
            if ep.name == "coding":
                provider_factory = ep.load()
                provider = provider_factory()
                deps = provider.get_dependencies()
    """
    return YAMLToolDependencyProvider(
        yaml_path=_YAML_CONFIG_PATH,
        canonicalize=True,
    )
