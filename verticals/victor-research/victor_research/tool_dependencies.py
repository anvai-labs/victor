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

"""Research Tool Dependencies - Tool relationships for research workflows.

Migrated to YAML-based configuration for declarative tool dependency management.
Configuration is loaded from tool_dependencies.yaml in this directory.

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.

The YAML-based approach provides:
- Declarative configuration that's easier to read and modify
- Consistent schema validation via Pydantic
- Automatic tool name canonicalization
- Caching for performance

For backward compatibility, the legacy constants are preserved and derived
from the YAML configuration at import time.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple

from victor_contracts.verticals.tool_dependencies import (
    YAMLToolDependencyProvider,
    load_tool_dependency_yaml,
)
from victor_contracts.verticals.protocols import ToolDependency

# Path to the YAML configuration file
_YAML_PATH = Path(__file__).parent / "tool_dependencies.yaml"


class ResearchToolDependencyProvider(YAMLToolDependencyProvider):
    """Tool dependency provider for research vertical.

    .. deprecated::
        Use ``create_vertical_tool_dependency_provider('research')`` instead.
        This class is maintained for backward compatibility.

    Extends YAMLToolDependencyProvider to load research-specific tool
    relationships from tool_dependencies.yaml.

    Provides configurations for:
    - Fact-checking workflows
    - Literature review workflows
    - Technical lookup workflows
    - Report writing workflows

    Uses tool names from ToolNames constants for consistency.

    Example:
        # Preferred (new code):
        from victor.framework.extensions import create_vertical_tool_dependency_provider
        provider = create_vertical_tool_dependency_provider("research")

        # Deprecated (backward compatible):
        provider = ResearchToolDependencyProvider()

    Note: canonicalize=False is used to preserve the original tool names
    as defined in the YAML (e.g., code_search vs grep). The tool names
    in the YAML match the ToolNames constants used in the original Python.
    """

    def __init__(self):
        """Initialize the provider from YAML configuration.

        .. deprecated::
            Use ``create_vertical_tool_dependency_provider('research')`` instead.
        """
        import warnings

        warnings.warn(
            "ResearchToolDependencyProvider is deprecated. "
            "Use create_vertical_tool_dependency_provider('research') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            yaml_path=_YAML_PATH,
            canonicalize=False,  # Preserve original tool names from ToolNames constants
        )


# =============================================================================
# BACKWARD COMPATIBILITY: Legacy constants derived from YAML
# =============================================================================
# These constants are provided for backward compatibility with code that
# imports them directly. They are derived from the YAML configuration.


def _get_legacy_config() -> Dict:
    """Load config for legacy constant initialization."""
    config = load_tool_dependency_yaml(_YAML_PATH, canonicalize=True)
    return config


# Lazy initialization to avoid circular imports
_legacy_config = None


def _ensure_legacy_config():
    """Ensure legacy config is loaded."""
    global _legacy_config
    if _legacy_config is None:
        _legacy_config = _get_legacy_config()
    return _legacy_config


# Legacy constants - these are properties that load from YAML on first access
# For import compatibility, we define them at module level after loading

# Load the configuration at module import time to populate legacy constants
try:
    _config = load_tool_dependency_yaml(_YAML_PATH, canonicalize=False)

    # Tool dependency graph for research workflows
    RESEARCH_TOOL_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = _config.transitions

    # Tools that should typically be used together
    RESEARCH_TOOL_CLUSTERS: Dict[str, Set[str]] = _config.clusters

    # Recommended tool sequences for common research patterns
    RESEARCH_TOOL_SEQUENCES: Dict[str, List[str]] = _config.sequences

    # Tool dependencies for research
    RESEARCH_TOOL_DEPENDENCIES: List[ToolDependency] = _config.dependencies

    # Required tools for research
    RESEARCH_REQUIRED_TOOLS: Set[str] = _config.required_tools

    # Optional tools that enhance research
    RESEARCH_OPTIONAL_TOOLS: Set[str] = _config.optional_tools

except Exception:
    # Fallback to empty values if YAML loading fails during import
    # This allows the module to be imported even if YAML is malformed
    RESEARCH_TOOL_TRANSITIONS = {}
    RESEARCH_TOOL_CLUSTERS = {}
    RESEARCH_TOOL_SEQUENCES = {}
    RESEARCH_TOOL_DEPENDENCIES = []
    RESEARCH_REQUIRED_TOOLS = set()
    RESEARCH_OPTIONAL_TOOLS = set()


__all__ = [
    "ResearchToolDependencyProvider",
    "get_provider",
    # Legacy constants for backward compatibility
    "RESEARCH_TOOL_DEPENDENCIES",
    "RESEARCH_TOOL_TRANSITIONS",
    "RESEARCH_TOOL_CLUSTERS",
    "RESEARCH_TOOL_SEQUENCES",
    "RESEARCH_REQUIRED_TOOLS",
    "RESEARCH_OPTIONAL_TOOLS",
]


# =============================================================================
# Entry Point Provider Factory
# =============================================================================


def get_provider() -> ResearchToolDependencyProvider:
    """Entry point provider factory for research vertical.

    This function is registered as an entry point in pyproject.toml:
        [project.entry-points."victor.tool_dependencies"]
        research = "victor_research.tool_dependencies:get_provider"

    Returns:
        A configured tool dependency provider for the research vertical.

    Example:
        # Framework usage via entry points:
        from importlib.metadata import entry_points
        eps = entry_points(group="victor.tool_dependencies")
        for ep in eps:
            if ep.name == "research":
                provider_factory = ep.load()
                provider = provider_factory()
                deps = provider.get_dependencies()
    """
    return ResearchToolDependencyProvider()
