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

"""RAG Tool Dependencies - Tool relationships for RAG workflows.

This module provides RAG-specific tool dependency configuration loaded from YAML.
Extends the core YAMLToolDependencyProvider with RAG-specific data.
Also provides composed tool patterns using ToolExecutionGraph.

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.

Migration Note:
    This module has been migrated to use YAML-based configuration.
    The tool dependencies are now loaded from tool_dependencies.yaml.
    Backward compatibility is maintained - all existing exports still work.

Example:
    from victor_rag.tool_dependencies import (
        RAGToolDependencyProvider,
        get_rag_tool_graph,
        RAG_COMPOSED_PATTERNS,
    )

    # Get tool graph for RAG workflows
    graph = get_rag_tool_graph()

    # Suggest next tools after searching
    suggestions = graph.suggest_next_tools("rag_search", history=["rag_query"])

    # Plan tool sequence for document ingestion
    plan = graph.plan_for_goal({"rag_ingest", "rag_stats"})
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from victor.framework.extensions import (
    YAMLToolDependencyProvider,
    load_tool_dependency_yaml,
)
from victor_contracts.verticals.protocols import ToolDependency
from victor.tools.tool_graph import ToolExecutionGraph

# Path to the YAML configuration file
_YAML_CONFIG_PATH = Path(__file__).parent / "tool_dependencies.yaml"


class RAGToolDependencyProvider(YAMLToolDependencyProvider):
    """Tool dependency provider for RAG vertical.

    .. deprecated::
        Use ``create_vertical_tool_dependency_provider('rag')`` instead.
        This class is maintained for backward compatibility.

    Extends YAMLToolDependencyProvider to load RAG-specific tool
    relationships from tool_dependencies.yaml.

    Uses canonical ToolNames constants for consistency.

    Example:
        # Preferred (new code):
        from victor.framework.extensions import create_vertical_tool_dependency_provider
        provider = create_vertical_tool_dependency_provider("rag")

        # Deprecated (backward compatible):
        provider = RAGToolDependencyProvider()

    This class maintains backward compatibility with the previous
    hand-coded Python implementation.
    """

    def __init__(self):
        """Initialize the provider with RAG-specific config from YAML.

        .. deprecated::
            Use ``create_vertical_tool_dependency_provider('rag')`` instead.
        """
        import warnings

        warnings.warn(
            "RAGToolDependencyProvider is deprecated. "
            "Use create_vertical_tool_dependency_provider('rag') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            yaml_path=_YAML_CONFIG_PATH,
            canonicalize=True,
        )


# =============================================================================
# Backward Compatibility Exports
# =============================================================================
# These module-level exports maintain backward compatibility with code that
# imports the raw data structures directly.


def _load_config():
    """Load and cache the YAML configuration."""
    return load_tool_dependency_yaml(_YAML_CONFIG_PATH, canonicalize=True)


def _get_transitions() -> Dict[str, List[Tuple[str, float]]]:
    """Get tool transitions from YAML config."""
    return _load_config().transitions


def _get_clusters() -> Dict[str, Set[str]]:
    """Get tool clusters from YAML config."""
    return _load_config().clusters


def _get_sequences() -> Dict[str, List[str]]:
    """Get tool sequences from YAML config."""
    return _load_config().sequences


def _get_dependencies() -> List[ToolDependency]:
    """Get tool dependencies from YAML config."""
    return _load_config().dependencies


def _get_required_tools() -> Set[str]:
    """Get required tools from YAML config."""
    return _load_config().required_tools


def _get_optional_tools() -> Set[str]:
    """Get optional tools from YAML config."""
    return _load_config().optional_tools


def _get_composed_patterns() -> Dict[str, Dict[str, Any]]:
    """Get composed patterns from YAML metadata.

    The composed patterns are stored in the metadata.composed_patterns
    section of the YAML file. We need to load them from the spec directly
    since they're in metadata.
    """
    from victor.framework.extensions import ToolDependencyLoader

    loader = ToolDependencyLoader(canonicalize=False)
    spec = loader._load_and_validate(_YAML_CONFIG_PATH)

    patterns = spec.metadata.get("composed_patterns", {})

    # Convert from YAML format to the expected Python format
    result = {}
    for name, data in patterns.items():
        result[name] = {
            "description": data.get("description", ""),
            "sequence": data.get("sequence", []),
            "inputs": (
                set(data.get("inputs", []))
                if isinstance(data.get("inputs"), list)
                else data.get("inputs", set())
            ),
            "outputs": (
                set(data.get("outputs", []))
                if isinstance(data.get("outputs"), list)
                else data.get("outputs", set())
            ),
            "weight": data.get("weight", 1.0),
        }

    return result


# Backward compatibility: module-level exports accessed via __getattr__
# These constants are deprecated and will emit warnings when accessed.
_DEPRECATED_CONSTANTS = {
    "RAG_TOOL_TRANSITIONS": _get_transitions,
    "RAG_TOOL_CLUSTERS": _get_clusters,
    "RAG_TOOL_SEQUENCES": _get_sequences,
    "RAG_TOOL_DEPENDENCIES": _get_dependencies,
    "RAG_REQUIRED_TOOLS": _get_required_tools,
    "RAG_OPTIONAL_TOOLS": _get_optional_tools,
    "RAG_COMPOSED_PATTERNS": _get_composed_patterns,
}


def _warn_deprecated(name: str) -> None:
    """Emit deprecation warning for legacy constant access."""
    import warnings

    warnings.warn(
        f"{name} is deprecated. Use RAGToolDependencyProvider() or "
        f"create_vertical_tool_dependency_provider('rag') instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def __getattr__(name: str) -> Any:
    """Lazy loading of module-level exports for backward compatibility.

    This allows existing code to import the module-level constants
    while actually loading them from the YAML file.

    .. deprecated::
        These constants are deprecated. Use RAGToolDependencyProvider() or
        create_vertical_tool_dependency_provider('rag') instead.
    """
    if name in _DEPRECATED_CONSTANTS:
        _warn_deprecated(name)
        return _DEPRECATED_CONSTANTS[name]()

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# ToolExecutionGraph Factory
# =============================================================================

# Cached tool graph instance
_rag_tool_graph: Optional[ToolExecutionGraph] = None


def get_rag_tool_graph() -> ToolExecutionGraph:
    """Get the RAG tool execution graph.

    Creates a ToolExecutionGraph configured with RAG-specific
    dependencies, transitions, sequences, and composed patterns.

    Returns:
        ToolExecutionGraph for RAG workflows

    Example:
        graph = get_rag_tool_graph()

        # Suggest next tools
        suggestions = graph.suggest_next_tools("rag_search")

        # Plan tool sequence
        plan = graph.plan_for_goal({"rag_query"})

        # Validate tool execution
        valid, missing = graph.validate_execution("rag_delete", {"rag_list"})
    """
    global _rag_tool_graph

    if _rag_tool_graph is not None:
        return _rag_tool_graph

    # Load data from YAML via internal loader functions (avoids deprecation warnings)
    transitions = _get_transitions()
    clusters = _get_clusters()
    sequences = _get_sequences()
    dependencies = _get_dependencies()
    composed_patterns = _get_composed_patterns()

    graph = ToolExecutionGraph(name="rag")

    # Add dependencies
    for dep in dependencies:
        graph.add_dependency(
            dep.tool_name,
            depends_on=dep.depends_on,
            enables=dep.enables,
            weight=dep.weight,
        )

    # Add transitions
    graph.add_transitions(transitions)

    # Add sequences
    for name, sequence in sequences.items():
        graph.add_sequence(sequence, weight=0.7)

    # Add clusters
    for name, tools in clusters.items():
        graph.add_cluster(name, tools)

    # Add composed patterns as sequences with higher weights
    for pattern_name, pattern_data in composed_patterns.items():
        graph.add_sequence(pattern_data["sequence"], weight=pattern_data["weight"])

    _rag_tool_graph = graph
    return graph


def reset_rag_tool_graph() -> None:
    """Reset the cached RAG tool graph.

    Useful for testing or when tool configurations change.
    """
    global _rag_tool_graph
    _rag_tool_graph = None


def get_composed_pattern(pattern_name: str) -> Optional[Dict[str, Any]]:
    """Get a composed tool pattern by name.

    Args:
        pattern_name: Name of the pattern (e.g., "document_ingestion")

    Returns:
        Pattern configuration dict or None if not found

    Example:
        pattern = get_composed_pattern("document_ingestion")
        if pattern:
            print(f"Sequence: {pattern['sequence']}")
            print(f"Inputs: {pattern['inputs']}")
    """
    patterns = _get_composed_patterns()
    return patterns.get(pattern_name)


def list_composed_patterns() -> List[str]:
    """List all available composed tool patterns.

    Returns:
        List of pattern names
    """
    patterns = _get_composed_patterns()
    return list(patterns.keys())


__all__ = [  # noqa: F822 - constants defined via __getattr__ for lazy loading
    # Provider class
    "RAGToolDependencyProvider",
    # Data exports (lazy-loaded with deprecation warnings)
    "RAG_TOOL_DEPENDENCIES",
    "RAG_TOOL_TRANSITIONS",
    "RAG_TOOL_CLUSTERS",
    "RAG_TOOL_SEQUENCES",
    "RAG_REQUIRED_TOOLS",
    "RAG_OPTIONAL_TOOLS",
    "RAG_COMPOSED_PATTERNS",
    # Graph functions
    "get_rag_tool_graph",
    "reset_rag_tool_graph",
    "get_composed_pattern",
    "list_composed_patterns",
]
