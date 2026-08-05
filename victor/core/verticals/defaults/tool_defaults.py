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

"""Default tool configurations shared across verticals.

Provides common tool dependencies, clusters, and transitions that are
applicable to most or all verticals. Verticals can import and extend these
rather than re-defining common patterns.

Uses canonical tool names from ToolNames for consistency.
"""

from typing import Dict, List, Set, Tuple

from victor.core.tool_types import ToolDependency
from victor.framework.tool_naming import ToolNames

# =============================================================================
# Common Tool Clusters
# =============================================================================

# Tool clusters that are common across most verticals
COMMON_TOOL_CLUSTERS: Dict[str, Set[str]] = {
    # File operations - applicable to all verticals
    "file_operations": {
        ToolNames.READ,
        ToolNames.WRITE,
        ToolNames.EDIT,
        ToolNames.LS,
    },
    # Search operations - applicable to all verticals
    "search_operations": {
        ToolNames.GREP,
        ToolNames.CODE_SEARCH,
    },
    # Web operations - applicable to research, RAG
    "web_operations": {
        ToolNames.WEB_SEARCH,
        ToolNames.WEB_FETCH,
    },
}


# =============================================================================
# Common Tool Dependencies
# =============================================================================

# Tool dependencies that are common across most verticals
COMMON_TOOL_DEPENDENCIES: List[ToolDependency] = [
    # Edit should be preceded by read (universal pattern)
    ToolDependency(
        tool_name=ToolNames.EDIT,
        depends_on={ToolNames.READ},
        enables={ToolNames.TEST, ToolNames.GIT},
        weight=0.9,
    ),
    # Write typically follows read or ls
    ToolDependency(
        tool_name=ToolNames.WRITE,
        depends_on={ToolNames.READ, ToolNames.LS},
        enables={ToolNames.TEST, ToolNames.GIT},
        weight=0.8,
    ),
    # Search enables read (universal pattern)
    ToolDependency(
        tool_name=ToolNames.GREP,
        depends_on=set(),
        enables={ToolNames.READ},
        weight=0.8,
    ),
    ToolDependency(
        tool_name=ToolNames.CODE_SEARCH,
        depends_on=set(),
        enables={ToolNames.READ},
        weight=0.8,
    ),
    # List enables read (universal pattern)
    ToolDependency(
        tool_name=ToolNames.LS,
        depends_on=set(),
        enables={ToolNames.READ, ToolNames.WRITE},
        weight=0.7,
    ),
    # Read is a root operation
    ToolDependency(
        tool_name=ToolNames.READ,
        depends_on=set(),
        enables={ToolNames.EDIT, ToolNames.WRITE},
        weight=1.0,
    ),
]


# =============================================================================
# Common Tool Transitions
# =============================================================================

# Transition probabilities that are common across most verticals
COMMON_TOOL_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = {
    # Read typically leads to edit, search, or write
    ToolNames.READ: [
        (ToolNames.EDIT, 0.4),
        (ToolNames.GREP, 0.3),
        (ToolNames.WRITE, 0.2),
        (ToolNames.LS, 0.1),
    ],
    # Search leads to read
    ToolNames.GREP: [
        (ToolNames.READ, 0.6),
        (ToolNames.CODE_SEARCH, 0.2),
        (ToolNames.LS, 0.2),
    ],
    ToolNames.CODE_SEARCH: [
        (ToolNames.READ, 0.6),
        (ToolNames.GREP, 0.2),
        (ToolNames.LS, 0.2),
    ],
    # List leads to read or search
    ToolNames.LS: [
        (ToolNames.READ, 0.5),
        (ToolNames.GREP, 0.3),
        (ToolNames.WRITE, 0.2),
    ],
    # Edit leads back to read or forward to test
    ToolNames.EDIT: [
        (ToolNames.READ, 0.4),
        (ToolNames.EDIT, 0.3),
        (ToolNames.LS, 0.3),
    ],
    # Write leads to read or list
    ToolNames.WRITE: [
        (ToolNames.READ, 0.5),
        (ToolNames.LS, 0.3),
        (ToolNames.EDIT, 0.2),
    ],
}


# =============================================================================
# Common Tool Sets
# =============================================================================

# Tools required by most verticals
COMMON_REQUIRED_TOOLS: Set[str] = {
    ToolNames.READ,
    ToolNames.WRITE,
    ToolNames.EDIT,
    ToolNames.LS,
}

# Optional tools applicable to most verticals
COMMON_OPTIONAL_TOOLS: Set[str] = {
    ToolNames.GREP,
    ToolNames.CODE_SEARCH,
    ToolNames.SYMBOL,
}


# =============================================================================
# Merge Utilities
# =============================================================================


def merge_clusters(
    base: Dict[str, Set[str]],
    override: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    """Merge tool cluster definitions.

    Combines cluster definitions, unioning sets for the same cluster name.

    Args:
        base: Base cluster definitions
        override: Override cluster definitions

    Returns:
        Merged cluster definitions
    """
    result = {k: set(v) for k, v in base.items()}

    for name, tools in override.items():
        if name in result:
            result[name] = result[name] | tools
        else:
            result[name] = set(tools)

    return result


def merge_dependencies(
    base: List[ToolDependency],
    override: List[ToolDependency],
) -> List[ToolDependency]:
    """Merge tool dependency definitions.

    Combines dependency lists, with override taking precedence for same tool.

    Args:
        base: Base dependency definitions
        override: Override dependency definitions

    Returns:
        Merged dependency list
    """
    # Index by tool name
    by_tool: Dict[str, ToolDependency] = {dep.tool_name: dep for dep in base}

    # Override with vertical-specific
    for dep in override:
        by_tool[dep.tool_name] = dep

    return list(by_tool.values())


def merge_transitions(
    base: Dict[str, List[Tuple[str, float]]],
    override: Dict[str, List[Tuple[str, float]]],
) -> Dict[str, List[Tuple[str, float]]]:
    """Merge tool transition definitions.

    Override completely replaces base for same source tool.

    Args:
        base: Base transition definitions
        override: Override transition definitions

    Returns:
        Merged transition definitions
    """
    result = {k: list(v) for k, v in base.items()}
    result.update(override)
    return result


__all__ = [
    "COMMON_TOOL_CLUSTERS",
    "COMMON_TOOL_DEPENDENCIES",
    "COMMON_TOOL_TRANSITIONS",
    "COMMON_REQUIRED_TOOLS",
    "COMMON_OPTIONAL_TOOLS",
    "merge_clusters",
    "merge_dependencies",
    "merge_transitions",
]
