from __future__ import annotations

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

"""Framework-level tool naming utilities.

This module provides a facade for the tool naming system, exposing
utilities for canonicalizing tool names across the framework. This
ensures consistent tool naming for RL Q-values, workflow patterns,
and vertical configurations.

Design Goals:
- Unify tool naming across all verticals (coding, devops, research, data_analysis)
- Prevent RL Q-value fragmentation from inconsistent names
- Maintain backward compatibility with legacy names via aliasing
- Provide utilities for batch canonicalization of tool sets and dicts

Example:
    from victor.framework.tool_naming import (
        ToolNames,
        canonicalize_tool_set,
        canonicalize_tool_dict,
        validate_tool_names,
    )

    # Use canonical names
    tools = {ToolNames.READ, ToolNames.SHELL, ToolNames.EDIT}

    # Convert legacy names to canonical
    legacy_tools = {"read_file", "execute_bash", "edit_files"}
    canonical = canonicalize_tool_set(legacy_tools)
    # Returns: {"read", "shell", "edit"}

    # Canonicalize dictionary keys
    transitions = {"read_file": [("edit_files", 0.4)]}
    canonical_transitions = canonicalize_tool_dict(transitions)
    # Returns: {"read": [("edit_files", 0.4)]}
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

# Re-export the SDK-owned naming registry for runtime utilities.
from victor_contracts.constants import (
    CANONICAL_TO_ALIASES,
    TOOL_ALIASES,
    ToolNameEntry,
    ToolNames,
    get_aliases,
    get_all_canonical_names,
    get_canonical_name,
    get_name_mapping,
    is_valid_tool_name,
)

__all__ = [
    # Re-exports from victor.tools.tool_names
    "ToolNames",
    "ToolNameEntry",
    "TOOL_ALIASES",
    "CANONICAL_TO_ALIASES",
    "get_canonical_name",
    "get_aliases",
    "is_valid_tool_name",
    "get_all_canonical_names",
    "get_name_mapping",
    # Framework-level utilities
    "canonicalize_tool_set",
    "canonicalize_tool_dict",
    "canonicalize_tool_list",
    "canonicalize_transitions",
    "canonicalize_dependencies",
    "validate_tool_names",
    "get_legacy_names_report",
]

logger = logging.getLogger(__name__)


def canonicalize_tool_set(tools: Set[str]) -> Set[str]:
    """Convert a set of tool names to canonical form.

    Transforms all tool names in the set to their canonical (short) form,
    ensuring consistent naming for RL Q-values and workflow patterns.

    Args:
        tools: Set of tool names (may contain legacy names)

    Returns:
        Set of canonical tool names

    Example:
        >>> canonicalize_tool_set({"read_file", "execute_bash", "shell"})
        {"read", "shell"}  # Note: duplicates collapse
    """
    return {get_canonical_name(t) for t in tools}


def canonicalize_tool_dict(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize tool names in dictionary keys.

    Transforms all dictionary keys (tool names) to their canonical form.
    Values are preserved as-is.

    Args:
        mapping: Dictionary with tool names as keys

    Returns:
        New dictionary with canonical tool names as keys

    Example:
        >>> canonicalize_tool_dict({"read_file": 0.8, "execute_bash": 0.5})
        {"read": 0.8, "shell": 0.5}
    """
    return {get_canonical_name(k): v for k, v in mapping.items()}


def canonicalize_tool_list(tools: List[str]) -> List[str]:
    """Convert a list of tool names to canonical form.

    Preserves order and duplicates (unlike canonicalize_tool_set).
    Useful for tool sequences where order matters.

    Args:
        tools: List of tool names (may contain legacy names)

    Returns:
        List of canonical tool names in same order

    Example:
        >>> canonicalize_tool_list(["read_file", "edit_files", "run_tests"])
        ["read", "edit", "test"]
    """
    return [get_canonical_name(t) for t in tools]


def canonicalize_transitions(
    transitions: Dict[str, List[Tuple[str, float]]],
) -> Dict[str, List[Tuple[str, float]]]:
    """Canonicalize tool names in transition probability mappings.

    Transforms both the outer dict keys and the tool names within
    the transition tuples to canonical form.

    Args:
        transitions: Dict mapping tool names to list of (next_tool, probability) tuples

    Returns:
        Canonicalized transitions dict

    Example:
        >>> transitions = {
        ...     "read_file": [("edit_files", 0.4), ("execute_bash", 0.3)],
        ... }
        >>> canonicalize_transitions(transitions)
        {"read": [("edit", 0.4), ("shell", 0.3)]}
    """
    result: Dict[str, List[Tuple[str, float]]] = {}
    for tool, next_tools in transitions.items():
        canonical_tool = get_canonical_name(tool)
        canonical_next = [(get_canonical_name(t), p) for t, p in next_tools]
        result[canonical_tool] = canonical_next
    return result


def canonicalize_dependencies(
    dependencies: List[Any],
) -> List[Any]:
    """Canonicalize tool names in ToolDependency objects.

    Creates new ToolDependency objects with canonical tool names.
    Works with any object that has tool_name, depends_on, and enables attributes.

    Args:
        dependencies: List of ToolDependency objects

    Returns:
        List of new ToolDependency objects with canonical names

    Example:
        >>> from victor.core.tool_types import ToolDependency
        >>> deps = [ToolDependency(
        ...     tool_name="edit_files",
        ...     depends_on={"read_file"},
        ...     enables={"run_tests"},
        ...     weight=0.9,
        ... )]
        >>> canonicalize_dependencies(deps)
        [ToolDependency(tool_name="edit", depends_on={"read"}, enables={"test"}, weight=0.9)]
    """
    # Import here to avoid circular imports
    from victor.core.tool_types import ToolDependency

    result = []
    for dep in dependencies:
        result.append(
            ToolDependency(
                tool_name=get_canonical_name(dep.tool_name),
                depends_on=canonicalize_tool_set(dep.depends_on),
                enables=canonicalize_tool_set(dep.enables),
                weight=dep.weight,
            )
        )
    return result


def validate_tool_names(
    tools: Set[str] | List[str] | Dict[str, Any],
    context: str = "",
    warn: bool = True,
) -> List[str]:
    """Validate tool names and optionally warn about legacy names.

    Checks all tool names and identifies any that are legacy (non-canonical).
    Optionally logs warnings for each legacy name found.

    Args:
        tools: Set, list, or dict of tool names to validate
        context: Optional context string for warning messages (e.g., "coding vertical")
        warn: Whether to log warnings for legacy names (default True)

    Returns:
        List of legacy tool names found (empty if all canonical)

    Example:
        >>> legacy = validate_tool_names(
        ...     {"read", "execute_bash", "edit_files"},
        ...     context="coding config",
        ... )
        # Logs: "Legacy tool name 'execute_bash' in coding config, use 'shell' instead"
        # Returns: ["execute_bash", "edit_files"]
    """
    legacy_found: List[str] = []

    # Extract tool names based on input type
    if isinstance(tools, dict):
        tool_names = set(tools.keys())
    elif isinstance(tools, list):
        tool_names = set(tools)
    else:
        tool_names = tools

    for name in tool_names:
        # Check if name is an alias (legacy)
        if name in TOOL_ALIASES:
            canonical = TOOL_ALIASES[name]
            legacy_found.append(name)
            if warn:
                ctx = f" in {context}" if context else ""
                logger.warning(f"Legacy tool name '{name}'{ctx}, use '{canonical}' instead")

    return legacy_found


def get_legacy_names_report(
    tools: Set[str] | List[str] | Dict[str, Any],
) -> Dict[str, str]:
    """Get a report mapping legacy names to their canonical equivalents.

    Useful for migration and debugging. Does not log warnings.

    Args:
        tools: Set, list, or dict of tool names to analyze

    Returns:
        Dict mapping legacy names found to their canonical equivalents

    Example:
        >>> get_legacy_names_report({"read", "execute_bash", "edit_files"})
        {"execute_bash": "shell", "edit_files": "edit"}
    """
    result: Dict[str, str] = {}

    # Extract tool names based on input type
    if isinstance(tools, dict):
        tool_names = set(tools.keys())
    elif isinstance(tools, list):
        tool_names = set(tools)
    else:
        tool_names = tools

    for name in tool_names:
        if name in TOOL_ALIASES:
            result[name] = TOOL_ALIASES[name]

    return result
