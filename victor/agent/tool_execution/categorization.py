# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tool categorization for async execution.

This module provides tool categorization to determine which tools
can be parallelized and which require sequential execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

from victor.agent.action_authorizer import (
    build_write_tool_set,
    normalize_tool_name_for_policy,
)
from victor.tools.base import AccessMode

logger = logging.getLogger(__name__)

__all__ = [
    "ToolCategory",
    "categorize_tool_call",
    "ToolCallSpec",
    "ExecutionPriority",
]


class ToolCategory(Enum):
    """Tool execution categories for parallelization decisions."""

    READ_ONLY = auto()  # File reads, searches, queries
    WRITE = auto()  # File writes, modifications
    NETWORK = auto()  # API calls, web requests
    COMPUTE = auto()  # Code execution, processing
    MIXED = auto()  # Operations with mixed effects


class ExecutionPriority(Enum):
    """Execution priority for scheduling."""

    CRITICAL = auto()  # User-facing operations (blocking)
    HIGH = auto()  # Important but not blocking
    NORMAL = auto()  # Standard operations
    LOW = auto()  # Background operations


@dataclass
class ToolCallSpec:
    """Specification for a tool call in async execution.

    Attributes:
        name: Tool name
        arguments: Tool arguments
        call_id: Unique identifier for this call
        category: Execution category
        dependencies: Set of call_ids this depends on
        priority: Execution priority (higher = more important)
        timeout: Timeout in seconds
    """

    name: str
    arguments: Dict[str, Any]
    call_id: str
    category: ToolCategory = ToolCategory.COMPUTE
    dependencies: Set[str] = field(default_factory=set)
    priority: int = 0
    # Increased from 30 to 60 seconds for semantic search operations
    timeout: float = 60.0


# Optimized tool categorization using frozenset for O(1) lookups
READ_TOOLS = frozenset(
    {
        "read",
        "ls",
        "code_search",
        "semantic_code_search",
        "grep_search",
        "plan",
        "git",
        "directory_tree",
        "file_info",
    }
)

WRITE_TOOLS = build_write_tool_set("docker", "patch", "create_file", "notebook_edit")

NETWORK_TOOLS = frozenset(
    {
        "web_search",
        "web_fetch",
        "http",
        "fetch",
    }
)

# Tool metadata for categorization
TOOL_METADATA: Dict[str, Dict[str, Any]] = {
    # File operation tools
    "read": {"category": ToolCategory.READ_ONLY, "side_effects": False},
    "ls": {"category": ToolCategory.READ_ONLY, "side_effects": False},
    "write": {"category": ToolCategory.WRITE, "side_effects": True},
    "edit": {"category": ToolCategory.WRITE, "side_effects": True},
    "shell": {"category": ToolCategory.WRITE, "side_effects": True},
    # Search tools
    "code_search": {"category": ToolCategory.READ_ONLY, "side_effects": False},
    "semantic_code_search": {"category": ToolCategory.READ_ONLY, "side_effects": False},
    "grep_search": {"category": ToolCategory.READ_ONLY, "side_effects": False},
    "plan": {"category": ToolCategory.READ_ONLY, "side_effects": False},
    # Network tools
    "web_search": {"category": ToolCategory.NETWORK, "side_effects": False},
    "web_fetch": {"category": ToolCategory.NETWORK, "side_effects": False},
    "http": {"category": ToolCategory.NETWORK, "side_effects": False},
    "fetch": {"category": ToolCategory.NETWORK, "side_effects": False},
    # Development tools
    "docker": {"category": ToolCategory.WRITE, "side_effects": True},
    "patch": {"category": ToolCategory.WRITE, "side_effects": True},
    "plan_files": {"category": ToolCategory.READ_ONLY, "side_effects": False},
}


def categorize_tool_call(tool_name: str, arguments: Dict[str, Any]) -> ToolCategory:
    """
    Categorize a tool call for execution planning.

    Args:
        tool_name: Name of the tool
        arguments: Tool arguments

    Returns:
        ToolCategory for the tool call
    """
    canonical_tool_name = normalize_tool_name_for_policy(tool_name)

    # Per-invocation narrowing: multi-command tools (shell readonly=True,
    # code grep/search) are statically write-capable but this specific call
    # may be provably read-only — which skips pre-tool checkpoints and
    # allows parallelization. Resolvers can only narrow, never widen.
    from victor.tools.effective_access import resolve_effective_access

    if resolve_effective_access(canonical_tool_name, arguments) is AccessMode.READONLY:
        return ToolCategory.READ_ONLY

    # Check metadata first
    if canonical_tool_name in TOOL_METADATA:
        return TOOL_METADATA[canonical_tool_name]["category"]

    # Fallback to frozenset checks
    if canonical_tool_name in READ_TOOLS:
        return ToolCategory.READ_ONLY
    elif canonical_tool_name in WRITE_TOOLS:
        return ToolCategory.WRITE
    elif canonical_tool_name in NETWORK_TOOLS:
        return ToolCategory.NETWORK

    # Default to COMPUTE for unknown tools
    return ToolCategory.COMPUTE


def extract_files_from_args(arguments: Dict[str, Any]) -> List[str]:
    """Extract file paths from tool arguments.

    Args:
        arguments: Tool arguments

    Returns:
        List of file paths
    """
    files = []

    for key, value in arguments.items():
        if key in ("file", "path", "file_path", "filepath", "files"):
            if isinstance(value, str):
                files.append(value)
            elif isinstance(value, list):
                files.extend(value)
        elif key == "ops" and isinstance(value, list):
            for op in value:
                if not isinstance(op, dict):
                    continue
                path = op.get("path")
                new_path = op.get("new_path")
                if isinstance(path, str):
                    files.append(path)
                if isinstance(new_path, str):
                    files.append(new_path)

    return files
