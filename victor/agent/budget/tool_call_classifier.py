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

"""Tool call classification.

This module provides ToolCallClassifier, which handles tool operation
classification. Extracted from BudgetManager to follow the Single
Responsibility Principle (SRP) and Open/Closed Principle (OCP).

Part of SOLID-based refactoring to eliminate god class anti-pattern.
"""

import logging
from enum import Enum
from typing import Optional, Set

from victor.agent.action_authorizer import (
    build_write_tool_set,
    normalize_tool_name_for_policy,
)
from victor.agent.protocols import IToolCallClassifier

logger = logging.getLogger(__name__)


class OperationType(str, Enum):
    """Tool operation type classification."""

    WRITE = "write"  # Write/modify operations
    READ = "read"  # Read-only operations
    UNKNOWN = "unknown"  # Unclassified operations


# Default set of write tools (can be extended via OCP)
DEFAULT_WRITE_TOOLS: Set[str] = build_write_tool_set(
    "git_commit",
    "git_push",
    "delete_file",
    "create_directory",
    "mkdir",
    "create_file",
    "rename_file",
    "notebook_edit",
)


class ToolCallClassifier(IToolCallClassifier):
    """Classifies tool calls by operation type.

    This class is responsible for:
    - Classifying tools as write operations vs read operations
    - Determining tool operation types
    - Supporting extensibility for new tool types (OCP compliance)

    SRP Compliance: Focuses only on tool classification, delegating
    budget tracking, multiplier calculation, and mode completion to
    specialized components.

    OCP Compliance: Open for extension (can add new tools via
    add_write_tool), closed for modification (default classification
    logic doesn't need to change).

    Attributes:
        _write_tools: Set of tool names considered write operations
    """

    def __init__(self, write_tools: Optional[Set[str]] = None):
        """Initialize the tool call classifier.

        Args:
            write_tools: Optional set of write tool names (uses default if not provided)
        """
        self._write_tools: Set[str] = write_tools or DEFAULT_WRITE_TOOLS

    def is_write_operation(self, tool_name: str) -> bool:
        """Check if a tool is a write/action operation.

        Args:
            tool_name: Name of the tool

        Returns:
            True if this is a write/modify operation
        """
        return normalize_tool_name_for_policy(tool_name.lower()) in self._write_tools

    def classify_operation(self, tool_name: str) -> OperationType:
        """Classify tool operation type.

        Args:
            tool_name: Name of the tool

        Returns:
            Operation type enum value
        """
        tool_lower = normalize_tool_name_for_policy(tool_name.lower())

        if tool_lower in self._write_tools:
            return OperationType.WRITE
        elif tool_lower in {"read", "grep", "search"}:
            return OperationType.READ
        else:
            return OperationType.UNKNOWN

    def add_write_tool(self, tool_name: str) -> None:
        """Add a tool to the write operations set (OCP compliance).

        This allows extending the classifier without modifying existing code.

        Args:
            tool_name: Name of the tool to add
        """
        self._write_tools = set(self._write_tools)  # Convert from frozenset if needed
        self._write_tools.add(normalize_tool_name_for_policy(tool_name.lower()))
        logger.debug(f"ToolCallClassifier: added write tool '{tool_name}'")

    def remove_write_tool(self, tool_name: str) -> None:
        """Remove a tool from the write operations set.

        Args:
            tool_name: Name of the tool to remove
        """
        canonical_tool_name = normalize_tool_name_for_policy(tool_name.lower())
        if canonical_tool_name in self._write_tools:
            self._write_tools = set(self._write_tools)  # Convert from frozenset if needed
            self._write_tools.discard(canonical_tool_name)
            logger.debug(f"ToolCallClassifier: removed write tool '{tool_name}'")

    def get_write_tools(self) -> Set[str]:
        """Get the current set of write tools.

        Returns:
            Set of tool names considered write operations
        """
        return self._write_tools.copy()
