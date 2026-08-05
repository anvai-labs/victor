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

"""Safety context for tracking vertical-specific safety information.

This module provides SafetyContext, a dataclass for tracking safety-related
context within a vertical. It provides tracking of operations, statistics,
and vertical metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SafetyOperation:
    """A single safety operation record.

    Attributes:
        tool_name: Tool that was called
        args: Arguments passed to the tool
        result: Safety check result
        timestamp: When the operation occurred
    """

    tool_name: str
    args: List[str]
    result: Any
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "args": self.args,
            "result": (
                self.result.to_dict() if hasattr(self.result, "to_dict") else str(self.result)
            ),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SafetyContext:
    """Safety context for verticals.

    Tracks safety-related information for a vertical, including:
    - Vertical name and metadata
    - Operations performed
    - Safety statistics
    - Last operation timestamp

    Attributes:
        vertical_name: Name of the vertical
        operations: List of operations performed
        metadata: Additional metadata about the vertical
    """

    vertical_name: str
    operations: List[SafetyOperation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def track_operation(self, tool_name: str, args: List[str], result: Any) -> None:
        """Track an operation in the safety context.

        Args:
            tool_name: Tool that was called
            args: Arguments passed to the tool
            result: Safety check result
        """
        operation = SafetyOperation(tool_name=tool_name, args=args, result=result)
        self.operations.append(operation)

        # Keep only last 100 operations to prevent unbounded growth
        if len(self.operations) > 100:
            self.operations = self.operations[-100:]

        logger.debug(
            f"Tracked safety operation for '{self.vertical_name}': "
            f"{tool_name} with {len(args)} args"
        )

    def get_recent_operations(self, limit: int = 10) -> List[SafetyOperation]:
        """Get recent operations.

        Args:
            limit: Maximum number of operations to return

        Returns:
            List of recent operations
        """
        return self.operations[-limit:]

    def get_operation_count(self) -> int:
        """Get total number of operations tracked.

        Returns:
            Number of operations
        """
        return len(self.operations)

    def clear_operations(self) -> None:
        """Clear all tracked operations."""
        self.operations.clear()
        logger.debug(f"Cleared operations for '{self.vertical_name}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary.

        Returns:
            Dictionary representation of the context
        """
        return {
            "vertical_name": self.vertical_name,
            "operation_count": len(self.operations),
            "recent_operations": [op.to_dict() for op in self.get_recent_operations(5)],
            "metadata": self.metadata.copy(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked operations.

        Returns:
            Dictionary with operation statistics
        """
        if not self.operations:
            return {
                "total_operations": 0,
                "tools_used": [],
                "last_operation": None,
            }

        tool_counts: Dict[str, int] = {}
        for op in self.operations:
            tool_counts[op.tool_name] = tool_counts.get(op.tool_name, 0) + 1

        last_op = self.operations[-1] if self.operations else None

        return {
            "total_operations": len(self.operations),
            "tools_used": sorted(tool_counts.items(), key=lambda x: x[1], reverse=True),
            "last_operation": last_op.to_dict() if last_op else None,
        }


__all__ = [
    "SafetyContext",
    "SafetyOperation",
]
