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

"""Safety Provider Protocols (ISP: Interface Segregation Principle).

This module contains protocols specifically for safety pattern management.
Following ISP, these protocols are focused on a single responsibility:
defining and providing dangerous operation patterns.

Usage:
    from victor.core.verticals.protocols.safety_provider import (
        SafetyExtensionProtocol,
    )

    class GitSafetyExtension(SafetyExtensionProtocol):
        def get_bash_patterns(self) -> List[SafetyPattern]:
            return [
                SafetyPattern(
                    pattern=r"git\\s+reset\\s+--hard",
                    description="Discard uncommitted changes",
                    risk_level="HIGH",
                    category="git",
                ),
            ]
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Protocol, runtime_checkable

from victor.security.safety.types import SafetyPattern

# =============================================================================
# Safety Extension Protocol
# =============================================================================


@runtime_checkable
class SafetyExtensionProtocol(Protocol):
    r"""Protocol for vertical-specific safety patterns.

    Extends the framework's core safety checker with domain-specific
    dangerous operation patterns.

    Example:
        class GitSafetyExtension(SafetyExtensionProtocol):
            def get_bash_patterns(self) -> List[SafetyPattern]:
                return [
                    SafetyPattern(
                        pattern=r"git\s+reset\s+--hard",
                        description="Discard uncommitted changes",
                        risk_level="HIGH",
                        category="git",
                    ),
                    SafetyPattern(
                        pattern=r"git\s+push\s+.*--force",
                        description="Force push (may lose commits)",
                        risk_level="HIGH",
                        category="git",
                    ),
                ]
    """

    @abstractmethod
    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Get bash command patterns for this vertical.

        Returns:
            List of safety patterns for dangerous bash commands
        """
        ...

    def get_file_patterns(self) -> List[SafetyPattern]:
        """Get file operation patterns for this vertical.

        Returns:
            List of safety patterns for dangerous file operations
        """
        return []

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Get tool-specific argument restrictions.

        Returns:
            Dict mapping tool names to list of restricted argument patterns
        """
        return {}

    def get_category(self) -> str:
        """Get the category name for these patterns.

        Returns:
            Category identifier (e.g., "coding", "devops")
        """
        return "custom"


__all__ = [
    "SafetyExtensionProtocol",
    "SafetyPattern",
]
