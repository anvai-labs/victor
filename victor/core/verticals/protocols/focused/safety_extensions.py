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

"""Safety Extensions Protocol (ISP-Compliant).

This module provides a focused protocol for safety-related vertical extensions,
following the Interface Segregation Principle (ISP). It extracts safety-specific
concerns from the larger VerticalExtensions interface.

This protocol allows verticals to provide domain-specific safety patterns for
dangerous operations like bash commands and file operations.

Usage:
    from victor.core.verticals.protocols.focused.safety_extensions import (
        SafetyExtensionsProtocol,
    )

    class CodingSafetyExtensions(SafetyExtensionsProtocol):
        def get_safety_extensions(self) -> List[SafetyExtensionProtocol]:
            return [GitSafetyExtension(), DockerSafetyExtension()]

    # Get all safety patterns
    extensions = CodingSafetyExtensions()
    patterns = extensions.get_all_safety_patterns()
"""

from __future__ import annotations

from abc import abstractmethod
from typing import List, Protocol, runtime_checkable

from victor.core.verticals.protocols.safety_provider import (
    SafetyExtensionProtocol,
    SafetyPattern,
)


@runtime_checkable
class SafetyExtensionsProtocol(Protocol):
    """Protocol for safety-related vertical extensions.

    This focused protocol allows verticals to provide domain-specific
    safety patterns that extend the framework's core safety checker.
    Verticals can register patterns for dangerous bash commands,
    file operations, and tool-specific argument restrictions.

    Example:
        class DevOpsSafetyExtensions(SafetyExtensionsProtocol):
            def get_safety_extensions(self) -> List[SafetyExtensionProtocol]:
                return [
                    KubernetesSafetyExtension(),
                    TerraformSafetyExtension(),
                ]
    """

    @abstractmethod
    def get_safety_extensions(self) -> List[SafetyExtensionProtocol]:
        """Get list of safety extension implementations.

        Returns:
            List of safety extension protocols providing dangerous
            operation patterns for this vertical
        """
        ...

    def get_all_safety_patterns(self) -> List[SafetyPattern]:
        """Collect all safety patterns from extensions.

        Aggregates bash and file patterns from all registered
        safety extensions.

        Returns:
            Combined list of safety patterns from all extensions
        """
        patterns: List[SafetyPattern] = []
        for ext in self.get_safety_extensions():
            patterns.extend(ext.get_bash_patterns())
            patterns.extend(ext.get_file_patterns())
        return patterns


__all__ = [
    "SafetyExtensionsProtocol",
    "SafetyExtensionProtocol",
    "SafetyPattern",
]
