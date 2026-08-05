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

"""Default safety extension with common dangerous-operation patterns.

Provides a ``DefaultSafetyExtension`` that implements
``SafetyExtensionProtocol`` with patterns every vertical should block
(force-push, ``rm -rf /``, credential files, etc.).  External verticals
inherit from this class and call ``super()`` to get baseline coverage::

    from victor.framework.defaults import DefaultSafetyExtension

    class InvestSafetyExtension(DefaultSafetyExtension):
        def get_bash_patterns(self):
            patterns = super().get_bash_patterns()
            patterns.extend(INVEST_PATTERNS)
            return patterns
"""

from __future__ import annotations

from typing import Dict, List

from victor.security.safety.types import SafetyPattern


class DefaultSafetyExtension:
    """Default safety extension with common dangerous-operation patterns.

    Implements ``SafetyExtensionProtocol`` with baseline patterns for git,
    filesystem, credential, and system-file operations that are dangerous
    regardless of domain.
    """

    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Get common dangerous bash command patterns.

        Returns:
            Patterns covering force-push, hard reset, root deletion,
            wildcard deletion, clean untracked, dangerous permissions,
            and raw disk writes.
        """
        return [
            SafetyPattern(
                pattern=r"git\s+push\s+.*--force",
                description="Force push (may lose remote commits)",
                risk_level="HIGH",
                category="git",
            ),
            SafetyPattern(
                pattern=r"git\s+reset\s+--hard",
                description="Hard reset (discards uncommitted changes)",
                risk_level="HIGH",
                category="git",
            ),
            SafetyPattern(
                pattern=r"rm\s+-rf\s+/",
                description="Recursive delete from root",
                risk_level="CRITICAL",
                category="filesystem",
            ),
            SafetyPattern(
                pattern=r"rm\s+-rf\s+\*",
                description="Recursive wildcard deletion",
                risk_level="HIGH",
                category="filesystem",
            ),
            SafetyPattern(
                pattern=r"git\s+clean\s+-fdx",
                description="Clean all untracked files and directories",
                risk_level="HIGH",
                category="git",
            ),
            SafetyPattern(
                pattern=r"chmod\s+-R\s+777",
                description="Recursively set world-writable permissions",
                risk_level="HIGH",
                category="filesystem",
            ),
            SafetyPattern(
                pattern=r"dd\s+.*of=/dev/",
                description="Raw disk write",
                risk_level="CRITICAL",
                category="filesystem",
            ),
        ]

    def get_file_patterns(self) -> List[SafetyPattern]:
        """Get common dangerous file-operation patterns.

        Returns:
            Patterns covering environment files, certificates/keys,
            and critical system files.
        """
        return [
            SafetyPattern(
                pattern=r"\.env$",
                description="Environment file (may contain secrets)",
                risk_level="HIGH",
                category="credentials",
            ),
            SafetyPattern(
                pattern=r"\.pem$|\.key$|\.crt$",
                description="Certificate or key file",
                risk_level="HIGH",
                category="credentials",
            ),
            SafetyPattern(
                pattern=r"/etc/passwd|/etc/shadow",
                description="System authentication file",
                risk_level="CRITICAL",
                category="system",
            ),
        ]

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Get tool-specific argument restrictions.

        Returns:
            Restrictions for ``write`` (credential paths) and
            ``shell`` (recursive deletion).
        """
        return {
            "write": [r"\.env", r"\.git/", r"secrets", r"credentials"],
            "shell": [r"rm -rf \."],
        }

    def get_category(self) -> str:
        """Get the category name for these patterns.

        Returns:
            ``"default"``
        """
        return "default"


__all__ = [
    "DefaultSafetyExtension",
]
