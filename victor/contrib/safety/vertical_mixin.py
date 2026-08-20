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

"""Vertical safety mixin with common safety rule helpers.

This module provides VerticalSafetyMixin, a mixin class with utility methods
for creating common safety patterns that verticals can use when implementing
get_vertical_rules() in their BaseSafetyExtension subclasses.
"""

from __future__ import annotations

from typing import List, Optional

from victor_contracts.safety import (
    SafetyAction,
    SafetyCategory,
    SafetyRule,
)


class VerticalSafetyMixin:
    """Mixin class providing common safety rule helpers.

    Provides utility methods for creating common safety patterns that
    verticals can use when implementing get_vertical_rules().

    This class is designed to be used as a mixin alongside BaseSafetyExtension:

        class MyVerticalSafetyExtension(BaseSafetyExtension, VerticalSafetyMixin):
            def get_vertical_name(self) -> str:
                return "myvertical"

            def get_vertical_rules(self) -> List[SafetyRule]:
                return [
                    self.create_dangerous_command_rule(
                        rule_id="myvertical_dangerous",
                        command_pattern=r"dangerous-cmd",
                        description="Dangerous command",
                    ),
                ]
    """

    @staticmethod
    def create_dangerous_command_rule(
        rule_id: str,
        command_pattern: str,
        description: str,
        severity: int = 8,
        tool_names: Optional[List[str]] = None,
        category: SafetyCategory = SafetyCategory.SHELL,
    ) -> SafetyRule:
        """Create a safety rule for dangerous shell commands.

        Args:
            rule_id: Unique rule identifier
            command_pattern: Regex pattern to match commands
            description: Human-readable description
            severity: Severity level (1-10, higher = more severe)
            tool_names: List of tools this applies to
            category: SafetyCategory for the rule

        Returns:
            SafetyRule instance
        """
        return SafetyRule(
            rule_id=rule_id,
            category=category,
            pattern=command_pattern,
            description=description,
            action=SafetyAction.REQUIRE_CONFIRMATION,
            severity=severity,
            tool_names=tool_names or ["shell", "execute_bash"],
            confirmation_prompt=f"Confirm execution of: {command_pattern}",
        )

    @staticmethod
    def create_file_deletion_rule(
        rule_id: str,
        file_pattern: str,
        description: str,
        severity: int = 7,
        tool_names: Optional[List[str]] = None,
    ) -> SafetyRule:
        """Create a safety rule for file deletion operations.

        Args:
            rule_id: Unique rule identifier
            file_pattern: Regex pattern to match file paths
            description: Human-readable description
            severity: Severity level (1-10, higher = more severe)
            tool_names: List of tools this applies to

        Returns:
            SafetyRule instance
        """
        return SafetyRule(
            rule_id=rule_id,
            category=SafetyCategory.FILE,
            pattern=file_pattern,
            description=description,
            action=SafetyAction.REQUIRE_CONFIRMATION,
            severity=severity,
            tool_names=tool_names or ["file_write", "file_delete"],
            confirmation_prompt=f"This will delete files matching: {file_pattern}",
        )

    @staticmethod
    def create_blocked_operation_rule(
        rule_id: str,
        pattern: str,
        description: str,
        tool_names: Optional[List[str]] = None,
        category: SafetyCategory = SafetyCategory.SHELL,
    ) -> SafetyRule:
        """Create a safety rule for blocked operations.

        Args:
            rule_id: Unique rule identifier
            pattern: Regex pattern to match
            description: Human-readable description
            tool_names: List of tools this applies to
            category: SafetyCategory for the rule

        Returns:
            SafetyRule instance
        """
        return SafetyRule(
            rule_id=rule_id,
            category=category,
            pattern=pattern,
            description=description,
            action=SafetyAction.BLOCK,
            severity=10,  # Maximum severity
            tool_names=tool_names or ["shell", "execute_bash"],
        )

    @staticmethod
    def create_git_force_push_rule(
        rule_id: str = "git_force_push",
        severity: int = 7,
    ) -> SafetyRule:
        """Create a safety rule for git force push operations.

        Args:
            rule_id: Unique rule identifier
            severity: Severity level (1-10, higher = more severe)

        Returns:
            SafetyRule instance
        """
        return SafetyRule(
            rule_id=rule_id,
            category=SafetyCategory.GIT,
            pattern=r"push.*--force",
            description="Force push",
            action=SafetyAction.REQUIRE_CONFIRMATION,
            severity=severity,
            tool_names=["git"],
            confirmation_prompt="Force push can rewrite history. Continue?",
        )

    @staticmethod
    def create_docker_container_deletion_rule(
        rule_id: str = "docker_rm_container",
        severity: int = 6,
    ) -> SafetyRule:
        """Create a safety rule for Docker container deletion.

        Args:
            rule_id: Unique rule identifier
            severity: Severity level (1-10, higher = more severe)

        Returns:
            SafetyRule instance
        """
        return SafetyRule(
            rule_id=rule_id,
            category=SafetyCategory.DOCKER,
            pattern=r"rm.*-f|container.*rm.*--force",
            description="Force remove Docker container",
            action=SafetyAction.WARN,
            severity=severity,
            tool_names=["docker"],
        )

    @staticmethod
    def create_system_write_rule(
        rule_id: str = "file_system_write",
        system_paths: Optional[List[str]] = None,
    ) -> SafetyRule:
        """Create a safety rule for writes to system directories.

        Args:
            rule_id: Unique rule identifier
            system_paths: List of system paths to protect (default: /etc, /usr, /bin, /sbin)

        Returns:
            SafetyRule instance
        """
        if system_paths is None:
            system_paths = ["/etc", "/usr", "/bin", "/sbin"]

        # Create pattern that matches any of the system paths
        path_pattern = "|".join(system_paths.replace("/", r"/") for system_paths in system_paths)
        pattern = rf"write.*({path_pattern})/"

        return SafetyRule(
            rule_id=rule_id,
            category=SafetyCategory.FILE,
            pattern=pattern,
            description="Write to system directory",
            action=SafetyAction.BLOCK,
            severity=10,
            tool_names=["write_file", "edit_files"],
        )


__all__ = [
    "VerticalSafetyMixin",
]
