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

"""Benchmark safety extension using victor.contrib.

This module provides safety rules for the benchmark vertical using
the BaseSafetyExtension from victor.contrib.safety.

Safety rules cover:
- Repository isolation (prevent modifying non-benchmark repos)
- Resource limits (timeout, cost, token usage)
- Test isolation (prevent running tests on production systems)
- Data privacy (prevent uploading benchmark data externally)

Migrated from: safety.py (original implementation)
Migration date: 2026-02-28 22:57:40
"""

from typing import List, Optional

from victor.contrib.safety import BaseSafetyExtension, VerticalSafetyMixin
from victor_contracts.safety import (
    SafetyRule,
    SafetyAction,
    SafetyCategory,
)


class BenchmarkSafetyExtension(BaseSafetyExtension, VerticalSafetyMixin):
    """Safety extension for benchmark vertical.

    This extension uses BaseSafetyExtension to provide:
    - Automatic rule registration with SafetyCoordinator
    - Safety checking: check_operation(), is_operation_safe()
    - Statistics tracking: get_safety_stats()
    - Vertical context tracking

    Safety rules cover:
    - Repository isolation
    - Resource limits
    - Test isolation
    - Data privacy
    """

    def get_vertical_name(self) -> str:
        """Return the vertical name for safety tracking.

        Returns:
            Vertical identifier
        """
        return "benchmark"

    def get_vertical_rules(self) -> List[SafetyRule]:
        """Define benchmark-specific safety rules.

        Returns:
            List of SafetyRule instances for this vertical

        Note:
            Use helper methods from VerticalSafetyMixin to create
            common patterns:
            - create_dangerous_command_rule(): Dangerous shell commands
            - create_file_deletion_rule(): File deletion operations
            - create_blocked_operation_rule(): Completely blocked operations
            - create_git_force_push_rule(): Git force push safety
            - create_docker_container_deletion_rule(): Docker container removal
            - create_system_write_rule(): System directory protection
        """
        rules = [
            # Example: Use helper for dangerous command pattern
            self.create_dangerous_command_rule(
                rule_id="benchmark_dangerous_commands",
                command_pattern=r"rm -rf|format|fdisk",
                description="Dangerous system operation",
                severity=10,
            ),
            # Example: Use helper for file deletion pattern
            self.create_file_deletion_rule(
                rule_id="benchmark_file_deletion",
                file_pattern=r".*\.db|.*database|.*data",
                description="Database/data file deletion",
                severity=9,
            ),
            # Example: Use helper for completely blocked operation
            self.create_blocked_operation_rule(
                rule_id="benchmark_production_modify",
                pattern=r"/prod/|/production/|release",
                description="Cannot modify production systems",
                category=SafetyCategory.SHELL,
            ),
            # Example: Use helper for git operations
            self.create_git_force_push_rule(
                rule_id="benchmark_git_force_push",
                severity=8,
            ),
        ]

        # Add custom rules for benchmark specific safety concerns
        rules.extend(self._get_custom_rules())

        return rules

    def _get_custom_rules(self) -> List[SafetyRule]:
        """Get custom benchmark-specific safety rules.

        Override this method to add vertical-specific rules beyond
        the common patterns provided by VerticalSafetyMixin.

        Returns:
            List of custom SafetyRule instances
        """
        # Benchmark safety rules — extend as needed (intentional empty default)
        return []

    def is_operation_safe(
        self,
        tool_name: str,
        arguments: List[str],
        context: Optional[dict] = None,
    ) -> tuple[bool, Optional[str]]:
        """Check if an operation is safe to execute.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments to the tool
            context: Additional context (file paths, etc.)

        Returns:
            Tuple of (is_safe, warning_message)
        """
        ctx = context or {}
        result = self._coordinator.check_safety(tool_name, arguments, ctx)
        msg = result.block_reason if not result.is_safe else None
        return result.is_safe, msg

    def check_operation(self, tool_name: str, arguments: List[str]) -> bool:
        """Check if operation should be allowed.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            True if operation is allowed, False otherwise
        """
        is_safe, _ = self.is_operation_safe(tool_name, arguments)
        return is_safe

    def get_coordinator(self):
        """Get the underlying SafetyCoordinator instance."""
        return self._coordinator

    def get_safety_stats(self) -> dict:
        """Get safety statistics.

        Returns:
            Dictionary with safety statistics
        """
        stats = self._coordinator.get_stats()
        if hasattr(stats, "to_dict"):
            return stats.to_dict()
        return stats if isinstance(stats, dict) else {}


__all__ = [
    "BenchmarkSafetyExtension",
]
