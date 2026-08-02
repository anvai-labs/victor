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

"""Enhanced safety integration for victor-coding using SafetyCoordinator.

This module provides coding-specific safety rules and integration with
the framework's SafetyCoordinator for enhanced safety enforcement.

Design Pattern: Extension + Delegation
- Defines coding-specific safety rules
- Registers them with SafetyCoordinator
- Provides safety checking interface for coding operations

Integration Point:
    Use in CodingAssistant.get_extensions() as enhanced safety extension
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from victor_contracts import SafetyAction, SafetyCategory, SafetyCoordinator, SafetyRule
from victor_contracts.verticals import SafetyExtensionProtocol, SafetyPattern

logger = logging.getLogger(__name__)


class CodingSafetyRules:
    """Coding-specific safety rules for the SafetyCoordinator.

    Provides comprehensive safety rules for coding operations including:
    - Git operations (force push, branch deletion, etc.)
    - File operations (deleting critical files, overwriting system files)
    - Package management (rmtree, pip uninstall --yes)
    - Build/Deploy operations (docker system prune, kubectl delete)
    - Database operations (DROP, DELETE without WHERE)
    """

    @staticmethod
    def get_git_rules() -> List[SafetyRule]:
        """Get git-specific safety rules.

        Returns:
            List of safety rules for git operations
        """
        return [
            # Force push to main/master is BLOCKED
            SafetyRule(
                rule_id="coding_git_force_push_main",
                category=SafetyCategory.GIT,
                pattern=r"push.*--force.*\b(main|master)\b",
                description="Force push to main/master branch",
                action=SafetyAction.BLOCK,
                severity=9,
                tool_names=["git"],
            ),
            # Force push to other branches requires confirmation
            SafetyRule(
                rule_id="coding_git_force_push",
                category=SafetyCategory.GIT,
                pattern=r"push.*--force",
                description="Force push to any branch",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=7,
                confirmation_prompt="Force push can rewrite history. Continue?",
                tool_names=["git"],
            ),
            # Branch -D (force delete) requires confirmation
            SafetyRule(
                rule_id="coding_git_branch_force_delete",
                category=SafetyCategory.GIT,
                pattern=r"branch.*-D|\bbranch.*--delete.*--force",
                description="Force delete git branch",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=6,
                confirmation_prompt="This will force delete the branch. Continue?",
                tool_names=["git"],
            ),
            # Reset --hard is dangerous
            SafetyRule(
                rule_id="coding_git_reset_hard",
                category=SafetyCategory.GIT,
                pattern=r"reset.*--hard",
                description="Git reset --hard (discards all local changes)",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=8,
                confirmation_prompt="This will discard all uncommitted changes. Continue?",
                tool_names=["git"],
            ),
        ]

    @staticmethod
    def get_file_rules() -> List[SafetyRule]:
        """Get file operation safety rules.

        Returns:
            List of safety rules for file operations
        """
        return [
            # Writing to system directories is BLOCKED
            SafetyRule(
                rule_id="coding_file_system_write",
                category=SafetyCategory.FILE,
                pattern=r"write.*/(etc|usr/bin|usr/sbin|System|Windows)/",
                description="Write to system directory",
                action=SafetyAction.BLOCK,
                severity=10,
                tool_names=["write_file", "edit_files"],
            ),
            # Deleting .git directory is BLOCKED
            SafetyRule(
                rule_id="coding_file_delete_git",
                category=SafetyCategory.FILE,
                pattern=r"delete.*\.git|rm.*-rf.*\.git",
                description="Delete .git directory",
                action=SafetyAction.BLOCK,
                severity=10,
                tool_names=["shell", "execute_bash"],
            ),
            # Recursive delete requires confirmation
            SafetyRule(
                rule_id="coding_file_recursive_delete",
                category=SafetyCategory.FILE,
                pattern=r"rm.*-rf|rmdir.*/s|delete.*recursive",
                description="Recursive file/directory deletion",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=7,
                confirmation_prompt="This will recursively delete files. Continue?",
                tool_names=["shell", "execute_bash", "file_ops"],
            ),
        ]

    @staticmethod
    def get_package_rules() -> List[SafetyRule]:
        """Get package management safety rules.

        Returns:
            List of safety rules for package operations
        """
        return [
            # pip uninstall --yes is dangerous
            SafetyRule(
                rule_id="coding_pip_uninstall_all",
                category=SafetyCategory.SHELL,
                pattern=r"pip.*uninstall.*--yes|-y",
                description="Uninstall packages without confirmation",
                action=SafetyAction.WARN,
                severity=5,
                tool_names=["shell", "execute_bash"],
            ),
            # npm/pip install with --force or --ignore-installed
            SafetyRule(
                rule_id="coding_package_force_install",
                category=SafetyCategory.SHELL,
                pattern=r"(npm|pip|yarn|pnpm).*install.*--force",
                description="Force install packages",
                action=SafetyAction.WARN,
                severity=4,
                tool_names=["shell", "execute_bash"],
            ),
        ]

    @staticmethod
    def get_database_rules() -> List[SafetyRule]:
        """Get database operation safety rules.

        Returns:
            List of safety rules for database operations
        """
        return [
            # DROP TABLE is BLOCKED
            SafetyRule(
                rule_id="coding_db_drop_table",
                category=SafetyCategory.SHELL,
                pattern=r"DROP\s+TABLE|DROP\s+DATABASE",
                description="Drop table or database",
                action=SafetyAction.BLOCK,
                severity=10,
                tool_names=["shell", "execute_bash", "database"],
            ),
            # DELETE without WHERE is dangerous
            SafetyRule(
                rule_id="coding_db_delete_all",
                category=SafetyCategory.SHELL,
                pattern=r"DELETE.*FROM.*\bWHERE\b.*DELETE.*FROM.*$",
                description="DELETE without WHERE clause",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=8,
                confirmation_prompt="This will delete all rows. Continue?",
                tool_names=["shell", "execute_bash", "database"],
            ),
        ]

    @staticmethod
    def get_all_rules() -> List[SafetyRule]:
        """Get all coding-specific safety rules.

        Returns:
            List of all safety rules for coding operations
        """
        rules = []
        rules.extend(CodingSafetyRules.get_git_rules())
        rules.extend(CodingSafetyRules.get_file_rules())
        rules.extend(CodingSafetyRules.get_package_rules())
        rules.extend(CodingSafetyRules.get_database_rules())
        return rules


class EnhancedCodingSafetyExtension(SafetyExtensionProtocol):
    """Enhanced safety extension using SafetyCoordinator.

    This class provides the SafetyExtensionProtocol interface while
    delegating to the framework's SafetyCoordinator for actual
    safety checking.

    Example:
        extension = EnhancedCodingSafetyExtension()

        # Check if an operation is safe
        result = extension.check_operation("git", ["push", "--force", "origin", "main"])
        if not result.is_safe:
            print(f"Blocked: {result.block_reason}")
    """

    def __init__(
        self,
        strict_mode: bool = False,
        enable_custom_rules: bool = True,
    ):
        """Initialize the enhanced safety extension.

        Args:
            strict_mode: If True, treat warnings as blocks
            enable_custom_rules: If True, enable custom coding-specific rules
        """
        self._strict_mode = strict_mode
        self._enable_custom_rules = enable_custom_rules

        # Create SafetyCoordinator with coding-specific rules
        self._coordinator = SafetyCoordinator(
            strict_mode=strict_mode,
            enable_default_rules=True,
        )

        # Register coding-specific rules
        if enable_custom_rules:
            for rule in CodingSafetyRules.get_all_rules():
                self._coordinator.register_rule(rule)

        logger.info(
            f"EnhancedCodingSafetyExtension initialized with "
            f"{len(self._coordinator.list_rules())} safety rules"
        )

    def check_operation(
        self,
        tool_name: str,
        args: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Check if an operation is safe.

        Args:
            tool_name: Name of the tool being called
            args: Arguments to the tool
            context: Optional context for the check

        Returns:
            SafetyCheckResult from the coordinator
        """
        return self._coordinator.check_safety(tool_name, args, context)

    def is_operation_safe(
        self,
        tool_name: str,
        args: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Quick check if an operation is safe.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            context: Optional context

        Returns:
            True if operation is safe, False otherwise
        """
        return self._coordinator.is_operation_safe(tool_name, args, context)

    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Get coding-specific bash command patterns.

        Returns:
            List of safety patterns for dangerous bash commands
        """
        # Return patterns compatible with legacy safety system
        from victor_contracts.safety_patterns import (
            BUILD_DEPLOY_PATTERNS,
            GIT_PATTERNS,
            PACKAGE_MANAGER_PATTERNS,
        )

        patterns: List[SafetyPattern] = []
        if self._enable_custom_rules:
            patterns.extend(GIT_PATTERNS)
            patterns.extend(PACKAGE_MANAGER_PATTERNS)
            patterns.extend(BUILD_DEPLOY_PATTERNS)
        return patterns

    def get_file_patterns(self) -> List[SafetyPattern]:
        """Get coding-specific file operation patterns.

        Returns:
            List of safety patterns for file operations
        """
        from victor_contracts.safety_patterns import SENSITIVE_FILE_PATTERNS

        return SENSITIVE_FILE_PATTERNS if self._enable_custom_rules else []

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Get tool-specific argument restrictions.

        Returns:
            Dictionary mapping tool names to restricted arguments
        """
        return {
            "git": ["push --force origin main", "push --force origin master"],
            "shell": ["rm -rf /", "format", "fdisk"],
        }

    def get_coordinator(self) -> SafetyCoordinator:
        """Get the underlying SafetyCoordinator.

        Returns:
            SafetyCoordinator instance
        """
        return self._coordinator

    def add_custom_rule(self, rule: SafetyRule) -> None:
        """Add a custom safety rule.

        Args:
            rule: Safety rule to add
        """
        self._coordinator.register_rule(rule)
        logger.debug(f"Added custom safety rule: {rule.rule_id}")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a safety rule.

        Args:
            rule_id: ID of the rule to remove

        Returns:
            True if rule was removed, False if not found
        """
        return self._coordinator.unregister_rule(rule_id)

    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety statistics.

        Returns:
            Dictionary with safety statistics
        """
        return self._coordinator.get_stats_dict()


__all__ = [
    "CodingSafetyRules",
    "EnhancedCodingSafetyExtension",
]
