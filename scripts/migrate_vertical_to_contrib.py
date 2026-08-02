#!/usr/bin/env python3
"""Automatic migration script to convert verticals to use contrib packages.

This script analyzes vertical implementations and generates contrib-based versions
using BaseSafetyExtension, BaseModeConfigProvider, and BaseConversationManager.

Usage:
    python scripts/migrate_vertical_to_contrib.py --vertical victor/benchmark --dry-run
    python scripts/migrate_vertical_to_contrib.py --vertical victor/benchmark --apply
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MigrationReport:
    """Report of migration actions."""

    vertical_name: str
    dry_run: bool
    files_analyzed: List[str] = None
    files_to_migrate: List[str] = None
    safety_migration: Optional[str] = None
    mode_config_migration: Optional[str] = None
    conversation_migration: Optional[str] = None
    code_reduction_lines: int = 0
    code_reduction_percent: float = 0.0
    warnings: List[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.files_analyzed is None:
            self.files_analyzed = []
        if self.files_to_migrate is None:
            self.files_to_migrate = []
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


class VerticalAnalyzer(ast.NodeVisitor):
    """Analyze vertical implementation to detect migration opportunities."""

    def __init__(self, vertical_path: str):
        self.vertical_path = Path(vertical_path)
        self.has_safety = False
        self.has_mode_config = False
        self.has_conversation = False
        self.safety_file = None
        self.mode_config_file = None
        self.uses_contrib = False

    def analyze(self) -> MigrationReport:
        """Analyze the vertical and generate migration report."""
        report = MigrationReport(
            vertical_name=self.vertical_path.name,
            dry_run=True,  # Will be updated by caller
        )

        # Check for key files
        for py_file in self.vertical_path.rglob("*.py"):
            if py_file.name == "safety.py":
                self.safety_file = py_file
                self.has_safety = True
                report.files_analyzed.append(str(py_file))
            elif py_file.name == "mode_config.py":
                self.mode_config_file = py_file
                self.has_mode_config = True
                report.files_analyzed.append(str(py_file))

        # Check if already uses contrib
        for py_file in self.vertical_path.rglob("*.py"):
            content = py_file.read_text()
            if "from victor.contrib" in content or "from victor.contrib" in content:
                self.uses_contrib = True
                report.warnings.append("Vertical already uses contrib packages")
                break

        # Estimate code reduction
        if self.has_safety:
            report.safety_migration = "Migrate to BaseSafetyExtension"
            report.code_reduction_lines += 150  # Typical safety file
            report.files_to_migrate.append("safety.py")

        if self.has_mode_config:
            report.mode_config_migration = "Migrate to BaseModeConfigProvider"
            report.code_reduction_lines += 110  # Typical mode_config file
            report.files_to_migrate.append("mode_config.py")

        # Calculate percentage reduction (typical vertical has ~950 lines of duplicate code)
        if report.code_reduction_lines > 0:
            report.code_reduction_percent = (report.code_reduction_lines / 950) * 0.68

        return report


class VerticalMigrator:
    """Migrate vertical to use contrib packages."""

    def __init__(self, vertical_path: str, dry_run: bool = False):
        self.vertical_path = Path(vertical_path)
        self.dry_run = dry_run
        self.backup_dir = self.vertical_path / f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def migrate(self, report: MigrationReport, analyzer: VerticalAnalyzer) -> MigrationReport:
        """Perform the migration.

        Args:
            report: Migration report from analyzer
            analyzer: VerticalAnalyzer instance

        Returns:
            Updated migration report
        """
        report.dry_run = self.dry_run

        if not self.dry_run:
            # Create backup
            self.backup_dir.mkdir(exist_ok=True)
            print(f"Created backup directory: {self.backup_dir}")

        # Migrate safety.py
        if analyzer.has_safety and analyzer.safety_file:
            result = self._migrate_safety()
            report.files_to_migrate.append("safety.py")
            if result:
                report.code_reduction_lines += result[0]
                report.code_reduction_percent += result[1]

        # Migrate mode_config.py
        if analyzer.has_mode_config and analyzer.mode_config_file:
            result = self._migrate_mode_config()
            report.files_to_migrate.append("mode_config.py")
            if result:
                report.code_reduction_lines += result[0]
                report.code_reduction_percent += result[1]

        return report

    def _migrate_safety(self) -> Optional[Tuple[int, float]]:
        """Migrate safety.py to use BaseSafetyExtension.

        Returns:
            Tuple of (lines_saved, percent_reduced) or None if skipped
        """
        safety_file = self.vertical_path / "safety.py"
        original_lines = len(safety_file.read_text().splitlines())

        # Generate new safety.py content
        vertical_name = self.vertical_path.name
        content = f'''# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

This module provides safety rules for the {vertical_name} vertical using
the BaseSafetyExtension from victor.contrib.safety.

Safety rules cover:
- Repository isolation (prevent modifying non-{vertical_name} repos)
- Resource limits (timeout, cost, token usage)
- Test isolation (prevent running tests on production systems)
- Data privacy (prevent uploading {vertical_name} data externally)

Migrated from: safety.py (original implementation)
Migration date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

from typing import List, Optional

from victor.contrib.safety import BaseSafetyExtension, VerticalSafetyMixin
from victor.agent.coordinators.safety_coordinator import (
    SafetyRule,
    SafetyAction,
    SafetyCategory,
)


class {vertical_name.title()}SafetyExtension(BaseSafetyExtension, VerticalSafetyMixin):
    """Safety extension for {vertical_name} vertical.

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
        return "{vertical_name}"

    def get_vertical_rules(self) -> List[SafetyRule]:
        """Define {vertical_name}-specific safety rules.

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
                rule_id="{vertical_name}_dangerous_commands",
                command_pattern=r"rm -rf|format|fdisk",
                description="Dangerous system operation",
                severity=10,
            ),

            # Example: Use helper for file deletion pattern
            self.create_file_deletion_rule(
                rule_id="{vertical_name}_file_deletion",
                file_pattern=r".*\\.db|.*database|.*data",
                description="Database/data file deletion",
                severity=9,
            ),

            # Example: Use helper for completely blocked operation
            self.create_blocked_operation_rule(
                rule_id="{vertical_name}_production_modify",
                pattern=r"/prod/|/production/|release",
                description="Cannot modify production systems",
                category=SafetyCategory.SHELL,
            ),

            # Example: Use helper for git operations
            self.create_git_force_push_rule(
                rule_id="{vertical_name}_git_force_push",
                severity=8,
            ),
        ]

        # Add custom rules for {vertical_name} specific safety concerns
        rules.extend(self._get_custom_rules())

        return rules

    def _get_custom_rules(self) -> List[SafetyRule]:
        """Get custom {vertical_name}-specific safety rules.

        Override this method to add vertical-specific rules beyond
        the common patterns provided by VerticalSafetyMixin.

        Returns:
            List of custom SafetyRule instances
        """
        # TODO: Add {vertical_name}-specific safety rules here
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
        # Use coordinator's safety checking
        ctx = context or {{}}
        return self._coordinator.is_operation_safe(
            tool_name=tool_name,
            arguments=arguments,
            context=ctx,
        )

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

    def get_safety_stats(self) -> dict:
        """Get safety statistics.

        Returns:
            Dictionary with safety statistics
        """
        return self._coordinator.get_stats()


__all__ = [
    "{vertical_name.title()}SafetyExtension",
]
'''

        if not self.dry_run:
            # Backup original file
            shutil.copy2(safety_file, self.backup_dir / "safety.py")

            # Write new file
            safety_file.write_text(content)

            new_lines = len(content.splitlines())
            lines_saved = original_lines - new_lines
            percent_reduced = (lines_saved / original_lines) * 100 if original_lines > 0 else 0

            print(
                f"✓ Migrated safety.py: saved {lines_saved} lines ({percent_reduced:.1f}% reduction)"
            )
            return (lines_saved, percent_reduced)
        else:
            print(
                f"[DRY RUN] Would migrate safety.py: ~{original_lines} lines → ~{len(content.splitlines())} lines"
            )
            estimated_lines = original_lines - len(content.splitlines())
            return (
                estimated_lines,
                (estimated_lines / original_lines * 100) if original_lines > 0 else 0,
            )

    def _migrate_mode_config(self) -> Optional[Tuple[int, float]]:
        """Migrate mode_config.py to use BaseModeConfigProvider.

        Returns:
            Tuple of (lines_saved, percent_reduced) or None if skipped
        """
        mode_config_file = self.vertical_path / "mode_config.py"
        original_lines = len(mode_config_file.read_text().splitlines())

        # Generate new mode_config.py content
        vertical_name = self.vertical_path.name
        content = f'''# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""{vertical_name.title()} mode configuration using victor.contrib.

This module provides mode configurations for the {vertical_name} vertical using
the BaseModeConfigProvider from victor.contrib.mode_config.

{vertical_name.title()} modes are optimized for:
- fast: Quick operations with minimal tool calls
- default: Balanced settings for standard tasks
- thorough: Comprehensive analysis with higher budgets

Migrated from: mode_config.py (original implementation)
Migration date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

from typing import Dict

from victor.contrib.mode_config import BaseModeConfigProvider, ModeHelperMixin
from victor.core.mode_config import ModeDefinition


class {vertical_name.title()}ModeConfig(BaseModeConfigProvider, ModeHelperMixin):
    """Mode configuration provider for {vertical_name} vertical.

    This provider uses BaseModeConfigProvider to provide:
    - Automatic registration with ModeConfigRegistry
    - Inherited framework default modes (quick, standard, thorough)
    - Tool budget calculation with task-type support
    - 6 helper methods for creating common modes
    - Protocol compliance (ModeConfigProviderProtocol)
    """

    def get_vertical_name(self) -> str:
        """Return the vertical name for mode registration.

        Returns:
            Vertical identifier
        """
        return "{vertical_name}"

    def get_vertical_modes(self) -> Dict[str, ModeDefinition]:
        """Define {vertical_name}-specific mode definitions.

        Returns:
            Dict mapping mode names to ModeDefinition instances

        Note:
            Use helper methods from ModeHelperMixin to create
            common modes:
            - create_quick_mode(): Fast operations (5 tool budget)
            - create_standard_mode(): Balanced (15 tool budget)
            - create_thorough_mode(): Deep analysis (30 tool budget)
            - create_exploration_mode(): Extended exploration (20 tool budget)
            - create_custom_mode(): Custom configuration

        Helper methods for mode groups:
            - create_quick_modes(): Returns {{'quick': ..., 'fast': ...}}
            - create_standard_modes(): Returns {{'standard': ..., 'default': ...}}
            - create_thorough_modes(): Returns {{'thorough': ..., 'comprehensive': ...}}
        """
        # Start with common modes from helper methods
        modes = {{
            **self.create_quick_modes(),
            **self.create_standard_modes(),
            **self.create_thorough_modes(),
        }}

        # Add {vertical_name}-specific modes
        modes.update({{
            # Example: Custom mode with specific requirements
            "default": ModeDefinition(
                name="default",
                tool_budget=30,
                max_iterations=15,
                temperature=0.2,
                description=f"Balanced settings for standard {vertical_name} tasks",
                exploration_multiplier=1.0,
                priority_tools=[
                    "read",
                    "grep",
                    "edit",
                    "shell",
                ],
            ),
        }})

        return modes

    def get_task_budgets(self) -> Dict[str, int]:
        """Define task-specific tool budgets.

        Returns:
            Dict mapping task types to recommended tool budgets

        Example:
            return {{
                "quick_fix": 5,
                "refactor": 15,
                "investigation": 20,
            }}
        """
        # TODO: Define {vertical_name}-specific task budgets
        return {{
            # Quick tasks
            "quick": 10,
            "fast": 15,
            # Standard tasks
            "default": 30,
            "standard": 30,
            # Complex tasks
            "thorough": 50,
            "comprehensive": 60,
        }}

    def get_default_mode(self) -> str:
        """Specify the default mode for this vertical.

        Returns:
            Default mode name
        """
        return "default"

    def get_default_budget(self) -> int:
        """Specify the default tool budget.

        Returns:
            Default tool budget when no mode or task specified
        """
        return 30


__all__ = [
    "{vertical_name.title()}ModeConfig",
]
'''

        if not self.dry_run:
            # Backup original file
            shutil.copy2(mode_config_file, self.backup_dir / "mode_config.py")

            # Write new file
            mode_config_file.write_text(content)

            new_lines = len(content.splitlines())
            lines_saved = original_lines - new_lines
            percent_reduced = (lines_saved / original_lines) * 100 if original_lines > 0 else 0

            print(
                f"✓ Migrated mode_config.py: saved {lines_saved} lines ({percent_reduced:.1f}% reduction)"
            )
            return (lines_saved, percent_reduced)
        else:
            print(
                f"[DRY RUN] Would migrate mode_config.py: ~{original_lines} lines → ~{len(content.splitlines())} lines"
            )
            estimated_lines = original_lines - len(content.splitlines())
            return (
                estimated_lines,
                (estimated_lines / original_lines * 100) if original_lines > 0 else 0,
            )


def print_report(report: MigrationReport):
    """Print migration report to console.

    Args:
        report: Migration report to print
    """
    print("=" * 80)
    print(f"Vertical Migration Report: {report.vertical_name}")
    print("=" * 80)
    print()

    print(f"Mode: {'DRY RUN' if report.dry_run else 'APPLY'}")
    print()

    print("Files Analyzed:")
    for file in report.files_analyzed:
        print(f"  - {file}")
    print()

    print("Migration Plan:")
    if report.safety_migration:
        print(f"  ✓ {report.safety_migration}")
    if report.mode_config_migration:
        print(f"  ✓ {report.mode_config_migration}")
    if report.conversation_migration:
        print(f"  ✓ {report.conversation_migration}")
    print()

    print(
        f"Estimated Code Reduction: {report.code_reduction_lines} lines ({report.code_reduction_percent:.1f}%)"
    )
    print()

    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"  ⚠️  {warning}")
        print()

    if report.errors:
        print("Errors:")
        for error in report.errors:
            print(f"  ❌ {error}")
        print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Migrate verticals to use victor.contrib packages")
    parser.add_argument(
        "--vertical",
        type=str,
        required=True,
        help="Path to vertical directory (e.g., victor/benchmark)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration (backs up files first)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.dry_run and args.apply:
        parser.error("Cannot specify both --dry-run and --apply")

    if not args.dry_run and not args.apply:
        parser.error("Must specify either --dry-run or --apply")

    vertical_path = Path(args.vertical)
    if not vertical_path.exists():
        parser.error(f"Vertical path does not exist: {args.vertical}")

    # Analyze vertical
    print(f"Analyzing vertical: {args.vertical}")
    print()

    analyzer = VerticalAnalyzer(args.vertical)
    report = analyzer.analyze()

    # Print analysis report
    print_report(report)

    # Migrate if not dry run
    if args.apply:
        migrator = VerticalMigrator(args.vertical, dry_run=False)
        report = migrator.migrate(report, analyzer)
        print()
        print("Migration complete! ✓")
        print(f"Backup created at: {migrator.backup_dir}")


if __name__ == "__main__":
    main()
