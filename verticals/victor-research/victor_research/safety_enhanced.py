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

"""Enhanced safety integration for victor-research using SafetyCoordinator.

This module provides research-specific safety rules and integration with
the framework's SafetyCoordinator for enhanced safety enforcement.

Design Pattern: Extension + Delegation
- Defines research-specific safety rules
- Registers them with SafetyCoordinator
- Provides safety checking interface for research operations

Integration Point:
    Use in ResearchAssistant.get_extensions() as enhanced safety extension
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from victor_contracts import SafetyAction, SafetyCategory, SafetyCoordinator, SafetyRule
from victor_contracts.verticals import SafetyExtensionProtocol, SafetyPattern

logger = logging.getLogger(__name__)


class ResearchSafetyRules:
    """Research-specific safety rules for the SafetyCoordinator.

    Provides comprehensive safety rules for research operations including:
    - Data collection operations (web scraping, API calls)
    - Experiment operations (model training, parameter tuning)
    - File operations (deleting research data, overwriting results)
    - Publication operations (auto-publishing, data disclosure)
    """

    @staticmethod
    def get_data_collection_rules() -> List[SafetyRule]:
        """Get data collection safety rules.

        Returns:
            List of safety rules for data collection operations
        """
        return [
            # Bulk web scraping requires confirmation
            SafetyRule(
                rule_id="research_bulk_scrape",
                category=SafetyCategory.SHELL,
                pattern=r"scrape.*bulk|crawl.*--all",
                description="Bulk web scraping operation",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=6,
                confirmation_prompt=(
                    "This will perform bulk data collection. Ensure you comply "
                    "with robots.txt and rate limits. Continue?"
                ),
                tool_names=["shell", "execute_bash"],
            ),
            # Sensitive data collection requires confirmation
            SafetyRule(
                rule_id="research_sensitive_data",
                category=SafetyCategory.SHELL,
                pattern=r"collect.*personal|scrape.*PII|download.*sensitive",
                description="Collection of potentially sensitive personal data",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=8,
                confirmation_prompt=(
                    "This may collect personal data. Ensure you have consent "
                    "and comply with privacy regulations. Continue?"
                ),
                tool_names=["shell", "execute_bash", "web"],
            ),
        ]

    @staticmethod
    def get_experiment_rules() -> List[SafetyRule]:
        """Get experiment safety rules.

        Returns:
            List of safety rules for experiment operations
        """
        return [
            # Deleting experiment data is dangerous
            SafetyRule(
                rule_id="research_delete_experiments",
                category=SafetyCategory.FILE,
                pattern=r"delete.*experiments|rm.*-rf.*results|wipe.*data",
                description="Delete experiment data and results",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=8,
                confirmation_prompt=(
                    "This will delete experiment data. Consider archiving "
                    "instead. Continue?"
                ),
                tool_names=["shell", "execute_bash", "file_ops"],
            ),
            # Overwriting published results requires confirmation
            SafetyRule(
                rule_id="research_overwrite_published",
                category=SafetyCategory.FILE,
                pattern=r"overwrite.*published|update.*results.*--force",
                description="Overwrite published research results",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=7,
                confirmation_prompt="This will modify published results. Continue?",
                tool_names=["shell", "execute_bash", "file_ops"],
            ),
        ]

    @staticmethod
    def get_computation_rules() -> List[SafetyRule]:
        """Get computation resource safety rules.

        Returns:
            List of safety rules for compute operations
        """
        return [
            # Large-scale compute requires confirmation
            SafetyRule(
                rule_id="research_large_compute",
                category=SafetyCategory.SHELL,
                pattern=r"(train|fit|compute).*--gpu.*all|(train|fit).*--epochs.*999",
                description="Large-scale computation (all GPUs, excessive epochs)",
                action=SafetyAction.WARN,
                severity=5,
                tool_names=["shell", "execute_bash"],
            ),
            # Expensive API calls require confirmation
            SafetyRule(
                rule_id="research_expensive_api",
                category=SafetyCategory.SHELL,
                pattern=r"api.*call.*--bulk|query.*--large.*cost",
                description="Bulk API calls with potential high cost",
                action=SafetyAction.WARN,
                severity=4,
                tool_names=["shell", "execute_bash", "web"],
            ),
        ]

    @staticmethod
    def get_publication_rules() -> List[SafetyRule]:
        """Get publication safety rules.

        Returns:
            List of safety rules for publication operations
        """
        return [
            # Auto-publishing requires confirmation
            SafetyRule(
                rule_id="research_auto_publish",
                category=SafetyCategory.SHELL,
                pattern=r"publish.*--auto|deploy.*research|release.*--force",
                description="Auto-publish research results",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=9,
                confirmation_prompt=(
                    "This will publish research results. Review before "
                    "publishing. Continue?"
                ),
                tool_names=["shell", "execute_bash"],
            ),
            # Data disclosure requires confirmation
            SafetyRule(
                rule_id="research_disclose_data",
                category=SafetyCategory.SHELL,
                pattern=r"share.*dataset|upload.*data.*public|publish.*data",
                description="Disclose or publish research data",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=7,
                confirmation_prompt="Ensure data doesn't contain sensitive information. Continue?",
                tool_names=["shell", "execute_bash", "file_ops"],
            ),
        ]

    @staticmethod
    def get_all_rules() -> List[SafetyRule]:
        """Get all research-specific safety rules.

        Returns:
            List of all safety rules for research operations
        """
        rules = []
        rules.extend(ResearchSafetyRules.get_data_collection_rules())
        rules.extend(ResearchSafetyRules.get_experiment_rules())
        rules.extend(ResearchSafetyRules.get_computation_rules())
        rules.extend(ResearchSafetyRules.get_publication_rules())
        return rules


class EnhancedResearchSafetyExtension(SafetyExtensionProtocol):
    """Enhanced safety extension for Research using SafetyCoordinator.

    This class provides the SafetyExtensionProtocol interface while
    delegating to the framework's SafetyCoordinator for actual
    safety checking.

    Example:
        extension = EnhancedResearchSafetyExtension()

        # Check if an operation is safe
        result = extension.check_operation("web", ["scrape", "--bulk", "target"])
        if not result.is_safe:
            print(f"Warning: {result.confirmation_prompt}")
    """

    def __init__(
        self,
        strict_mode: bool = False,
        enable_custom_rules: bool = True,
    ):
        """Initialize the enhanced safety extension.

        Args:
            strict_mode: If True, treat warnings as blocks
            enable_custom_rules: If True, enable custom research-specific rules
        """
        self._strict_mode = strict_mode
        self._enable_custom_rules = enable_custom_rules

        # Create SafetyCoordinator with research-specific rules
        self._coordinator = SafetyCoordinator(
            strict_mode=strict_mode,
            enable_default_rules=True,
        )

        # Register research-specific rules
        if enable_custom_rules:
            for rule in ResearchSafetyRules.get_all_rules():
                self._coordinator.register_rule(rule)

        logger.info(
            f"EnhancedResearchSafetyExtension initialized with "
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
        """Get research-specific bash command patterns.

        Returns:
            List of safety patterns for dangerous bash commands
        """
        return []

    def get_file_patterns(self) -> List[SafetyPattern]:
        """Get research-specific file operation patterns.

        Returns:
            List of safety patterns for file operations
        """
        return []

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Get tool-specific argument restrictions.

        Returns:
            Dictionary mapping tool names to restricted arguments
        """
        return {
            "web": ["scrape --bulk", "crawl --all"],
            "shell": ["rm -rf results/*", "delete experiments/*"],
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
    "ResearchSafetyRules",
    "EnhancedResearchSafetyExtension",
]
