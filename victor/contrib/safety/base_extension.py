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

"""Base safety extension for Victor verticals.

This module provides BaseSafetyExtension, a reusable base class that implements
the SafetyExtensionProtocol using the framework's SafetyCoordinator. Verticals can
inherit from this base class to get common safety functionality while adding
vertical-specific safety rules.

Design Pattern: Template Method
- Base class provides common safety checking infrastructure
- Verticals override get_vertical_rules() to provide domain-specific rules
- Verticals can optionally override bash/file patterns and tool restrictions

Usage:
    from victor.contrib.safety import BaseSafetyExtension, SafetyContext
    from victor.core.verticals.protocols import SafetyPattern

    class MyVerticalSafetyExtension(BaseSafetyExtension):
        def get_vertical_name(self) -> str:
            return "myvertical"

        def get_vertical_rules(self) -> List[SafetyRule]:
            return [
                SafetyRule(
                    rule_id="myvertical_rule",
                    pattern=r"dangerous-command",
                    description="Dangerous operation",
                    action=SafetyAction.WARN,
                    severity=5,
                )
            ]
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from victor_contracts.safety import (
    SafetyAction,
    SafetyCategory,
    SafetyCoordinator,
    SafetyRule,
)
from victor.core.verticals.protocols import SafetyExtensionProtocol, SafetyPattern
from victor.contrib.safety.safety_context import SafetyContext
from victor.contrib.safety.vertical_mixin import VerticalSafetyMixin

logger = logging.getLogger(__name__)


class BaseSafetyExtension(SafetyExtensionProtocol):
    """Base safety extension for Victor verticals.

    Provides common safety functionality using SafetyCoordinator:
    - Automatic registration of vertical-specific rules
    - Safety checking for operations
    - Statistics and observability
    - Vertical context tracking

    Verticals should:
    1. Inherit from BaseSafetyExtension
    2. Implement get_vertical_name() to return vertical identifier
    3. Implement get_vertical_rules() to return vertical-specific SafetyRule list
    4. Optionally override get_bash_patterns(), get_file_patterns(), get_tool_restrictions()
    """

    def __init__(self, strict_mode: bool = False, enable_custom_rules: bool = True):
        """Initialize the safety extension.

        Args:
            strict_mode: If True, blocks operations; if False, warns
            enable_custom_rules: If True, enables custom rule registration
        """
        self._coordinator = SafetyCoordinator(strict_mode=strict_mode)
        self._context = SafetyContext(vertical_name=self.get_vertical_name())

        # Register vertical-specific rules
        for rule in self.get_vertical_rules():
            self._coordinator.register_rule(rule)

        # Register custom rules if enabled
        if enable_custom_rules:
            for rule in self._get_custom_rules():
                self._coordinator.register_rule(rule)

        logger.info(
            f"{self.__class__.__name__} initialized for '{self.get_vertical_name()}' "
            f"with {len(self.get_vertical_rules())} rules"
        )

    # ==========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # ==========================================================================

    @abstractmethod
    def get_vertical_name(self) -> str:
        """Get the vertical name for logging and context tracking.

        Returns:
            Vertical name (e.g., "devops", "rag", "research")
        """
        ...

    @abstractmethod
    def get_vertical_rules(self) -> List[SafetyRule]:
        """Get vertical-specific safety rules.

        Returns:
            List of SafetyRule instances for this vertical
        """
        ...

    # ==========================================================================
    # Template Methods - Can be overridden by subclasses
    # ==========================================================================

    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Get bash command safety patterns for this vertical.

        Returns:
            List of SafetyPattern instances for bash commands
        """
        return []

    def get_file_patterns(self) -> List[SafetyPattern]:
        """Get file operation safety patterns for this vertical.

        Returns:
            List of SafetyPattern instances for file operations
        """
        return []

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Get tool usage restrictions for this vertical.

        Returns:
            Dict mapping tool names to restricted operation lists
        """
        return {}

    # ==========================================================================
    # SafetyExtensionProtocol Implementation - Common for all verticals
    # ==========================================================================

    def check_operation(
        self,
        tool_name: str,
        args: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Check if an operation is safe.

        Delegates to SafetyCoordinator with vertical context.

        Args:
            tool_name: Name of the tool being called
            args: Arguments being passed to the tool
            context: Optional context information

        Returns:
            Safety check result from coordinator
        """
        # Enrich context with vertical information
        enriched_context = self._enrich_context(context)

        result = self._coordinator.check_safety(tool_name, args, enriched_context)

        # Track the operation in vertical context
        self._context.track_operation(tool_name, args, result)

        return result

    def is_operation_safe(
        self,
        tool_name: str,
        args: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if an operation is safe (boolean result).

        Args:
            tool_name: Name of the tool being called
            args: Arguments being passed to the tool
            context: Optional context information

        Returns:
            True if operation is safe, False otherwise
        """
        enriched_context = self._enrich_context(context)
        return self._coordinator.is_operation_safe(tool_name, args, enriched_context)

    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety statistics.

        Returns:
            Dictionary with safety statistics from coordinator
        """
        stats = self._coordinator.get_stats_dict()
        stats["vertical"] = self.get_vertical_name()
        stats["context"] = self._context.to_dict()
        return stats

    def get_coordinator(self) -> SafetyCoordinator:
        """Get the underlying SafetyCoordinator instance.

        Returns:
            SafetyCoordinator instance
        """
        return self._coordinator

    def get_context(self) -> SafetyContext:
        """Get the vertical safety context.

        Returns:
            SafetyContext instance with vertical information
        """
        return self._context

    # ==========================================================================
    # Private Helper Methods
    # ==========================================================================

    def _enrich_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enrich context with vertical-specific information.

        Args:
            context: Original context

        Returns:
            Enriched context with vertical information
        """
        if context is None:
            context = {}

        # Add vertical context
        context["vertical"] = self.get_vertical_name()
        context["vertical_context"] = self._context.to_dict()

        return context

    def _get_custom_rules(self) -> List[SafetyRule]:
        """Get custom safety rules registered at runtime.

        Subclasses can override this to support dynamic rule registration.

        Returns:
            List of custom SafetyRule instances
        """
        return []

    def _register_rule(self, rule: SafetyRule) -> None:
        """Register an additional safety rule at runtime.

        Args:
            rule: SafetyRule to register
        """
        self._coordinator.register_rule(rule)
        logger.debug(f"Registered custom rule '{rule.rule_id}' for '{self.get_vertical_name()}'")


__all__ = [
    "BaseSafetyExtension",
]
