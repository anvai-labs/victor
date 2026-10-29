"""Safety-extension kernel shared by every vertical's ``safety_enhanced``.

Each vertical's ``EnhancedXSafetyExtension`` is the same delegation shell:
build a ``SafetyCoordinator``, register the vertical's rule table, expose the
``SafetyExtensionProtocol`` surface plus coordinator access, rule mutation,
and stats. The only per-vertical content is the tables. This module holds the
shell once; verticals subclass with their data.

Usage::

    from victor_contracts.verticals import CoordinatorSafetyExtension

    class EnhancedMySafetyExtension(CoordinatorSafetyExtension):
        def __init__(self, strict_mode: bool = False):
            super().__init__(
                "my",
                MySafetyRules.get_all_rules(),
                strict_mode=strict_mode,
                tool_restrictions={"shell": ["wipe --all"]},
            )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from victor_contracts import SafetyCoordinator, SafetyRule
from victor_contracts.verticals.protocols.promoted import (  # direct: avoid package cycle
    SafetyExtensionProtocol,
)
from victor_contracts.verticals.protocols.promoted_types import SafetyPattern

logger = logging.getLogger(__name__)


class CoordinatorSafetyExtension(SafetyExtensionProtocol):
    """``SafetyExtensionProtocol`` over a ``SafetyCoordinator`` with vertical tables.

    Args:
        vertical_name: Label for logs only.
        rules: The vertical's safety rules, registered when ``enable_custom_rules``.
        strict_mode: Passed to the coordinator (warnings become blocks).
        enable_custom_rules: When False, the vertical rules and pattern tables
            are withheld (the coordinator's default rules still apply).
        bash_patterns: Patterns for ``get_bash_patterns``.
        file_patterns: Patterns for ``get_file_patterns``.
        tool_restrictions: Tool -> restricted-argument mapping.
    """

    def __init__(
        self,
        vertical_name: str,
        rules: Sequence[SafetyRule] = (),
        *,
        strict_mode: bool = False,
        enable_custom_rules: bool = True,
        bash_patterns: Sequence[SafetyPattern] = (),
        file_patterns: Sequence[SafetyPattern] = (),
        tool_restrictions: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._vertical = vertical_name
        self._strict_mode = strict_mode
        self._enable_custom_rules = enable_custom_rules
        self._bash_patterns = list(bash_patterns)
        self._file_patterns = list(file_patterns)
        self._tool_restrictions: Dict[str, List[str]] = dict(tool_restrictions or {})
        self._coordinator = SafetyCoordinator(
            strict_mode=strict_mode,
            enable_default_rules=True,
        )
        if enable_custom_rules:
            for rule in rules:
                self._coordinator.register_rule(rule)
        logger.info(
            "Enhanced%sSafetyExtension initialized with %d safety rules",
            vertical_name.capitalize() if vertical_name else "",
            len(self._coordinator.list_rules()),
        )

    def check_operation(
        self,
        tool_name: str,
        args: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Check an operation; returns the coordinator's SafetyCheckResult."""
        return self._coordinator.check_safety(tool_name, args, context)

    def is_operation_safe(
        self,
        tool_name: str,
        args: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Quick safety check."""
        return self._coordinator.is_operation_safe(tool_name, args, context)

    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Dangerous-bash patterns (empty when custom rules are disabled)."""
        return list(self._bash_patterns) if self._enable_custom_rules else []

    def get_file_patterns(self) -> List[SafetyPattern]:
        """Dangerous-file-operation patterns (empty when custom rules are disabled)."""
        return list(self._file_patterns) if self._enable_custom_rules else []

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Tool -> restricted arguments mapping."""
        return dict(self._tool_restrictions)

    def get_coordinator(self) -> SafetyCoordinator:
        """The underlying coordinator."""
        return self._coordinator

    def add_custom_rule(self, rule: SafetyRule) -> None:
        """Register an additional safety rule."""
        self._coordinator.register_rule(rule)
        logger.debug("Added custom safety rule: %s", rule.rule_id)

    def remove_rule(self, rule_id: str) -> bool:
        """Unregister a rule by id."""
        return self._coordinator.unregister_rule(rule_id)

    def get_safety_stats(self) -> Dict[str, Any]:
        """Coordinator statistics."""
        return self._coordinator.get_stats_dict()
