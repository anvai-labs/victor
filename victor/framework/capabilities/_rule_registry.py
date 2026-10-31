# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Shared custom-rule bookkeeping for rule-based capability providers.

Grounding rules and safety rules both combine class-level default rules with an instance list of
caller-supplied custom rules, and memoize a derived view that must be invalidated whenever the
custom list changes. This base owns that bookkeeping; subclasses own the defaults, the derived
view and how rules are filtered and ordered (:meth:`CustomRuleRegistry.get_rules`).
"""

from __future__ import annotations

from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar


class _IdentifiedRule(Protocol):
    """Any rule definition carrying a unique identifier."""

    rule_id: str


RuleT = TypeVar("RuleT", bound=_IdentifiedRule)


class CustomRuleRegistry(Generic[RuleT]):
    """Default rules plus mutable custom rules, with a cache invalidated on every change.

    Attributes:
        _custom_rules: Caller-supplied rules (the given list is used as-is, not copied).
        _strict_mode: Whether strict rule enforcement is enabled.
        _rule_cache: Subclass-defined derived view of all rules; None means "rebuild".
    """

    def __init__(
        self,
        custom_rules: Optional[List[RuleT]] = None,
        strict_mode: bool = False,
    ) -> None:
        """Initialize the rule registry.

        Args:
            custom_rules: Optional custom rules to add/override defaults
            strict_mode: Whether to enable strict rule enforcement
        """
        self._custom_rules: List[RuleT] = custom_rules or []
        self._strict_mode = strict_mode
        self._rule_cache: Optional[Dict[Any, List[RuleT]]] = None

    def get_rules(self) -> List[RuleT]:
        """Return every rule (defaults and custom) in the subclass's canonical order."""
        raise NotImplementedError

    def add_rule(self, rule: RuleT) -> None:
        """Add a custom rule.

        Args:
            rule: Rule to add
        """
        self._custom_rules.append(rule)
        self._rule_cache = None  # Clear cache

    def remove_rule(self, rule_id: str) -> bool:
        """Remove the first custom rule with the given ID (default rules are never removed).

        Args:
            rule_id: ID of the rule to remove

        Returns:
            True if rule was removed, False if not found
        """
        for i, rule in enumerate(self._custom_rules):
            if rule.rule_id == rule_id:
                self._custom_rules.pop(i)
                self._rule_cache = None
                return True
        return False

    def get_rule(self, rule_id: str) -> Optional[RuleT]:
        """Get a specific rule (default or custom) by ID.

        Args:
            rule_id: Rule identifier

        Returns:
            The first matching rule in :meth:`get_rules` order, or None if not found
        """
        for rule in self.get_rules():
            if rule.rule_id == rule_id:
                return rule
        return None


__all__ = ["CustomRuleRegistry"]
