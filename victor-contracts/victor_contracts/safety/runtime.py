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

"""SDK-owned safety coordinator contracts and default implementation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from victor_contracts.constants import ToolNames, get_canonical_name

logger = logging.getLogger(__name__)


class SafetyAction(Enum):
    """Actions to take when a safety rule is triggered."""

    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    REQUIRE_CONFIRMATION = "require_confirmation"


class SafetyCategory(Enum):
    """Categories of safety rules."""

    GIT = "git"
    FILE = "file"
    SHELL = "shell"
    DOCKER = "docker"
    NETWORK = "network"
    SYSTEM = "system"


@dataclass
class SafetyRule:
    """A safety rule definition."""

    rule_id: str
    category: SafetyCategory
    pattern: str
    description: str
    action: SafetyAction
    severity: int = 5
    tool_names: List[str] = field(default_factory=list)
    confirmation_prompt: str = ""

    def __post_init__(self) -> None:
        """Normalize configured tool names to canonical runtime forms."""
        self.tool_names = list(dict.fromkeys(get_canonical_name(name) for name in self.tool_names))

    def matches(self, tool_name: str, args: List[str]) -> bool:
        """Check whether the rule matches a tool invocation."""
        canonical_tool_name = get_canonical_name(tool_name)
        if self.tool_names and canonical_tool_name not in self.tool_names:
            return False

        if self.pattern:
            if re.search(self.pattern, canonical_tool_name, re.IGNORECASE):
                return True

            args_str = " ".join(args)
            if re.search(self.pattern, args_str, re.IGNORECASE):
                return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "rule_id": self.rule_id,
            "category": self.category.value,
            "pattern": self.pattern,
            "description": self.description,
            "action": self.action.value,
            "severity": self.severity,
            "tool_names": self.tool_names,
            "confirmation_prompt": self.confirmation_prompt,
        }


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""

    is_safe: bool
    action: SafetyAction
    matched_rules: List[SafetyRule] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    block_reason: str = ""
    confirmation_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "is_safe": self.is_safe,
            "action": self.action.value,
            "matched_rules": [rule.to_dict() for rule in self.matched_rules],
            "warnings": self.warnings,
            "block_reason": self.block_reason,
            "confirmation_prompt": self.confirmation_prompt,
        }


@dataclass
class SafetyStats:
    """Statistics about safety checks."""

    total_checks: int = 0
    blocked_operations: int = 0
    warned_operations: int = 0
    confirmed_operations: int = 0
    rule_hits: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "total_checks": self.total_checks,
            "blocked_operations": self.blocked_operations,
            "warned_operations": self.warned_operations,
            "confirmed_operations": self.confirmed_operations,
            "rule_hits": self.rule_hits.copy(),
        }


class SafetyCoordinator:
    """Coordinates safety checks for orchestrators and vertical extensions."""

    DEFAULT_RULES: List[SafetyRule] = [
        SafetyRule(
            rule_id="git_force_push_main",
            category=SafetyCategory.GIT,
            pattern=r"push.*--force.*main|push.*--force.*master",
            description="Force push to main/master branch",
            action=SafetyAction.BLOCK,
            severity=9,
            tool_names=["git"],
            confirmation_prompt="Force pushing to main branch can rewrite history. Are you sure?",
        ),
        SafetyRule(
            rule_id="git_force_push",
            category=SafetyCategory.GIT,
            pattern=r"push.*--force",
            description="Force push",
            action=SafetyAction.REQUIRE_CONFIRMATION,
            severity=7,
            tool_names=["git"],
            confirmation_prompt="Force push can rewrite history. Continue?",
        ),
        SafetyRule(
            rule_id="git_delete_branch",
            category=SafetyCategory.GIT,
            pattern=r"branch.*-D|branch.*--delete.*--force",
            description="Force delete branch",
            action=SafetyAction.WARN,
            severity=5,
            tool_names=["git"],
        ),
        SafetyRule(
            rule_id="file_delete_recursive",
            category=SafetyCategory.FILE,
            pattern=r"rm.*-rf|delete.*recursive",
            description="Recursive file deletion",
            action=SafetyAction.REQUIRE_CONFIRMATION,
            severity=8,
            tool_names=[ToolNames.SHELL],
            confirmation_prompt="This will recursively delete files. Continue?",
        ),
        SafetyRule(
            rule_id="file_overwrite_system",
            category=SafetyCategory.FILE,
            pattern=r"/(etc|usr|bin|sbin)/",
            description="Write to system directory",
            action=SafetyAction.BLOCK,
            severity=10,
            tool_names=[ToolNames.WRITE, ToolNames.EDIT],
        ),
        SafetyRule(
            rule_id="shell_format_disk",
            category=SafetyCategory.SHELL,
            pattern=r"mkfs|format|fdisk",
            description="Disk formatting operation",
            action=SafetyAction.BLOCK,
            severity=10,
            tool_names=[ToolNames.SHELL],
        ),
        SafetyRule(
            rule_id="shell_remove_user",
            category=SafetyCategory.SHELL,
            pattern=r"userdel.*-r|user.*remove",
            description="Remove user account",
            action=SafetyAction.BLOCK,
            severity=9,
            tool_names=[ToolNames.SHELL],
        ),
        SafetyRule(
            rule_id="docker_rm_container",
            category=SafetyCategory.DOCKER,
            pattern=r"rm.*-f|container.*rm.*--force",
            description="Force remove Docker container",
            action=SafetyAction.WARN,
            severity=6,
            tool_names=["docker"],
        ),
        SafetyRule(
            rule_id="docker_rmi_image",
            category=SafetyCategory.DOCKER,
            pattern=r"rmi|image.*rm",
            description="Remove Docker image",
            action=SafetyAction.WARN,
            severity=5,
            tool_names=["docker"],
        ),
    ]

    def __init__(
        self,
        strict_mode: bool = False,
        enable_default_rules: bool = True,
        max_severity_to_allow: int = 10,
    ):
        self._strict_mode = strict_mode
        self._max_severity_to_allow = max_severity_to_allow
        self._rules: List[SafetyRule] = []
        self._rule_index: Dict[str, SafetyRule] = {}
        self._stats = SafetyStats()

        if enable_default_rules:
            for rule in self.DEFAULT_RULES:
                self.register_rule(rule)

        logger.debug(
            "SafetyCoordinator initialized with %s rules, strict_mode=%s",
            len(self._rules),
            strict_mode,
        )

    def register_rule(self, rule: SafetyRule) -> None:
        """Register a safety rule."""
        self._rules.append(rule)
        self._rule_index[rule.rule_id] = rule
        logger.debug("Registered safety rule: %s", rule.rule_id)

    def unregister_rule(self, rule_id: str) -> bool:
        """Unregister a safety rule."""
        if rule_id not in self._rule_index:
            return False

        rule = self._rule_index[rule_id]
        self._rules.remove(rule)
        del self._rule_index[rule_id]

        logger.debug("Unregistered safety rule: %s", rule_id)
        return True

    def get_rule(self, rule_id: str) -> Optional[SafetyRule]:
        """Get a safety rule by identifier."""
        return self._rule_index.get(rule_id)

    def list_rules(self, category: Optional[SafetyCategory] = None) -> List[SafetyRule]:
        """List registered safety rules."""
        if category:
            return [rule for rule in self._rules if rule.category == category]
        return self._rules.copy()

    def clear_rules(self) -> None:
        """Clear all safety rules."""
        self._rules.clear()
        self._rule_index.clear()
        logger.info("Cleared all safety rules")

    def check_safety(
        self,
        tool_name: str,
        args: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> SafetyCheckResult:
        """Check whether a tool operation is safe."""
        del context
        self._stats.total_checks += 1

        matched_rules: List[SafetyRule] = []
        warnings: List[str] = []

        for rule in self._rules:
            if rule.matches(tool_name, args):
                matched_rules.append(rule)
                self._stats.rule_hits[rule.rule_id] = self._stats.rule_hits.get(rule.rule_id, 0) + 1

                if rule.action == SafetyAction.WARN:
                    warnings.append(f"{rule.description} (severity: {rule.severity})")
                    self._stats.warned_operations += 1

        if not matched_rules:
            return SafetyCheckResult(is_safe=True, action=SafetyAction.ALLOW)

        matched_rules.sort(key=lambda rule: rule.severity, reverse=True)
        most_severe_rule = matched_rules[0]
        action = most_severe_rule.action

        if self._strict_mode and action == SafetyAction.WARN:
            action = SafetyAction.BLOCK

        if action != SafetyAction.BLOCK and most_severe_rule.severity > self._max_severity_to_allow:
            action = SafetyAction.BLOCK

        if action == SafetyAction.BLOCK:
            self._stats.blocked_operations += 1
            return SafetyCheckResult(
                is_safe=False,
                action=action,
                matched_rules=matched_rules,
                block_reason=most_severe_rule.description,
            )

        if action == SafetyAction.REQUIRE_CONFIRMATION:
            self._stats.confirmed_operations += 1
            return SafetyCheckResult(
                is_safe=False,
                action=action,
                matched_rules=matched_rules,
                warnings=warnings,
                confirmation_prompt=most_severe_rule.confirmation_prompt,
            )

        return SafetyCheckResult(
            is_safe=True,
            action=action,
            matched_rules=matched_rules,
            warnings=warnings,
        )

    def is_operation_safe(
        self,
        tool_name: str,
        args: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Return whether an operation is safe."""
        return self.check_safety(tool_name, args, context).is_safe

    def get_stats(self) -> SafetyStats:
        """Get safety statistics."""
        return self._stats

    def get_stats_dict(self) -> Dict[str, Any]:
        """Get statistics as a dictionary."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset collected statistics."""
        self._stats = SafetyStats()
        logger.info("Safety statistics reset")

    def set_strict_mode(self, strict: bool) -> None:
        """Set strict mode."""
        self._strict_mode = strict
        logger.info("Strict mode set to %s", strict)

    def set_max_severity_to_allow(self, severity: int) -> None:
        """Set maximum severity to allow without confirmation."""
        self._max_severity_to_allow = severity
        logger.info("Max severity to allow set to %s", severity)

    def get_observability_data(self) -> Dict[str, Any]:
        """Get observability data for dashboards and logs."""
        return {
            "source_id": f"coordinator:safety:{id(self)}",
            "source_type": "coordinator",
            "coordinator_type": "safety",
            "stats": self._stats.to_dict(),
            "config": {
                "strict_mode": self._strict_mode,
                "max_severity_to_allow": self._max_severity_to_allow,
                "total_rules": len(self._rules),
            },
        }


__all__ = [
    "SafetyAction",
    "SafetyCategory",
    "SafetyRule",
    "SafetyCheckResult",
    "SafetyStats",
    "SafetyCoordinator",
]
