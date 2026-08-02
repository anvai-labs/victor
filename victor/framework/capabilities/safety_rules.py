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

"""Safety rules capability for framework-level safety pattern definitions.

This module provides reusable safety pattern definitions for common operations
that can be used across verticals.

Design Pattern: Capability Provider + Rule Engine
- Centralized safety pattern definitions
- Category-based safety rules
- Pattern matching for dangerous operations
- Configurable strictness levels

Integration Point:
    Wire into existing SafetyProvider protocol

Example:
    capability = SafetyRulesCapabilityProvider()
    patterns = capability.get_patterns()

    # Check if an operation is safe
    is_safe = capability.is_operation_safe("git", ["push", "--force"])

Phase 1: Promote Generic Capabilities to Framework
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum


class SafetyCategory(Enum):
    """Categories of safety rules."""

    GIT = "git"  # Git operation safety
    FILE = "file"  # File operation safety
    SHELL = "shell"  # Shell command safety
    DOCKER = "docker"  # Docker operation safety
    NETWORK = "network"  # Network operation safety
    SYSTEM = "system"  # System-level operations


class SafetyAction(Enum):
    """Actions to take when safety rule is triggered."""

    ALLOW = "allow"  # Allow the operation
    WARN = "warn"  # Allow with warning
    BLOCK = "block"  # Block the operation
    REQUIRE_CONFIRMATION = "require_confirmation"  # Require user confirmation


@dataclass
class SafetyRule:
    """A safety rule definition.

    Attributes:
        rule_id: Unique identifier for the rule
        category: Category this rule belongs to
        pattern: Pattern to match (regex for commands, tool name for tools)
        description: Human-readable description
        action: Action to take when rule is triggered
        severity: Rule severity (1-10, higher = more severe)
        tool_names: Tool names this rule applies to
        confirmation_prompt: Prompt for user confirmation (if action is REQUIRE_CONFIRMATION)
    """

    rule_id: str
    category: SafetyCategory
    pattern: str
    description: str
    action: SafetyAction
    severity: int = 5
    tool_names: List[str] = field(default_factory=list)
    confirmation_prompt: str = ""

    def matches(self, tool_name: str, args: List[str]) -> bool:
        """Check if this rule matches the given tool call.

        Args:
            tool_name: Name of the tool being called
            args: Arguments to the tool

        Returns:
            True if the rule matches, False otherwise
        """
        # Check tool name
        if self.tool_names and tool_name not in self.tool_names:
            return False

        # Check pattern match against arguments
        args_str = " ".join(args)
        try:
            return bool(re.search(self.pattern, args_str))
        except re.error:
            # Invalid regex, treat as literal match
            return self.pattern in args_str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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


class SafetyRulesCapabilityProvider:
    """Generic safety rules capability provider.

    Provides reusable safety pattern definitions for:
    - Git operation safety (force push, main branch operations)
    - File operation safety (destructive operations)
    - Shell command safety (rm, format, etc.)
    - Docker operation safety (container deletion)

    These rules ensure consistent safety behavior across verticals while
    allowing customization and extensions.

    Attributes:
        strict_mode: Whether to enable strict safety enforcement
        custom_rules: Optional custom rules to add/override defaults
    """

    # Git safety rules
    GIT_RULES: List[SafetyRule] = [
        SafetyRule(
            rule_id="git_force_push_main",
            category=SafetyCategory.GIT,
            pattern=r"(push|fetch)\s+.*(-f|--force|--force-with-lease)",
            description="Force pushing to main/master/production branches",
            action=SafetyAction.BLOCK,
            severity=10,
            tool_names=["git"],
        ),
        SafetyRule(
            rule_id="git_delete_branch_main",
            category=SafetyCategory.GIT,
            pattern=r"branch\s+(-d|-D|--delete)\s+(main|master|production|prod)",
            description="Deleting main/master/production branch",
            action=SafetyAction.BLOCK,
            severity=10,
            tool_names=["git"],
        ),
        SafetyRule(
            rule_id="git_reset_hard",
            category=SafetyCategory.GIT,
            pattern=r"reset\s+--hard",
            description="Hard reset (discards all uncommitted changes)",
            action=SafetyAction.WARN,
            severity=8,
            tool_names=["git"],
        ),
        SafetyRule(
            rule_id="git_clean_force",
            category=SafetyCategory.GIT,
            pattern=r"clean\s+.*(-f|-fd|--force)",
            description="Force clean (removes untracked files)",
            action=SafetyAction.WARN,
            severity=6,
            tool_names=["git"],
        ),
        SafetyRule(
            rule_id="git_rebase_interactive",
            category=SafetyCategory.GIT,
            pattern=r"rebase\s+.*(-i|--interactive)",
            description="Interactive rebase (can rewrite history)",
            action=SafetyAction.WARN,
            severity=5,
            tool_names=["git"],
        ),
    ]

    # File operation safety rules
    FILE_RULES: List[SafetyRule] = [
        SafetyRule(
            rule_id="file_delete_recursive",
            category=SafetyCategory.FILE,
            pattern=r"rm\s+.*(-r|-rf|--recursive)",
            description="Recursive file deletion",
            action=SafetyAction.WARN,
            severity=8,
            tool_names=["shell"],
        ),
        SafetyRule(
            rule_id="file_overwrite_force",
            category=SafetyCategory.FILE,
            pattern=r"(write|edit).*--force",
            description="Force file overwrite",
            action=SafetyAction.WARN,
            severity=6,
            tool_names=["write", "edit"],
        ),
        SafetyRule(
            rule_id="file_chmod_critical",
            category=SafetyCategory.FILE,
            pattern=r"chmod\s+.*(000|777)",
            description="Setting dangerous permissions (no access or world-writable)",
            action=SafetyAction.WARN,
            severity=7,
            tool_names=["shell"],
        ),
    ]

    # Shell command safety rules
    SHELL_RULES: List[SafetyRule] = [
        SafetyRule(
            rule_id="shell_remove_system",
            category=SafetyCategory.SHELL,
            pattern=r"rm\s+(-rf|/|/\s|$)",
            description="Removing system files or root directory",
            action=SafetyAction.BLOCK,
            severity=10,
            tool_names=["shell"],
        ),
        SafetyRule(
            rule_id="shell_format_disk",
            category=SafetyCategory.SHELL,
            pattern=r"(mkfs|format)\s+",
            description="Formatting a disk",
            action=SafetyAction.REQUIRE_CONFIRMATION,
            severity=10,
            tool_names=["shell"],
            confirmation_prompt="This will format a disk. All data on the disk will be lost. Continue?",
        ),
        SafetyRule(
            rule_id="shell_kill_process",
            category=SafetyCategory.SHELL,
            pattern=r"kill\s+.*(-9|-KILL)",
            description="Force killing a process",
            action=SafetyAction.WARN,
            severity=6,
            tool_names=["shell"],
        ),
        SafetyRule(
            rule_id="shell_modify_system",
            category=SafetyCategory.SHELL,
            pattern=r"(iptables|sysctl|modprobe|insmod)",
            description="Modifying system configuration",
            action=SafetyAction.WARN,
            severity=7,
            tool_names=["shell"],
        ),
    ]

    # Docker operation safety rules
    DOCKER_RULES: List[SafetyRule] = [
        SafetyRule(
            rule_id="docker_remove_container_force",
            category=SafetyCategory.DOCKER,
            pattern=r"(rm|container\s+rm)\s+.*(-f|--force)",
            description="Force removing a running container",
            action=SafetyAction.WARN,
            severity=7,
            tool_names=["docker", "shell"],
        ),
        SafetyRule(
            rule_id="docker_remove_all",
            category=SafetyCategory.DOCKER,
            pattern=r"(rm|container\s+rm)\s+.*\$(docker\s+ps\s+-aq)",
            description="Removing all containers",
            action=SafetyAction.REQUIRE_CONFIRMATION,
            severity=9,
            tool_names=["docker", "shell"],
            confirmation_prompt="This will remove ALL containers. Continue?",
        ),
        SafetyRule(
            rule_id="docker_prune",
            category=SafetyCategory.DOCKER,
            pattern=r"system\s+prune|volume\s+prune|image\s+prune",
            description="Pruning Docker resources (removes unused data)",
            action=SafetyAction.WARN,
            severity=6,
            tool_names=["docker", "shell"],
        ),
        SafetyRule(
            rule_id="docker_network_host",
            category=SafetyCategory.DOCKER,
            pattern=r"run\s+.*--network\s+host",
            description="Running container with host network (security risk)",
            action=SafetyAction.WARN,
            severity=7,
            tool_names=["docker", "shell"],
        ),
    ]

    # Network operation safety rules
    NETWORK_RULES: List[SafetyRule] = [
        SafetyRule(
            rule_id="network_port_scan",
            category=SafetyCategory.NETWORK,
            pattern=r"(nmap|netcat|nc)\s+.*(-sS|-p-)",
            description="Port scanning (may be detected as intrusion)",
            action=SafetyAction.WARN,
            severity=5,
            tool_names=["shell"],
        ),
        SafetyRule(
            rule_id="network_download_untrusted",
            category=SafetyCategory.NETWORK,
            pattern=r"(curl|wget)\s+.*(http://|--insecure)",
            description="Downloading from untrusted HTTP sources",
            action=SafetyAction.WARN,
            severity=6,
            tool_names=["shell"],
        ),
    ]

    def __init__(
        self,
        custom_rules: Optional[List[SafetyRule]] = None,
        strict_mode: bool = False,
    ):
        """Initialize the safety rules capability provider.

        Args:
            custom_rules: Optional custom rules to add/override defaults
            strict_mode: Whether to enable strict safety enforcement
        """
        self._custom_rules = custom_rules or []
        self._strict_mode = strict_mode
        self._rule_cache: Optional[Dict[SafetyCategory, List[SafetyRule]]] = None

    def get_rules(self, category: Optional[SafetyCategory] = None) -> List[SafetyRule]:
        """Get safety rules, optionally filtered by category.

        Args:
            category: Optional category to filter by

        Returns:
            List of safety rules, sorted by severity (descending)
        """
        if self._rule_cache is None:
            self._build_cache()

        if category:
            return self._rule_cache.get(category, []).copy()
        else:
            # Return all rules
            all_rules = []
            for rules in self._rule_cache.values():
                all_rules.extend(rules)
            # Sort by severity
            all_rules.sort(key=lambda r: r.severity, reverse=True)
            return all_rules

    def _build_cache(self) -> None:
        """Build the rule cache."""
        self._rule_cache = {
            SafetyCategory.GIT: self.GIT_RULES.copy(),
            SafetyCategory.FILE: self.FILE_RULES.copy(),
            SafetyCategory.SHELL: self.SHELL_RULES.copy(),
            SafetyCategory.DOCKER: self.DOCKER_RULES.copy(),
            SafetyCategory.NETWORK: self.NETWORK_RULES.copy(),
        }

        # Add custom rules
        for rule in self._custom_rules:
            if rule.category not in self._rule_cache:
                self._rule_cache[rule.category] = []
            self._rule_cache[rule.category].append(rule)

    def check_operation(self, tool_name: str, args: List[str]) -> Tuple[bool, List[SafetyRule]]:
        """Check if an operation is safe.

        Args:
            tool_name: Name of the tool being called
            args: Arguments to the tool

        Returns:
            Tuple of (is_safe, matching_rules)
        """
        all_rules = self.get_rules()
        matching_rules = []

        for rule in all_rules:
            if rule.matches(tool_name, args):
                matching_rules.append(rule)

        # Operation is safe if no BLOCK rules match
        has_block = any(r.action == SafetyAction.BLOCK for r in matching_rules)
        return (not has_block, matching_rules)

    def is_operation_safe(self, tool_name: str, args: List[str]) -> bool:
        """Check if an operation is safe (simplified).

        Args:
            tool_name: Name of the tool being called
            args: Arguments to the tool

        Returns:
            True if safe, False if blocked
        """
        is_safe, _ = self.check_operation(tool_name, args)
        return is_safe

    def get_required_confirmations(self, tool_name: str, args: List[str]) -> List[str]:
        """Get confirmation prompts for an operation.

        Args:
            tool_name: Name of the tool being called
            args: Arguments to the tool

        Returns:
            List of confirmation prompts
        """
        _, matching_rules = self.check_operation(tool_name, args)
        confirmations = []

        for rule in matching_rules:
            if rule.action == SafetyAction.REQUIRE_CONFIRMATION and rule.confirmation_prompt:
                confirmations.append(rule.confirmation_prompt)

        return confirmations

    def get_warnings(self, tool_name: str, args: List[str]) -> List[str]:
        """Get warning messages for an operation.

        Args:
            tool_name: Name of the tool being called
            args: Arguments to the tool

        Returns:
            List of warning messages
        """
        _, matching_rules = self.check_operation(tool_name, args)
        warnings = []

        for rule in matching_rules:
            if rule.action in (SafetyAction.WARN, SafetyAction.REQUIRE_CONFIRMATION):
                warnings.append(f"{rule.description} (severity: {rule.severity}/10)")

        return warnings

    def add_rule(self, rule: SafetyRule) -> None:
        """Add a custom safety rule.

        Args:
            rule: SafetyRule to add
        """
        self._custom_rules.append(rule)
        self._rule_cache = None  # Clear cache

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a custom safety rule by ID.

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

    def get_rule(self, rule_id: str) -> Optional[SafetyRule]:
        """Get a specific rule by ID.

        Args:
            rule_id: Rule identifier

        Returns:
            SafetyRule or None if not found
        """
        all_rules = self.get_rules()
        for rule in all_rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    def list_categories(self) -> List[SafetyCategory]:
        """List all available safety categories.

        Returns:
            List of SafetyCategory enum values
        """
        return list(SafetyCategory)

    def clear_cache(self) -> None:
        """Clear the rule cache."""
        self._rule_cache = None


# Pre-configured safety rules for common vertical types
class SafetyRulesPresets:
    """Pre-configured safety rules for common verticals."""

    @staticmethod
    def coding() -> SafetyRulesCapabilityProvider:
        """Get safety rules optimized for coding vertical."""
        # Coding uses git, file, and shell rules
        return SafetyRulesCapabilityProvider()

    @staticmethod
    def devops() -> SafetyRulesCapabilityProvider:
        """Get safety rules optimized for DevOps vertical."""
        # DevOps uses all categories, especially docker and system operations
        custom_rules = [
            SafetyRule(
                rule_id="devops_kubectl_delete_all",
                category=SafetyCategory.SYSTEM,
                pattern=r"kubectl\s+delete\s+.*--all",
                description="Deleting all Kubernetes resources",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=10,
                tool_names=["shell", "kubectl"],
                confirmation_prompt="This will delete ALL Kubernetes resources in the namespace. Continue?",
            ),
        ]
        return SafetyRulesCapabilityProvider(custom_rules=custom_rules)

    @staticmethod
    def research() -> SafetyRulesCapabilityProvider:
        """Get safety rules optimized for research vertical."""
        # Research mostly uses network operations
        return SafetyRulesCapabilityProvider()

    @staticmethod
    def local_mode() -> SafetyRulesCapabilityProvider:
        """Get safety rules for local development mode (less strict)."""
        # In local mode, we might want to allow more operations with warnings
        return SafetyRulesCapabilityProvider(strict_mode=False)

    @staticmethod
    def production_mode() -> SafetyRulesCapabilityProvider:
        """Get safety rules for production mode (very strict)."""
        # In production mode, enable strict enforcement
        return SafetyRulesCapabilityProvider(strict_mode=True)


__all__ = [
    "SafetyRulesCapabilityProvider",
    "SafetyRulesPresets",
    "SafetyCategory",
    "SafetyAction",
    "SafetyRule",
]
