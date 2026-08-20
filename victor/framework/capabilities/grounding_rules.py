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

"""Grounding rules capability for framework-level constraint definitions.

This module provides centralized grounding rules for common vertical constraints
that apply across multiple domains (coding, devops, research, dataanalysis).

Design Pattern: Capability Provider
- Centralized grounding rule definitions
- Category-based rule organization
- Vertical-specific rule extensions
- Consistent rule application across verticals

Integration Point:
    Use in VerticalBase.get_system_prompt() to append grounding rules section

Example:
    capability = GroundingRulesCapability()
    rules = capability.get_rules()

    # Get rules for specific vertical
    coding_rules = capability.get_vertical_rules("coding")

    # Get custom rules with extensions
    custom_rules = capability.get_custom_rules({
        "file_safety": ["Additional rule here"]
    })

Phase 1: Promote Generic Capabilities to Framework
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum


class RuleCategory(Enum):
    """Categories of grounding rules."""

    BASE = "base"  # Base rules that apply to all verticals
    FILE_SAFETY = "file_safety"  # File operation constraints
    GIT_SAFETY = "git_safety"  # Git operation constraints
    TEST_REQUIREMENTS = "test_requirements"  # Testing constraints
    PRIVACY = "privacy"  # Privacy and data protection
    TOOL_USAGE = "tool_usage"  # Tool usage constraints


@dataclass
class GroundingRule:
    """A single grounding rule definition.

    Attributes:
        rule_id: Unique identifier for the rule
        category: Category this rule belongs to
        text: The rule text to include in prompts
        verticals: Verticals this rule applies to (empty = all)
        priority: Rule priority for ordering (higher = more important)
    """

    rule_id: str
    category: RuleCategory
    text: str
    verticals: List[str] = field(default_factory=list)
    priority: int = 50

    def applies_to(self, vertical: str) -> bool:
        """Check if this rule applies to a specific vertical.

        Args:
            vertical: Vertical name

        Returns:
            True if rule applies, False otherwise
        """
        return not self.verticals or vertical in self.verticals

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "category": self.category.value,
            "text": self.text,
            "verticals": self.verticals,
            "priority": self.priority,
        }


class GroundingRulesCapability:
    """Generic grounding rules capability provider.

    Provides centralized grounding rules for common vertical constraints:
    - File operation safety (read before write)
    - Git operation constraints (no force push to main)
    - Test requirements (verify after changes)
    - Privacy rules (no sensitive data exposure)

    These rules ensure consistent behavior across verticals while allowing
    vertical-specific customization and extensions.

    Attributes:
        custom_rules: Optional custom rules to add/override defaults
        strict_mode: Whether to enable strict rule enforcement
    """

    # Base grounding rules (apply to all verticals)
    BASE_RULES: List[GroundingRule] = [
        GroundingRule(
            rule_id="ground_on_tool_output",
            category=RuleCategory.BASE,
            text="Base ALL responses on tool output only. Never invent file paths "
            "or content. Quote exactly from tool output. If more info needed, "
            "call another tool.",
            priority=100,
        ),
        GroundingRule(
            rule_id="acknowledge_uncertainty",
            category=RuleCategory.BASE,
            text="When tool output is unclear or incomplete, acknowledge this "
            "limitation explicitly. Do not guess or fabricate information.",
            priority=90,
        ),
    ]

    # File operation safety rules
    FILE_SAFETY_RULES: List[GroundingRule] = [
        GroundingRule(
            rule_id="read_before_write",
            category=RuleCategory.FILE_SAFETY,
            text="Always read a file before writing to it. Understand its current "
            "content and structure before making changes.",
            priority=95,
        ),
        GroundingRule(
            rule_id="verify_file_paths",
            category=RuleCategory.FILE_SAFETY,
            text="Verify file paths exist before using them. Use tools like 'ls' to "
            "check directory structure.",
            priority=80,
        ),
        GroundingRule(
            rule_id="preserve_existing_structure",
            category=RuleCategory.FILE_SAFETY,
            text="When modifying files, preserve the existing structure, formatting, "
            "and coding conventions unless explicitly asked to change them.",
            priority=70,
        ),
    ]

    # Git operation safety rules
    GIT_SAFETY_RULES: List[GroundingRule] = [
        GroundingRule(
            rule_id="no_force_push_main",
            category=RuleCategory.GIT_SAFETY,
            text="NEVER use --force or --force-with-lease when pushing to main, "
            "master, or production branches.",
            priority=100,
        ),
        GroundingRule(
            rule_id="check_branch_status",
            category=RuleCategory.GIT_SAFETY,
            text="Always check the current branch and its status before performing "
            "git operations.",
            priority=85,
        ),
        GroundingRule(
            rule_id="review_git_diff",
            category=RuleCategory.GIT_SAFETY,
            text="Review git diff before committing to ensure only intended changes "
            "are included.",
            priority=75,
        ),
    ]

    # Test requirements rules
    TEST_REQUIREMENT_RULES: List[GroundingRule] = [
        GroundingRule(
            rule_id="verify_after_changes",
            category=RuleCategory.TEST_REQUIREMENTS,
            text="After making changes, verify they work correctly by running tests "
            "or checking for errors.",
            priority=90,
        ),
        GroundingRule(
            rule_id="run_affected_tests",
            category=RuleCategory.TEST_REQUIREMENTS,
            text="Run tests that are affected by the changes made. Use test discovery "
            "to find relevant tests.",
            priority=80,
        ),
    ]

    # Privacy rules
    PRIVACY_RULES: List[GroundingRule] = [
        GroundingRule(
            rule_id="no_sensitive_data_exposure",
            category=RuleCategory.PRIVACY,
            text="Never expose sensitive data such as passwords, API keys, tokens, "
            "or personal information in responses.",
            priority=100,
        ),
        GroundingRule(
            rule_id="redact_secrets",
            category=RuleCategory.PRIVACY,
            text="When reading files that may contain secrets, redact or mask them "
            "in your responses.",
            priority=95,
        ),
    ]

    # Tool usage rules
    TOOL_USAGE_RULES: List[GroundingRule] = [
        GroundingRule(
            rule_id="use_appropriate_tools",
            category=RuleCategory.TOOL_USAGE,
            text="Choose the most appropriate tool for the task. Use 'read' to view "
            "files, 'grep' to search content, 'ls' to list directories.",
            priority=70,
        ),
        GroundingRule(
            rule_id="respect_tool_limits",
            category=RuleCategory.TOOL_USAGE,
            text="Respect tool output limits. If output is truncated, acknowledge this "
            "and request specific sections if needed.",
            priority=60,
        ),
    ]

    # Vertical-specific rule extensions
    VERTICAL_EXTENSIONS: Dict[str, List[str]] = {
        "coding": [
            "Follow existing code patterns and conventions in the project.",
            "Ensure code passes linting and formatting checks.",
        ],
        "devops": [
            "Always verify infrastructure configuration before applying changes.",
            "Check resource quotas and limits before provisioning.",
        ],
        "research": [
            "Always cite sources for information obtained from web searches.",
            "Verify information from multiple sources when possible.",
        ],
        "dataanalysis": [
            "Verify data types and schemas before analysis.",
            "Document data transformations and assumptions made.",
        ],
        "rag": [
            "Base responses on retrieved context only.",
            "Cite the specific documents or passages used in answers.",
        ],
    }

    def __init__(
        self,
        custom_rules: Optional[List[GroundingRule]] = None,
        strict_mode: bool = False,
    ):
        """Initialize the grounding rules capability.

        Args:
            custom_rules: Optional custom rules to add/override defaults
            strict_mode: Whether to enable strict rule enforcement
        """
        self._custom_rules = custom_rules or []
        self._strict_mode = strict_mode
        self._rule_cache: Optional[Dict[str, List[GroundingRule]]] = None

    def get_rules(self, category: Optional[RuleCategory] = None) -> List[GroundingRule]:
        """Get grounding rules, optionally filtered by category.

        Args:
            category: Optional category to filter by

        Returns:
            List of grounding rules, sorted by priority (descending)
        """
        rules = []

        # Add base rules
        rules.extend(self.BASE_RULES)
        rules.extend(self.FILE_SAFETY_RULES)
        rules.extend(self.GIT_SAFETY_RULES)
        rules.extend(self.TEST_REQUIREMENT_RULES)
        rules.extend(self.PRIVACY_RULES)
        rules.extend(self.TOOL_USAGE_RULES)

        # Add custom rules
        rules.extend(self._custom_rules)

        # Filter by category if specified
        if category:
            rules = [r for r in rules if r.category == category]

        # Sort by priority (descending)
        rules.sort(key=lambda r: r.priority, reverse=True)

        return rules

    def get_vertical_rules(
        self, vertical: str, category: Optional[RuleCategory] = None
    ) -> List[GroundingRule]:
        """Get rules that apply to a specific vertical.

        Args:
            vertical: Vertical name
            category: Optional category to filter by

        Returns:
            List of applicable grounding rules
        """
        all_rules = self.get_rules(category=category)
        return [r for r in all_rules if r.applies_to(vertical)]

    def get_rules_text(
        self, vertical: Optional[str] = None, include_extensions: bool = True
    ) -> str:
        """Get grounding rules as formatted text for prompts.

        Args:
            vertical: Optional vertical name for vertical-specific rules
            include_extensions: Whether to include vertical-specific extensions

        Returns:
            Formatted grounding rules text
        """
        if vertical:
            rules = self.get_vertical_rules(vertical)
        else:
            rules = self.get_rules()

        # Build rules text
        rule_lines = ["# Grounding Rules", ""]

        for rule in rules:
            rule_lines.append(f"- {rule.text}")

        # Add vertical-specific extensions if requested
        if include_extensions and vertical and vertical in self.VERTICAL_EXTENSIONS:
            rule_lines.append("")
            rule_lines.append(f"# Additional Rules for {vertical.title()}")
            rule_lines.append("")
            for extension in self.VERTICAL_EXTENSIONS[vertical]:
                rule_lines.append(f"- {extension}")

        return "\n".join(rule_lines)

    def get_custom_rules(self, overrides: Dict[str, List[str]]) -> Dict[str, List[GroundingRule]]:
        """Get custom rules with category overrides.

        Args:
            overrides: Dictionary mapping category names to rule text lists

        Returns:
            Dictionary of custom rules by category
        """
        result = {}

        for category_str, rule_texts in overrides.items():
            try:
                category = RuleCategory(category_str)
                rules = []
                for i, text in enumerate(rule_texts):
                    rule = GroundingRule(
                        rule_id=f"custom_{category_str}_{i}",
                        category=category,
                        text=text,
                        priority=50,
                    )
                    rules.append(rule)
                result[category_str] = rules
            except ValueError:
                # Invalid category, skip
                continue

        return result

    def add_rule(self, rule: GroundingRule) -> None:
        """Add a custom grounding rule.

        Args:
            rule: GroundingRule to add
        """
        self._custom_rules.append(rule)
        self._rule_cache = None  # Clear cache

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a custom grounding rule by ID.

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

    def get_rule(self, rule_id: str) -> Optional[GroundingRule]:
        """Get a specific rule by ID.

        Args:
            rule_id: Rule identifier

        Returns:
            GroundingRule or None if not found
        """
        all_rules = self.get_rules()
        for rule in all_rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    def list_categories(self) -> List[RuleCategory]:
        """List all available rule categories.

        Returns:
            List of RuleCategory enum values
        """
        return list(RuleCategory)

    def clear_cache(self) -> None:
        """Clear the rule cache."""
        self._rule_cache = None


# Pre-configured grounding rules for common vertical types
class GroundingRulesPresets:
    """Pre-configured grounding rules for common verticals."""

    @staticmethod
    def coding() -> GroundingRulesCapability:
        """Get grounding rules optimized for coding vertical."""
        custom_rules = [
            GroundingRule(
                rule_id="coding_preserve_patterns",
                category=RuleCategory.FILE_SAFETY,
                text="Follow existing code patterns and conventions in the project.",
                verticals=["coding"],
                priority=85,
            ),
            GroundingRule(
                rule_id="coding_lint_check",
                category=RuleCategory.TEST_REQUIREMENTS,
                text="Ensure code passes linting and formatting checks.",
                verticals=["coding"],
                priority=75,
            ),
        ]
        return GroundingRulesCapability(custom_rules=custom_rules)

    @staticmethod
    def devops() -> GroundingRulesCapability:
        """Get grounding rules optimized for DevOps vertical."""
        custom_rules = [
            GroundingRule(
                rule_id="devops_verify_config",
                category=RuleCategory.FILE_SAFETY,
                text="Always verify infrastructure configuration before applying changes.",
                verticals=["devops"],
                priority=90,
            ),
            GroundingRule(
                rule_id="devops_check_quotas",
                category=RuleCategory.TOOL_USAGE,
                text="Check resource quotas and limits before provisioning.",
                verticals=["devops"],
                priority=80,
            ),
        ]
        return GroundingRulesCapability(custom_rules=custom_rules)

    @staticmethod
    def research() -> GroundingRulesCapability:
        """Get grounding rules optimized for research vertical."""
        custom_rules = [
            GroundingRule(
                rule_id="research_cite_sources",
                category=RuleCategory.BASE,
                text="Always cite sources for information obtained from web searches.",
                verticals=["research"],
                priority=95,
            ),
            GroundingRule(
                rule_id="research_verify_sources",
                category=RuleCategory.BASE,
                text="Verify information from multiple sources when possible.",
                verticals=["research"],
                priority=85,
            ),
        ]
        return GroundingRulesCapability(custom_rules=custom_rules)

    @staticmethod
    def data_analysis() -> GroundingRulesCapability:
        """Get grounding rules optimized for data analysis vertical."""
        custom_rules = [
            GroundingRule(
                rule_id="data_verify_schema",
                category=RuleCategory.FILE_SAFETY,
                text="Verify data types and schemas before analysis.",
                verticals=["dataanalysis"],
                priority=90,
            ),
            GroundingRule(
                rule_id="data_document_transformations",
                category=RuleCategory.TOOL_USAGE,
                text="Document data transformations and assumptions made.",
                verticals=["dataanalysis"],
                priority=75,
            ),
        ]
        return GroundingRulesCapability(custom_rules=custom_rules)

    @staticmethod
    def rag() -> GroundingRulesCapability:
        """Get grounding rules optimized for RAG vertical."""
        custom_rules = [
            GroundingRule(
                rule_id="rag_base_on_context",
                category=RuleCategory.BASE,
                text="Base responses on retrieved context only.",
                verticals=["rag"],
                priority=100,
            ),
            GroundingRule(
                rule_id="rag_cite_documents",
                category=RuleCategory.BASE,
                text="Cite the specific documents or passages used in answers.",
                verticals=["rag"],
                priority=95,
            ),
        ]
        return GroundingRulesCapability(custom_rules=custom_rules)


__all__ = [
    "GroundingRulesCapability",
    "GroundingRulesPresets",
    "RuleCategory",
    "GroundingRule",
]
