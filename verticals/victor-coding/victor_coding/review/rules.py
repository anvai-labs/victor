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

"""Built-in review rules and rule management.

Provides default rulesets and rule configuration.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from victor_coding.review.protocol import (
    ReviewCategory,
    ReviewRule,
    ReviewRuleSet,
    Severity,
)

logger = logging.getLogger(__name__)


# Default rules organized by category
DEFAULT_RULES: list[ReviewRule] = [
    # Complexity rules
    ReviewRule(
        id="complexity-cyclomatic",
        name="Cyclomatic Complexity",
        description="Checks cyclomatic complexity of functions",
        category=ReviewCategory.COMPLEXITY,
        severity=Severity.WARNING,
        parameters={"max": 10},
        tags=["complexity", "maintainability"],
    ),
    ReviewRule(
        id="complexity-function-length",
        name="Function Length",
        description="Checks function length in lines",
        category=ReviewCategory.COMPLEXITY,
        severity=Severity.WARNING,
        parameters={"max": 50},
        tags=["complexity", "readability"],
    ),
    ReviewRule(
        id="complexity-nesting",
        name="Nesting Depth",
        description="Checks maximum nesting depth",
        category=ReviewCategory.COMPLEXITY,
        severity=Severity.WARNING,
        parameters={"max": 4},
        tags=["complexity", "readability"],
    ),
    ReviewRule(
        id="complexity-parameters",
        name="Function Parameters",
        description="Checks number of function parameters",
        category=ReviewCategory.COMPLEXITY,
        severity=Severity.INFO,
        parameters={"max": 5},
        tags=["complexity", "api"],
    ),
    ReviewRule(
        id="complexity-class-methods",
        name="Class Methods Count",
        description="Checks number of methods in a class",
        category=ReviewCategory.COMPLEXITY,
        severity=Severity.INFO,
        parameters={"max": 20},
        tags=["complexity", "design"],
    ),
    # Naming rules
    ReviewRule(
        id="naming-class-case",
        name="Class Naming",
        description="Classes should use PascalCase",
        category=ReviewCategory.NAMING,
        severity=Severity.WARNING,
        tags=["naming", "pep8"],
    ),
    ReviewRule(
        id="naming-function-case",
        name="Function Naming",
        description="Functions should use snake_case",
        category=ReviewCategory.NAMING,
        severity=Severity.WARNING,
        tags=["naming", "pep8"],
    ),
    ReviewRule(
        id="naming-variable-case",
        name="Variable Naming",
        description="Variables should use snake_case or UPPER_CASE",
        category=ReviewCategory.NAMING,
        severity=Severity.INFO,
        tags=["naming", "pep8"],
    ),
    ReviewRule(
        id="naming-function-length",
        name="Function Name Length",
        description="Function names should be descriptive but not too long",
        category=ReviewCategory.NAMING,
        severity=Severity.INFO,
        parameters={"min": 2, "max": 30},
        tags=["naming", "readability"],
    ),
    ReviewRule(
        id="naming-single-char",
        name="Single Character Variables",
        description="Avoid single character variable names except common ones",
        category=ReviewCategory.NAMING,
        severity=Severity.INFO,
        parameters={"allowed": ["i", "j", "k", "x", "y", "z", "_"]},
        tags=["naming", "readability"],
    ),
    # Documentation rules
    ReviewRule(
        id="doc-module-docstring",
        name="Module Docstring",
        description="Modules should have a docstring",
        category=ReviewCategory.DOCUMENTATION,
        severity=Severity.INFO,
        tags=["documentation"],
    ),
    ReviewRule(
        id="doc-class-docstring",
        name="Class Docstring",
        description="Public classes should have a docstring",
        category=ReviewCategory.DOCUMENTATION,
        severity=Severity.WARNING,
        tags=["documentation"],
    ),
    ReviewRule(
        id="doc-function-docstring",
        name="Function Docstring",
        description="Public functions should have a docstring",
        category=ReviewCategory.DOCUMENTATION,
        severity=Severity.WARNING,
        tags=["documentation"],
    ),
    ReviewRule(
        id="doc-function-params",
        name="Document Parameters",
        description="Function parameters should be documented",
        category=ReviewCategory.DOCUMENTATION,
        severity=Severity.INFO,
        tags=["documentation"],
    ),
    ReviewRule(
        id="doc-function-return",
        name="Document Return Value",
        description="Function return values should be documented",
        category=ReviewCategory.DOCUMENTATION,
        severity=Severity.INFO,
        tags=["documentation"],
    ),
    # Security rules
    ReviewRule(
        id="security-dangerous-calls",
        name="Dangerous Function Calls",
        description="Detect potentially dangerous function calls",
        category=ReviewCategory.SECURITY,
        severity=Severity.ERROR,
        tags=["security", "injection"],
    ),
    ReviewRule(
        id="security-sql-injection",
        name="SQL Injection",
        description="Detect potential SQL injection vulnerabilities",
        category=ReviewCategory.SECURITY,
        severity=Severity.ERROR,
        tags=["security", "injection", "sql"],
    ),
    ReviewRule(
        id="security-hardcoded-secrets",
        name="Hardcoded Secrets",
        description="Detect hardcoded passwords, keys, and tokens",
        category=ReviewCategory.SECURITY,
        severity=Severity.ERROR,
        tags=["security", "secrets"],
    ),
    # Best practices rules
    ReviewRule(
        id="bp-bare-except",
        name="Bare Except",
        description="Avoid bare except clauses",
        category=ReviewCategory.BEST_PRACTICES,
        severity=Severity.WARNING,
        tags=["best-practices", "exception"],
    ),
    ReviewRule(
        id="bp-broad-except",
        name="Broad Exception",
        description="Avoid catching broad exceptions silently",
        category=ReviewCategory.BEST_PRACTICES,
        severity=Severity.WARNING,
        tags=["best-practices", "exception"],
    ),
    ReviewRule(
        id="bp-wildcard-import",
        name="Wildcard Import",
        description="Avoid wildcard imports",
        category=ReviewCategory.BEST_PRACTICES,
        severity=Severity.WARNING,
        tags=["best-practices", "imports"],
    ),
    ReviewRule(
        id="bp-none-comparison",
        name="None Comparison",
        description="Use 'is None' instead of '== None'",
        category=ReviewCategory.BEST_PRACTICES,
        severity=Severity.INFO,
        tags=["best-practices", "style"],
    ),
    ReviewRule(
        id="bp-bool-comparison",
        name="Boolean Comparison",
        description="Avoid explicit True/False comparisons",
        category=ReviewCategory.BEST_PRACTICES,
        severity=Severity.INFO,
        tags=["best-practices", "style"],
    ),
]


# Pre-defined rulesets
DEFAULT_RULESETS: dict[str, ReviewRuleSet] = {
    "minimal": ReviewRuleSet(
        name="minimal",
        description="Minimal ruleset for quick checks",
        rules=[r for r in DEFAULT_RULES if r.severity == Severity.ERROR],
    ),
    "standard": ReviewRuleSet(
        name="standard",
        description="Standard ruleset for most projects",
        rules=[r for r in DEFAULT_RULES if r.severity in (Severity.ERROR, Severity.WARNING)],
    ),
    "strict": ReviewRuleSet(
        name="strict",
        description="Strict ruleset for high-quality code",
        rules=DEFAULT_RULES.copy(),
    ),
    "security": ReviewRuleSet(
        name="security",
        description="Security-focused ruleset",
        rules=[r for r in DEFAULT_RULES if r.category == ReviewCategory.SECURITY],
    ),
    "documentation": ReviewRuleSet(
        name="documentation",
        description="Documentation-focused ruleset",
        rules=[r for r in DEFAULT_RULES if r.category == ReviewCategory.DOCUMENTATION],
    ),
}


class RuleRegistry:
    """Registry for review rules.

    Manages loading, configuration, and retrieval of rules.
    """

    def __init__(self):
        """Initialize the registry with default rules."""
        self._rules: dict[str, ReviewRule] = {r.id: r for r in DEFAULT_RULES}
        self._rulesets: dict[str, ReviewRuleSet] = DEFAULT_RULESETS.copy()

    def get_rule(self, rule_id: str) -> Optional[ReviewRule]:
        """Get a rule by ID.

        Args:
            rule_id: Rule identifier

        Returns:
            ReviewRule or None
        """
        return self._rules.get(rule_id)

    def register_rule(self, rule: ReviewRule) -> None:
        """Register a custom rule.

        Args:
            rule: Rule to register
        """
        self._rules[rule.id] = rule
        logger.debug(f"Registered rule: {rule.id}")

    def get_ruleset(self, name: str) -> Optional[ReviewRuleSet]:
        """Get a ruleset by name.

        Args:
            name: Ruleset name

        Returns:
            ReviewRuleSet or None
        """
        return self._rulesets.get(name)

    def register_ruleset(self, ruleset: ReviewRuleSet) -> None:
        """Register a custom ruleset.

        Args:
            ruleset: Ruleset to register
        """
        self._rulesets[ruleset.name] = ruleset
        logger.debug(f"Registered ruleset: {ruleset.name}")

    def get_rules_by_category(
        self,
        category: ReviewCategory,
    ) -> list[ReviewRule]:
        """Get all rules for a category.

        Args:
            category: Review category

        Returns:
            List of rules
        """
        return [r for r in self._rules.values() if r.category == category]

    def get_rules_by_tags(self, tags: list[str]) -> list[ReviewRule]:
        """Get rules matching any of the given tags.

        Args:
            tags: Tags to match

        Returns:
            List of matching rules
        """
        return [r for r in self._rules.values() if r.matches_tags(tags)]

    def load_from_yaml(self, path: Path) -> None:
        """Load rules from a YAML file.

        Args:
            path: Path to YAML file

        Expected format:
        ```yaml
        rules:
          - id: custom-rule
            name: Custom Rule
            description: A custom rule
            category: security
            severity: error
            parameters:
              max: 10
            tags: [custom, security]

        rulesets:
          - name: custom-set
            description: Custom ruleset
            extends: [standard]
            rules: [custom-rule, security-dangerous-calls]
            disable: [bp-none-comparison]
        ```
        """
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load rules from {path}: {e}")
            return

        # Load rules
        for rule_data in data.get("rules", []):
            rule = self._parse_rule(rule_data)
            if rule:
                self.register_rule(rule)

        # Load rulesets
        for ruleset_data in data.get("rulesets", []):
            ruleset = self._parse_ruleset(ruleset_data)
            if ruleset:
                self.register_ruleset(ruleset)

    def _parse_rule(self, data: dict[str, Any]) -> Optional[ReviewRule]:
        """Parse a rule from dictionary data."""
        try:
            return ReviewRule(
                id=data["id"],
                name=data["name"],
                description=data.get("description", ""),
                category=ReviewCategory(data["category"]),
                severity=Severity(data.get("severity", "warning")),
                enabled=data.get("enabled", True),
                parameters=data.get("parameters", {}),
                tags=data.get("tags", []),
            )
        except Exception as e:
            logger.warning(f"Failed to parse rule: {e}")
            return None

    def _parse_ruleset(self, data: dict[str, Any]) -> Optional[ReviewRuleSet]:
        """Parse a ruleset from dictionary data."""
        try:
            rules = []

            # Start with extended rulesets
            for base_name in data.get("extends", []):
                base_set = self.get_ruleset(base_name)
                if base_set:
                    rules.extend(base_set.rules)

            # Add specified rules
            for rule_id in data.get("rules", []):
                rule = self.get_rule(rule_id)
                if rule and rule not in rules:
                    rules.append(rule)

            # Remove disabled rules
            disabled = set(data.get("disable", []))
            rules = [r for r in rules if r.id not in disabled]

            return ReviewRuleSet(
                name=data["name"],
                description=data.get("description", ""),
                rules=rules,
                extends=data.get("extends", []),
            )
        except Exception as e:
            logger.warning(f"Failed to parse ruleset: {e}")
            return None

    def configure_rule(
        self,
        rule_id: str,
        enabled: Optional[bool] = None,
        severity: Optional[Severity] = None,
        parameters: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Configure an existing rule.

        Args:
            rule_id: Rule to configure
            enabled: Whether rule is enabled
            severity: New severity level
            parameters: New parameters

        Returns:
            True if rule was configured
        """
        rule = self._rules.get(rule_id)
        if not rule:
            return False

        if enabled is not None:
            rule.enabled = enabled
        if severity is not None:
            rule.severity = severity
        if parameters is not None:
            rule.parameters.update(parameters)

        return True

    def export_to_yaml(self, path: Path) -> None:
        """Export rules to a YAML file.

        Args:
            path: Output path
        """
        data = {
            "rules": [
                {
                    "id": r.id,
                    "name": r.name,
                    "description": r.description,
                    "category": r.category.value,
                    "severity": r.severity.value,
                    "enabled": r.enabled,
                    "parameters": r.parameters,
                    "tags": r.tags,
                }
                for r in self._rules.values()
            ],
            "rulesets": [
                {
                    "name": rs.name,
                    "description": rs.description,
                    "extends": rs.extends,
                    "rules": [r.id for r in rs.rules],
                }
                for rs in self._rulesets.values()
            ],
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


# Global registry singleton
_rule_registry: Optional[RuleRegistry] = None


def get_rule_registry() -> RuleRegistry:
    """Get the global rule registry."""
    global _rule_registry
    if _rule_registry is None:
        _rule_registry = RuleRegistry()
    return _rule_registry


def reset_rule_registry() -> None:
    """Reset the global registry."""
    global _rule_registry
    _rule_registry = None
