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

"""Safety Extensions - ISP-compliant composite for safety-related protocols.

This module provides a focused extension for safety-related vertical capabilities:
- Safety patterns for dangerous operation detection
- Validators for custom safety checks

This replaces the safety-related parts of the monolithic VerticalExtensions class,
following Interface Segregation Principle (ISP).

Usage:
    from victor.core.verticals.extensions import SafetyExtensions
    from victor.security.safety.types import SafetyPattern

    safety_ext = SafetyExtensions(
        safety_patterns=[
            SafetyPattern(
                pattern=r"rm\\s+-rf",
                description="Recursive deletion",
                risk_level="HIGH",
                category="filesystem",
            ),
        ],
        validators=[CustomSecurityValidator()],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from victor.security.safety.types import SafetyPattern


@dataclass
class SafetyExtensions:
    """Focused extension for safety-related vertical capabilities.

    Groups safety patterns and validators - the safety-specific parts
    that were previously bundled in VerticalExtensions.

    Attributes:
        safety_patterns: List of SafetyPattern instances for dangerous
            operation detection. Patterns can be for bash commands,
            file operations, or tool-specific restrictions.
        validators: List of custom safety validators for domain-specific
            checks beyond pattern matching.

    Example:
        safety_ext = SafetyExtensions(
            safety_patterns=[
                SafetyPattern(
                    pattern=r"git\\s+push.*--force",
                    description="Force push may lose commits",
                    risk_level="HIGH",
                    category="git",
                ),
                SafetyPattern(
                    pattern=r"DROP\\s+DATABASE",
                    description="Database deletion",
                    risk_level="CRITICAL",
                    category="database",
                ),
            ],
            validators=[
                CredentialLeakValidator(),
                PathTraversalValidator(),
            ],
        )

        # Get all patterns
        patterns = safety_ext.get_all_patterns()

        # Get patterns by category
        git_patterns = safety_ext.get_patterns_by_category("git")

        # Get high-risk patterns
        high_risk = safety_ext.get_patterns_by_risk("HIGH")
    """

    safety_patterns: List[SafetyPattern] = field(default_factory=list)
    validators: List[Any] = field(default_factory=list)  # List[SafetyValidator]

    def get_all_patterns(self) -> List[SafetyPattern]:
        """Get all safety patterns.

        Returns:
            List of all SafetyPattern instances
        """
        return list(self.safety_patterns)

    def get_patterns_by_category(self, category: str) -> List[SafetyPattern]:
        """Get safety patterns for a specific category.

        Args:
            category: Category to filter by (e.g., "git", "filesystem", "database")

        Returns:
            List of patterns matching the category
        """
        return [p for p in self.safety_patterns if p.category == category]

    def get_patterns_by_risk(self, risk_level: str) -> List[SafetyPattern]:
        """Get safety patterns by risk level.

        Args:
            risk_level: Risk level to filter by (e.g., "LOW", "MEDIUM", "HIGH", "CRITICAL")

        Returns:
            List of patterns matching the risk level
        """
        return [p for p in self.safety_patterns if p.risk_level == risk_level]

    def get_categories(self) -> Set[str]:
        """Get all unique categories from patterns.

        Returns:
            Set of category names
        """
        return {p.category for p in self.safety_patterns}

    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Get patterns applicable to bash commands.

        Returns patterns in "bash", "shell", or "filesystem" categories.

        Returns:
            List of bash-related patterns
        """
        bash_categories = {"bash", "shell", "filesystem", "system"}
        return [p for p in self.safety_patterns if p.category in bash_categories]

    def get_file_patterns(self) -> List[SafetyPattern]:
        """Get patterns applicable to file operations.

        Returns:
            List of file operation patterns
        """
        file_categories = {"file", "filesystem", "path"}
        return [p for p in self.safety_patterns if p.category in file_categories]

    async def validate_operation(
        self,
        operation: str,
        context: Dict[str, Any],
    ) -> List[str]:
        """Run all validators on an operation.

        Args:
            operation: The operation to validate (command, file content, etc.)
            context: Additional context for validation

        Returns:
            List of validation error messages (empty if all pass)
        """
        errors: List[str] = []
        for validator in self.validators:
            if hasattr(validator, "validate"):
                try:
                    result = validator.validate(operation, context)
                    # Handle both sync and async validators
                    if hasattr(result, "__await__"):
                        result = await result
                    if result:
                        if isinstance(result, str):
                            errors.append(result)
                        elif isinstance(result, list):
                            errors.extend(result)
                except Exception as e:
                    errors.append(f"Validator {type(validator).__name__} failed: {e}")
        return errors

    def add_pattern(self, pattern: SafetyPattern) -> None:
        """Add a safety pattern.

        Args:
            pattern: SafetyPattern to add
        """
        self.safety_patterns.append(pattern)

    def add_patterns_from_extension(self, extension: Any) -> None:
        """Add patterns from a SafetyExtensionProtocol implementation.

        Args:
            extension: SafetyExtensionProtocol implementation
        """
        if hasattr(extension, "get_bash_patterns"):
            self.safety_patterns.extend(extension.get_bash_patterns())
        if hasattr(extension, "get_file_patterns"):
            self.safety_patterns.extend(extension.get_file_patterns())

    def merge(self, other: "SafetyExtensions") -> "SafetyExtensions":
        """Merge with another SafetyExtensions instance.

        Patterns and validators are concatenated (deduplicated by pattern string).

        Args:
            other: Another SafetyExtensions to merge from

        Returns:
            New SafetyExtensions with merged content
        """
        # Merge patterns (deduplicate by pattern string)
        seen_patterns = {p.pattern for p in self.safety_patterns}
        merged_patterns = list(self.safety_patterns)
        for p in other.safety_patterns:
            if p.pattern not in seen_patterns:
                merged_patterns.append(p)
                seen_patterns.add(p.pattern)

        # Merge validators (deduplicate by instance)
        seen_validators = set(id(v) for v in self.validators)
        merged_validators = list(self.validators)
        for v in other.validators:
            if id(v) not in seen_validators:
                merged_validators.append(v)
                seen_validators.add(id(v))

        return SafetyExtensions(
            safety_patterns=merged_patterns,
            validators=merged_validators,
        )

    def __bool__(self) -> bool:
        """Return True if any content is present."""
        return bool(self.safety_patterns or self.validators)


__all__ = ["SafetyExtensions"]
