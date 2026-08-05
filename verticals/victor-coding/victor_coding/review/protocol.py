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

"""Code review protocol types.

Defines data structures for automated code review.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ReviewSeverity(Enum):
    """Severity level for code review findings.

    Renamed from Severity to be semantically distinct from other severity types:
    - CVESeverity (victor.security.protocol): CVE/CVSS-based severity
    - AuditSeverity: Audit event severity (like log levels)
    - IaCSeverity: IaC issue severity
    - ReviewSeverity (here): Code review severity
    """

    ERROR = "error"  # Must fix, blocks merge
    WARNING = "warning"  # Should fix
    INFO = "info"  # Suggestion
    HINT = "hint"  # Nice to have


# Backward compatibility alias
Severity = ReviewSeverity


class ReviewCategory(Enum):
    """Categories of code review findings."""

    STYLE = "style"  # Code style and formatting
    COMPLEXITY = "complexity"  # Code complexity issues
    SECURITY = "security"  # Security vulnerabilities
    PERFORMANCE = "performance"  # Performance issues
    MAINTAINABILITY = "maintainability"  # Maintainability concerns
    DOCUMENTATION = "documentation"  # Missing or poor documentation
    TESTING = "testing"  # Test coverage issues
    ERROR_HANDLING = "error_handling"  # Error handling issues
    NAMING = "naming"  # Naming convention issues
    DUPLICATION = "duplication"  # Code duplication
    BEST_PRACTICES = "best_practices"  # General best practices


@dataclass
class SourceLocation:
    """Location in source code."""

    file_path: Path
    start_line: int
    start_column: int = 0
    end_line: Optional[int] = None
    end_column: Optional[int] = None

    def __post_init__(self):
        if self.end_line is None:
            self.end_line = self.start_line


@dataclass
class ReviewFinding:
    """A single review finding/issue."""

    rule_id: str
    message: str
    severity: ReviewSeverity
    category: ReviewCategory
    location: SourceLocation
    code_snippet: str = ""
    suggestion: str = ""
    fix_available: bool = False
    fix_code: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "file": str(self.location.file_path),
            "line": self.location.start_line,
            "column": self.location.start_column,
            "snippet": self.code_snippet,
            "suggestion": self.suggestion,
            "fix_available": self.fix_available,
        }


@dataclass
class ReviewRule:
    """A configurable review rule."""

    id: str
    name: str
    description: str
    category: ReviewCategory
    severity: ReviewSeverity = ReviewSeverity.WARNING
    enabled: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def matches_tags(self, tags: list[str]) -> bool:
        """Check if rule matches any of the given tags."""
        if not tags:
            return True
        return bool(set(self.tags) & set(tags))


@dataclass
class ReviewRuleSet:
    """A collection of review rules."""

    name: str
    description: str = ""
    rules: list[ReviewRule] = field(default_factory=list)
    extends: list[str] = field(default_factory=list)  # Base rulesets to extend

    def get_enabled_rules(self) -> list[ReviewRule]:
        """Get all enabled rules."""
        return [r for r in self.rules if r.enabled]

    def get_rules_by_category(
        self,
        category: ReviewCategory,
    ) -> list[ReviewRule]:
        """Get rules for a specific category."""
        return [r for r in self.rules if r.category == category and r.enabled]


@dataclass
class FileReview:
    """Review results for a single file."""

    file_path: Path
    findings: list[ReviewFinding] = field(default_factory=list)
    lines_analyzed: int = 0
    duration_ms: float = 0.0

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == ReviewSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == ReviewSeverity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == ReviewSeverity.INFO)


@dataclass
class ReviewResult:
    """Complete review result for multiple files."""

    file_reviews: list[FileReview] = field(default_factory=list)
    total_files: int = 0
    total_findings: int = 0
    ruleset_name: str = ""
    duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        """Check if review passed (no errors)."""
        return all(fr.error_count == 0 for fr in self.file_reviews)

    @property
    def total_errors(self) -> int:
        return sum(fr.error_count for fr in self.file_reviews)

    @property
    def total_warnings(self) -> int:
        return sum(fr.warning_count for fr in self.file_reviews)

    def get_findings_by_category(
        self,
        category: ReviewCategory,
    ) -> list[ReviewFinding]:
        """Get all findings for a category."""
        findings = []
        for fr in self.file_reviews:
            findings.extend(f for f in fr.findings if f.category == category)
        return findings

    def get_findings_by_severity(
        self,
        severity: ReviewSeverity,
    ) -> list[ReviewFinding]:
        """Get all findings for a severity level."""
        findings = []
        for fr in self.file_reviews:
            findings.extend(f for f in fr.findings if f.severity == severity)
        return findings


@dataclass
class ReviewConfig:
    """Configuration for code review."""

    enabled_categories: list[ReviewCategory] = field(default_factory=lambda: list(ReviewCategory))
    min_severity: ReviewSeverity = ReviewSeverity.INFO
    fail_on_error: bool = True
    fail_on_warning: bool = False
    max_findings_per_file: int = 100
    exclude_patterns: list[str] = field(default_factory=list)
    include_patterns: list[str] = field(default_factory=lambda: ["**/*.py"])
    custom_rules: list[ReviewRule] = field(default_factory=list)


@dataclass
class ComplexityMetrics:
    """Code complexity metrics."""

    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    lines_of_code: int = 0
    lines_of_comment: int = 0
    number_of_functions: int = 0
    number_of_classes: int = 0
    max_nesting_depth: int = 0
    average_function_length: float = 0.0
    maintainability_index: float = 100.0


@dataclass
class SecurityIssue:
    """A security-related finding."""

    vulnerability_type: str
    severity: ReviewSeverity
    cwe_id: Optional[str] = None
    description: str = ""
    remediation: str = ""
    confidence: float = 1.0  # 0-1


@dataclass
class DuplicationResult:
    """Result of duplicate code detection."""

    duplicate_blocks: list[tuple[SourceLocation, SourceLocation]] = field(default_factory=list)
    total_duplicate_lines: int = 0
    duplication_percentage: float = 0.0
