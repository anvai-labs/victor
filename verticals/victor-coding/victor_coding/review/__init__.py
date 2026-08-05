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

"""Automated code review module.

This module provides automated code review capabilities with
configurable rules and multiple analyzers.

Example usage:
    from victor_coding.review import get_review_manager, ReviewConfig
    from pathlib import Path

    # Get manager
    manager = get_review_manager()

    # Review a single file
    result = manager.review_file(Path("my_module.py"))

    # Review with specific ruleset
    result = manager.review_file(
        Path("my_module.py"),
        ruleset_name="strict",
    )

    # Review a directory
    result = manager.review_directory(
        Path("src/"),
        recursive=True,
    )

    # Review only changed files
    result = manager.review_diff(base_ref="main")

    # Format as report
    report = manager.format_report(result, format="markdown")
    print(report)

    # Configure rules
    from victor_coding.review import get_rule_registry
    registry = get_rule_registry()
    registry.configure_rule(
        "complexity-cyclomatic",
        parameters={"max": 15},
    )
"""

from victor_coding.review.protocol import (
    ComplexityMetrics,
    DuplicationResult,
    FileReview,
    ReviewCategory,
    ReviewConfig,
    ReviewFinding,
    ReviewResult,
    ReviewRule,
    ReviewRuleSet,
    SecurityIssue,
    Severity,
    SourceLocation,
)
from victor_coding.review.analyzers import (
    BaseAnalyzer,
    BestPracticesAnalyzer,
    ComplexityAnalyzer,
    DocumentationAnalyzer,
    NamingAnalyzer,
    SecurityAnalyzer,
)
from victor_coding.review.rules import (
    DEFAULT_RULES,
    DEFAULT_RULESETS,
    RuleRegistry,
    get_rule_registry,
    reset_rule_registry,
)
from victor_coding.review.manager import (
    ReviewManager,
    get_review_manager,
    reset_review_manager,
)

__all__ = [
    # Protocol types
    "ComplexityMetrics",
    "DuplicationResult",
    "FileReview",
    "ReviewCategory",
    "ReviewConfig",
    "ReviewFinding",
    "ReviewResult",
    "ReviewRule",
    "ReviewRuleSet",
    "SecurityIssue",
    "Severity",
    "SourceLocation",
    # Analyzers
    "BaseAnalyzer",
    "BestPracticesAnalyzer",
    "ComplexityAnalyzer",
    "DocumentationAnalyzer",
    "NamingAnalyzer",
    "SecurityAnalyzer",
    # Rules
    "DEFAULT_RULES",
    "DEFAULT_RULESETS",
    "RuleRegistry",
    "get_rule_registry",
    "reset_rule_registry",
    # Manager
    "ReviewManager",
    "get_review_manager",
    "reset_review_manager",
]
