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

"""Code review manager for orchestrating automated reviews.

Provides high-level API for code review operations.
"""

import logging
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

from victor_coding.review.analyzers import (
    BaseAnalyzer,
    BestPracticesAnalyzer,
    ComplexityAnalyzer,
    DocumentationAnalyzer,
    NamingAnalyzer,
    SecurityAnalyzer,
)
from victor_coding.review.protocol import (
    FileReview,
    ReviewCategory,
    ReviewConfig,
    ReviewFinding,
    ReviewResult,
    ReviewRule,
    Severity,
)
from victor_coding.review.rules import RuleRegistry, get_rule_registry

logger = logging.getLogger(__name__)


class ReviewManager:
    """High-level manager for code review operations.

    Orchestrates analyzers and rules to perform comprehensive
    code reviews.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        config: Optional[ReviewConfig] = None,
        rule_registry: Optional[RuleRegistry] = None,
    ):
        """Initialize the review manager.

        Args:
            project_root: Root directory of the project
            config: Review configuration
            rule_registry: Custom rule registry
        """
        self.project_root = project_root or Path.cwd()
        self.config = config or ReviewConfig()
        self.rule_registry = rule_registry or get_rule_registry()

        # Initialize analyzers
        self._analyzers: dict[ReviewCategory, BaseAnalyzer] = {
            ReviewCategory.COMPLEXITY: ComplexityAnalyzer(),
            ReviewCategory.NAMING: NamingAnalyzer(),
            ReviewCategory.DOCUMENTATION: DocumentationAnalyzer(),
            ReviewCategory.SECURITY: SecurityAnalyzer(),
            ReviewCategory.BEST_PRACTICES: BestPracticesAnalyzer(),
        }

    def register_analyzer(self, analyzer: BaseAnalyzer) -> None:
        """Register a custom analyzer.

        Args:
            analyzer: Analyzer to register
        """
        self._analyzers[analyzer.category] = analyzer
        logger.debug(f"Registered analyzer for {analyzer.category.value}")

    def review_file(
        self,
        file_path: Path,
        ruleset_name: str = "standard",
        config: Optional[ReviewConfig] = None,
    ) -> FileReview:
        """Review a single file.

        Args:
            file_path: Path to the file
            ruleset_name: Name of ruleset to use
            config: Optional custom configuration

        Returns:
            FileReview with findings
        """
        start_time = time.time()
        config = config or self.config
        review = FileReview(file_path=file_path)

        try:
            source = file_path.read_text()
            review.lines_analyzed = len(source.split("\n"))
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            review.findings.append(
                ReviewFinding(
                    rule_id="internal-error",
                    message=f"Failed to read file: {e}",
                    severity=Severity.ERROR,
                    category=ReviewCategory.BEST_PRACTICES,
                    location=self._create_location(file_path, 1),
                )
            )
            return review

        # Get ruleset
        ruleset = self.rule_registry.get_ruleset(ruleset_name)
        if not ruleset:
            logger.warning(f"Ruleset '{ruleset_name}' not found, using 'standard'")
            ruleset = self.rule_registry.get_ruleset("standard")

        # Group rules by category
        rules_by_category: dict[ReviewCategory, list[ReviewRule]] = {}
        for rule in ruleset.get_enabled_rules():
            if rule.category not in rules_by_category:
                rules_by_category[rule.category] = []
            rules_by_category[rule.category].append(rule)

        # Run analyzers
        for category in config.enabled_categories:
            if category not in self._analyzers:
                continue

            analyzer = self._analyzers[category]
            rules = rules_by_category.get(category, [])

            if not rules:
                continue

            try:
                findings = analyzer.analyze(source, file_path, rules)
                review.findings.extend(findings)
            except Exception as e:
                logger.error(f"Analyzer {category.value} failed: {e}")

        # Filter by severity
        review.findings = [
            f
            for f in review.findings
            if self._severity_meets_threshold(f.severity, config.min_severity)
        ]

        # Limit findings
        if len(review.findings) > config.max_findings_per_file:
            review.findings = review.findings[: config.max_findings_per_file]

        review.duration_ms = (time.time() - start_time) * 1000
        return review

    def review_directory(
        self,
        directory: Path,
        ruleset_name: str = "standard",
        config: Optional[ReviewConfig] = None,
        recursive: bool = True,
    ) -> ReviewResult:
        """Review all files in a directory.

        Args:
            directory: Directory to review
            ruleset_name: Ruleset to use
            config: Optional configuration
            recursive: Whether to recurse into subdirectories

        Returns:
            ReviewResult with all findings
        """
        start_time = time.time()
        config = config or self.config
        result = ReviewResult(ruleset_name=ruleset_name)

        # Find files
        files = self._find_files(directory, config, recursive)
        result.total_files = len(files)

        # Review each file
        for file_path in files:
            file_review = self.review_file(file_path, ruleset_name, config)
            result.file_reviews.append(file_review)
            result.total_findings += len(file_review.findings)

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    def review_diff(
        self,
        base_ref: str = "HEAD",
        target_ref: str = "",
        ruleset_name: str = "standard",
    ) -> ReviewResult:
        """Review only changed files between git refs.

        Args:
            base_ref: Base git reference
            target_ref: Target git reference (empty for working tree)
            ruleset_name: Ruleset to use

        Returns:
            ReviewResult for changed files
        """
        import subprocess

        result = ReviewResult(ruleset_name=ruleset_name)

        try:
            # Get changed files
            if target_ref:
                cmd = ["git", "diff", "--name-only", base_ref, target_ref]
            else:
                cmd = ["git", "diff", "--name-only", base_ref]

            output = subprocess.check_output(
                cmd,
                cwd=self.project_root,
                text=True,
            )
            changed_files = [
                self.project_root / f.strip()
                for f in output.strip().split("\n")
                if f.strip() and f.endswith(".py")
            ]
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get git diff: {e}")
            return result

        result.total_files = len(changed_files)

        for file_path in changed_files:
            if file_path.exists():
                file_review = self.review_file(file_path, ruleset_name)
                result.file_reviews.append(file_review)
                result.total_findings += len(file_review.findings)

        return result

    def format_report(
        self,
        result: ReviewResult,
        format: str = "text",
    ) -> str:
        """Format review result as a report.

        Args:
            result: Review result
            format: Output format (text, markdown, json)

        Returns:
            Formatted report
        """
        if format == "json":
            return self._format_json(result)
        elif format == "markdown":
            return self._format_markdown(result)
        else:
            return self._format_text(result)

    def _format_text(self, result: ReviewResult) -> str:
        """Format as plain text."""
        lines = []
        lines.append("=" * 60)
        lines.append("CODE REVIEW REPORT")
        lines.append("=" * 60)
        lines.append(f"Ruleset: {result.ruleset_name}")
        lines.append(f"Files analyzed: {result.total_files}")
        lines.append(f"Total findings: {result.total_findings}")
        lines.append(f"Errors: {result.total_errors}")
        lines.append(f"Warnings: {result.total_warnings}")
        lines.append(f"Status: {'PASSED' if result.passed else 'FAILED'}")
        lines.append("-" * 60)

        for file_review in result.file_reviews:
            if not file_review.findings:
                continue

            lines.append(f"\n{file_review.file_path}")
            lines.append("-" * 40)

            for finding in file_review.findings:
                icon = self._severity_icon(finding.severity)
                lines.append(
                    f"  {icon} [{finding.rule_id}] Line {finding.location.start_line}: "
                    f"{finding.message}"
                )
                if finding.suggestion:
                    lines.append(f"     Suggestion: {finding.suggestion}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _format_markdown(self, result: ReviewResult) -> str:
        """Format as Markdown."""
        lines = []
        lines.append("# Code Review Report")
        lines.append("")
        lines.append(f"**Ruleset:** {result.ruleset_name}")
        lines.append(f"**Files analyzed:** {result.total_files}")
        lines.append(f"**Total findings:** {result.total_findings}")
        lines.append("")

        # Summary table
        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        lines.append(f"| :x: Errors | {result.total_errors} |")
        lines.append(f"| :warning: Warnings | {result.total_warnings} |")
        lines.append("")

        status = ":white_check_mark: PASSED" if result.passed else ":x: FAILED"
        lines.append(f"**Status:** {status}")
        lines.append("")

        # Findings by file
        for file_review in result.file_reviews:
            if not file_review.findings:
                continue

            lines.append(f"## `{file_review.file_path.name}`")
            lines.append("")

            for finding in file_review.findings:
                icon = self._severity_emoji(finding.severity)
                lines.append(
                    f"- {icon} **{finding.rule_id}** (line {finding.location.start_line}): "
                    f"{finding.message}"
                )
                if finding.suggestion:
                    lines.append(f"  - *Suggestion:* {finding.suggestion}")

            lines.append("")

        return "\n".join(lines)

    def _format_json(self, result: ReviewResult) -> str:
        """Format as JSON."""
        import json

        data = {
            "ruleset": result.ruleset_name,
            "total_files": result.total_files,
            "total_findings": result.total_findings,
            "errors": result.total_errors,
            "warnings": result.total_warnings,
            "passed": result.passed,
            "duration_ms": result.duration_ms,
            "files": [
                {
                    "path": str(fr.file_path),
                    "lines_analyzed": fr.lines_analyzed,
                    "findings": [f.to_dict() for f in fr.findings],
                }
                for fr in result.file_reviews
            ],
        }
        return json.dumps(data, indent=2)

    def _find_files(
        self,
        directory: Path,
        config: ReviewConfig,
        recursive: bool,
    ) -> list[Path]:
        """Find files to review."""
        files = []
        pattern = "**/*" if recursive else "*"

        for file_path in directory.glob(pattern):
            if not file_path.is_file():
                continue

            # Check include patterns
            included = any(fnmatch(str(file_path), p) for p in config.include_patterns)
            if not included:
                continue

            # Check exclude patterns
            excluded = any(fnmatch(str(file_path), p) for p in config.exclude_patterns)
            if excluded:
                continue

            files.append(file_path)

        return files

    def _severity_meets_threshold(
        self,
        severity: Severity,
        threshold: Severity,
    ) -> bool:
        """Check if severity meets threshold."""
        order = [Severity.HINT, Severity.INFO, Severity.WARNING, Severity.ERROR]
        return order.index(severity) >= order.index(threshold)

    def _severity_icon(self, severity: Severity) -> str:
        """Get text icon for severity."""
        return {
            Severity.ERROR: "[E]",
            Severity.WARNING: "[W]",
            Severity.INFO: "[I]",
            Severity.HINT: "[H]",
        }.get(severity, "[?]")

    def _severity_emoji(self, severity: Severity) -> str:
        """Get emoji for severity."""
        return {
            Severity.ERROR: ":x:",
            Severity.WARNING: ":warning:",
            Severity.INFO: ":information_source:",
            Severity.HINT: ":bulb:",
        }.get(severity, ":grey_question:")

    def _create_location(self, file_path: Path, line: int):
        """Create a source location."""
        from victor_coding.review.protocol import SourceLocation

        return SourceLocation(file_path=file_path, start_line=line)


# Global manager singleton
_review_manager: Optional[ReviewManager] = None


def get_review_manager(
    project_root: Optional[Path] = None,
    config: Optional[ReviewConfig] = None,
) -> ReviewManager:
    """Get the global review manager.

    Args:
        project_root: Project root directory
        config: Review configuration

    Returns:
        ReviewManager instance
    """
    global _review_manager
    if _review_manager is None or (project_root and _review_manager.project_root != project_root):
        _review_manager = ReviewManager(project_root=project_root, config=config)
    return _review_manager


def reset_review_manager() -> None:
    """Reset the global manager."""
    global _review_manager
    _review_manager = None
