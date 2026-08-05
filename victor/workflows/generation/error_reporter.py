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

"""Error aggregation and reporting for workflow validation.

This module provides comprehensive error reporting with multiple output formats
for different consumers (humans, LLMs, logs, etc.).

Design Principles (SOLID):
    - SRP: Focused solely on error formatting and aggregation
    - OCP: Extensible via new report formats
    - ISP: Separate methods for each report type

Key Features:
    - Human-readable reports with actionable suggestions
    - LLM-friendly reports for refinement
    - JSON/Markdown export for programmatic use
    - Error grouping and prioritization
    - Compact summaries

Example:
    from victor.workflows.generation import ErrorReporter, WorkflowValidator

    validator = WorkflowValidator()
    result = validator.validate(workflow)

    reporter = ErrorReporter()

    # Human-readable report
    print(reporter.human_report(result))

    # LLM-friendly report
    llm_prompt = reporter.llm_report(result)
"""

import json
from typing import Dict, List, Optional

from victor.workflows.generation.types import (
    ErrorCategory,
    ErrorSeverity,
    WorkflowValidationError,
    WorkflowGenerationValidationResult,
)


class ErrorReport:
    """Aggregated error report with grouping and prioritization.

    Attributes:
        total_errors: Total number of errors
        critical_count: Number of critical errors
        error_count: Number of error-level issues
        warning_count: Number of warnings
        info_count: Number of info messages
        errors_by_category: Errors grouped by category
        errors_by_severity: Errors grouped by severity
        errors_by_node: Errors grouped by node ID

    Example:
        report = ErrorReporter().aggregate_errors(result)
        print(f"Total: {report.total_errors}")
        print(f"Critical: {report.critical_count}")
    """

    def __init__(
        self,
        result: WorkflowGenerationValidationResult,
        errors_by_category: Dict[str, List[WorkflowValidationError]],
        errors_by_severity: Dict[str, List[WorkflowValidationError]],
        errors_by_node: Dict[str, List[WorkflowValidationError]],
    ):
        self.result = result
        self.errors_by_category = errors_by_category
        self.errors_by_severity = errors_by_severity
        self.errors_by_node = errors_by_node

        self.total_errors = len(result.all_errors)
        self.critical_count = len(errors_by_severity.get("critical", []))
        self.error_count = len(errors_by_severity.get("error", []))
        self.warning_count = len(errors_by_severity.get("warning", []))
        self.info_count = len(errors_by_severity.get("info", []))


class ErrorReporter:
    """Formats validation results for different consumers.

    Provides multiple report formats:
    - Human-readable: Detailed with emojis and suggestions
    - LLM-friendly: Structured for refinement prompts
    - JSON: Machine-readable for export
    - Markdown: Documentation-friendly
    - Compact: Single-line summaries

    Example:
        reporter = ErrorReporter()

        # For terminal output
        print(reporter.human_report(result))

        # For LLM refinement
        prompt = reporter.llm_report(result)

        # For export
        json_str = reporter.json_report(result)
    """

    def aggregate_errors(self, result: WorkflowGenerationValidationResult) -> ErrorReport:
        """Aggregate errors by category, severity, and node.

        Args:
            result: Validation result

        Returns:
            ErrorReport with grouped errors
        """
        # Group by category
        errors_by_category: Dict[str, List[WorkflowValidationError]] = {
            "schema": result.schema_errors,
            "structure": result.structure_errors,
            "semantic": result.semantic_errors,
            "security": result.security_errors,
        }

        # Group by severity
        errors_by_severity: Dict[str, List[WorkflowValidationError]] = {
            "critical": [],
            "error": [],
            "warning": [],
            "info": [],
        }
        for error in result.all_errors:
            severity = error.severity.value
            errors_by_severity[severity].append(error)

        # Group by node
        errors_by_node = result.group_by_node()

        return ErrorReport(
            result=result,
            errors_by_category=errors_by_category,
            errors_by_severity=errors_by_severity,
            errors_by_node=errors_by_node,
        )

    def human_report(
        self,
        result: WorkflowGenerationValidationResult,
        show_suggestions: bool = True,
        show_context: bool = False,
    ) -> str:
        """Generate human-readable error report.

        Args:
            result: Validation result
            show_suggestions: Include fix suggestions
            show_context: Include error context

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("WORKFLOW VALIDATION REPORT")
        lines.append("=" * 80)

        # Status line
        status = "VALID ✓" if result.is_valid else "INVALID ✗"
        lines.append(f"Status: {status}")

        if result.workflow_name:
            lines.append(f"Workflow: {result.workflow_name}")

        lines.append("")

        # If valid, just show summary
        if result.is_valid:
            warnings = [e for e in result.all_errors if e.severity == ErrorSeverity.WARNING]
            if warnings:
                lines.append(f"Validation passed with {len(warnings)} warnings:")
                lines.append("-" * 80)
                for error in warnings:
                    lines.append(f"  ⚠️  {error.location}: {error.message}")
                    if show_suggestions and error.suggestion:
                        lines.append(f"     💡 {error.suggestion}")
            else:
                lines.append("✓ All checks passed!")
            lines.append("=" * 80)
            return "\n".join(lines)

        # Show errors by priority
        report = self.aggregate_errors(result)

        # Critical errors first
        if report.critical_count > 0:
            lines.append(f"CRITICAL ERRORS ({report.critical_count}):")
            lines.append("-" * 80)
            for error in report.errors_by_severity["critical"]:
                lines.append(f"  🚨 {error.location}: {error.message}")
                if show_suggestions and error.suggestion:
                    lines.append(f"     💡 Suggestion: {error.suggestion}")
                if show_context and error.context:
                    lines.append(f"     Context: {json.dumps(error.context, indent=6)}")
            lines.append("")

        # Error-level issues
        if report.error_count > 0:
            lines.append(f"ERRORS ({report.error_count}):")
            lines.append("-" * 80)
            for error in report.errors_by_severity["error"]:
                lines.append(f"  ❌ {error.location}: {error.message}")
                if show_suggestions and error.suggestion:
                    lines.append(f"     💡 {error.suggestion}")
                if show_context and error.context:
                    lines.append(f"     Context: {json.dumps(error.context, indent=6)}")
            lines.append("")

        # Warnings
        if report.warning_count > 0:
            lines.append(f"WARNINGS ({report.warning_count}):")
            lines.append("-" * 80)
            for error in report.errors_by_severity["warning"]:
                lines.append(f"  ⚠️  {error.location}: {error.message}")
                if show_suggestions and error.suggestion:
                    lines.append(f"     💡 {error.suggestion}")
            lines.append("")

        # Info messages (rarely shown)
        if report.info_count > 0 and show_context:
            lines.append(f"INFO ({report.info_count}):")
            lines.append("-" * 80)
            for error in report.errors_by_severity["info"]:
                lines.append(f"  ℹ️  {error.location}: {error.message}")
            lines.append("")

        # Summary
        lines.append("-" * 80)
        lines.append(f"Total Issues: {report.total_errors}")
        lines.append(f"  Critical: {report.critical_count}")
        lines.append(f"  Errors: {report.error_count}")
        lines.append(f"  Warnings: {report.warning_count}")
        lines.append("=" * 80)

        return "\n".join(lines)

    def llm_report(
        self,
        result: WorkflowGenerationValidationResult,
        include_fixes: bool = True,
        include_context: bool = False,
        max_errors_per_category: int = 20,
    ) -> str:
        """Generate LLM-friendly error report for refinement.

        Optimized for LLM consumption:
        - Clear structure with sections
        - Concise error descriptions
        - Actionable fix suggestions
        - No emojis or special formatting

        Args:
            result: Validation result
            include_fixes: Include fix suggestions
            include_context: Include error context
            max_errors_per_category: Limit errors per category

        Returns:
            Formatted report string for LLM
        """
        # If valid, return simple message
        if result.is_valid:
            warnings = [e for e in result.all_errors if e.severity == ErrorSeverity.WARNING]
            if warnings:
                return f"Validation passed with {len(warnings)} warnings. No changes required."
            return "Validation passed. No changes required."

        lines = []
        lines.append("VALIDATION FAILED - FIXES REQUIRED:")
        lines.append("")

        # Group errors by category
        report = self.aggregate_errors(result)

        # Schema errors
        if report.errors_by_category["schema"]:
            lines.append(f"SCHEMA ERRORS ({len(report.errors_by_category['schema'])}):")
            for error in report.errors_by_category["schema"][:max_errors_per_category]:
                lines.append(f"  - Location: {error.location}")
                lines.append(f"    Error: {error.message}")
                if include_fixes and error.suggestion:
                    lines.append(f"    Fix: {error.suggestion}")
                if include_context and error.context:
                    lines.append(f"    Context: {json.dumps(error.context)}")
            if len(report.errors_by_category["schema"]) > max_errors_per_category:
                lines.append(
                    f"  ... and {len(report.errors_by_category['schema']) - max_errors_per_category} more schema errors"
                )
            lines.append("")

        # Structure errors
        if report.errors_by_category["structure"]:
            lines.append(f"STRUCTURE ERRORS ({len(report.errors_by_category['structure'])}):")
            for error in report.errors_by_category["structure"][:max_errors_per_category]:
                lines.append(f"  - Location: {error.location}")
                lines.append(f"    Error: {error.message}")
                if include_fixes and error.suggestion:
                    lines.append(f"    Fix: {error.suggestion}")
                if include_context and error.context:
                    lines.append(f"    Context: {json.dumps(error.context)}")
            if len(report.errors_by_category["structure"]) > max_errors_per_category:
                lines.append(
                    f"  ... and {len(report.errors_by_category['structure']) - max_errors_per_category} more structure errors"
                )
            lines.append("")

        # Semantic errors
        if report.errors_by_category["semantic"]:
            lines.append(f"SEMANTIC ERRORS ({len(report.errors_by_category['semantic'])}):")
            for error in report.errors_by_category["semantic"][:max_errors_per_category]:
                lines.append(f"  - Location: {error.location}")
                lines.append(f"    Error: {error.message}")
                if include_fixes and error.suggestion:
                    lines.append(f"    Fix: {error.suggestion}")
                if include_context and error.context:
                    lines.append(f"    Context: {json.dumps(error.context)}")
            if len(report.errors_by_category["semantic"]) > max_errors_per_category:
                lines.append(
                    f"  ... and {len(report.errors_by_category['semantic']) - max_errors_per_category} more semantic errors"
                )
            lines.append("")

        # Security errors
        if report.errors_by_category["security"]:
            lines.append(f"SECURITY ERRORS ({len(report.errors_by_category['security'])}):")
            for error in report.errors_by_category["security"][:max_errors_per_category]:
                lines.append(f"  - Location: {error.location}")
                lines.append(f"    Error: {error.message}")
                if include_fixes and error.suggestion:
                    lines.append(f"    Fix: {error.suggestion}")
                if include_context and error.context:
                    lines.append(f"    Context: {json.dumps(error.context)}")
            if len(report.errors_by_category["security"]) > max_errors_per_category:
                lines.append(
                    f"  ... and {len(report.errors_by_category['security']) - max_errors_per_category} more security errors"
                )
            lines.append("")

        # Summary
        lines.append("SUMMARY:")
        lines.append(f"  Total errors: {report.total_errors}")
        lines.append(f"  Critical: {report.critical_count}")
        lines.append(f"  Must fix: {report.critical_count + report.error_count}")
        lines.append("")
        lines.append("CRITICAL: You must fix all critical and error-level issues.")
        lines.append("Warnings should be addressed if possible.")

        return "\n".join(lines)

    def json_report(self, result: WorkflowGenerationValidationResult, indent: int = 2) -> str:
        """Generate JSON report for programmatic consumption.

        Args:
            result: Validation result
            indent: JSON indentation

        Returns:
            JSON string
        """
        return json.dumps(result.to_dict(), indent=indent)

    def markdown_report(
        self,
        result: WorkflowGenerationValidationResult,
        include_suggestions: bool = True,
    ) -> str:
        """Generate Markdown report for documentation.

        Args:
            result: Validation result
            include_suggestions: Include fix suggestions

        Returns:
            Markdown-formatted string
        """
        lines = []

        # Title
        status = "**VALID**" if result.is_valid else "**INVALID**"
        lines.append(f"# Workflow Validation Report: {status}")

        if result.workflow_name:
            lines.append(f"\n**Workflow:** {result.workflow_name}")

        lines.append("")

        # Summary
        report = self.aggregate_errors(result)
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Total Issues: {report.total_errors}")
        lines.append(f"- Critical: {report.critical_count}")
        lines.append(f"- Errors: {report.error_count}")
        lines.append(f"- Warnings: {report.warning_count}")
        lines.append("")

        # If valid
        if result.is_valid:
            if report.warning_count > 0:
                lines.append("## Warnings")
                lines.append("")
                for error in report.errors_by_severity["warning"]:
                    lines.append(f"- **{error.location}**: {error.message}")
                    if include_suggestions and error.suggestion:
                        lines.append(f"  - *Suggestion*: {error.suggestion}")
            lines.append("")
            lines.append("*Validation passed successfully.*")
            return "\n".join(lines)

        # Critical errors
        if report.critical_count > 0:
            lines.append("## Critical Errors")
            lines.append("")
            for error in report.errors_by_severity["critical"]:
                lines.append(f"### {error.location}")
                lines.append(f"{error.message}")
                if include_suggestions and error.suggestion:
                    lines.append(f"\n**Suggestion:** {error.suggestion}")
                lines.append("")

        # Errors
        if report.error_count > 0:
            lines.append("## Errors")
            lines.append("")
            for error in report.errors_by_severity["error"]:
                lines.append(f"### {error.location}")
                lines.append(f"{error.message}")
                if include_suggestions and error.suggestion:
                    lines.append(f"\n**Suggestion:** {error.suggestion}")
                lines.append("")

        # Warnings
        if report.warning_count > 0:
            lines.append("## Warnings")
            lines.append("")
            for error in report.errors_by_severity["warning"]:
                lines.append(f"### {error.location}")
                lines.append(f"{error.message}")
                if include_suggestions and error.suggestion:
                    lines.append(f"\n**Suggestion:** {error.suggestion}")
                lines.append("")

        return "\n".join(lines)

    def compact_report(self, result: WorkflowGenerationValidationResult) -> str:
        """Generate compact one-line summary.

        Args:
            result: Validation result

        Returns:
            Single-line summary string
        """
        if result.is_valid:
            return "✓ Valid"

        report = self.aggregate_errors(result)
        return (
            f"✗ {report.critical_count} critical, "
            f"{report.error_count} errors, "
            f"{report.warning_count} warnings"
        )

    def node_report(self, result: WorkflowGenerationValidationResult, node_id: str) -> str:
        """Generate report for a specific node.

        Useful for focused debugging of problematic nodes.

        Args:
            result: Validation result
            node_id: Node ID to report on

        Returns:
            Formatted report string
        """
        report = self.aggregate_errors(result)
        node_errors = report.errors_by_node.get(node_id, [])

        if not node_errors:
            return f"Node '{node_id}': No errors"

        lines = []
        lines.append(f"Node '{node_id}' Errors:")
        lines.append("-" * 40)

        for error in node_errors:
            severity_emoji = {
                "critical": "🚨",
                "error": "❌",
                "warning": "⚠️",
                "info": "ℹ️",
            }.get(error.severity.value, "")

            lines.append(f"{severity_emoji} [{error.severity.value.upper()}] {error.message}")
            if error.suggestion:
                lines.append(f"   💡 {error.suggestion}")

        return "\n".join(lines)

    def category_report(
        self, result: WorkflowGenerationValidationResult, category: ErrorCategory
    ) -> str:
        """Generate report for a specific error category.

        Useful for focused debugging by validation layer.

        Args:
            result: Validation result
            category: Error category to report

        Returns:
            Formatted report string
        """
        category_map = {
            ErrorCategory.SCHEMA: result.schema_errors,
            ErrorCategory.STRUCTURE: result.structure_errors,
            ErrorCategory.SEMANTIC: result.semantic_errors,
            ErrorCategory.SECURITY: result.security_errors,
        }

        errors = category_map.get(category, [])

        if not errors:
            return f"{category.value.capitalize()} validation: No errors"

        lines = []
        lines.append(f"{category.value.capitalize()} Errors ({len(errors)}):")
        lines.append("-" * 40)

        for error in errors:
            lines.append(f"  {error.location}: {error.message}")
            if error.suggestion:
                lines.append(f"    💡 {error.suggestion}")

        return "\n".join(lines)

    def prioritize_errors(
        self, errors: List[WorkflowValidationError]
    ) -> List[WorkflowValidationError]:
        """Sort errors by severity and importance.

        Priority order:
        1. Critical errors
        2. Schema errors (foundational)
        3. Structure errors (affects execution)
        4. Semantic errors (node-specific)
        5. Security errors (safety)
        6. Warnings

        Args:
            errors: List of validation errors

        Returns:
            Sorted list of errors
        """
        # Define priority order for categories
        category_priority = {
            ErrorCategory.SCHEMA: 1,
            ErrorCategory.STRUCTURE: 2,
            ErrorCategory.SEMANTIC: 3,
            ErrorCategory.SECURITY: 4,
        }

        # Sort by severity first, then category
        return sorted(
            errors,
            key=lambda e: (
                # Primary sort: severity (critical > error > warning > info)
                {
                    ErrorSeverity.CRITICAL: 0,
                    ErrorSeverity.ERROR: 1,
                    ErrorSeverity.WARNING: 2,
                    ErrorSeverity.INFO: 3,
                }.get(e.severity, 4),
                # Secondary sort: category priority
                category_priority.get(e.category, 5),
                # Tertiary sort: location (alphabetical)
                e.location,
            ),
        )


__all__ = [
    "ErrorReporter",
    "ErrorReport",
]
