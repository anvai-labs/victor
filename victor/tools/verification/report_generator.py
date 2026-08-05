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

"""Verification report generator for codebase analysis results.

Generates structured reports with semantic validation, false positive
filtering, documentation cross-reference, and severity weighting.
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from victor.tools.verification import (
    ClaimIssue,
    ClaimVerifier,
    DocumentationCrossReference,
    EnhancedClaimResult,
    FalsePositiveDetector,
    SeverityLevel,
    SeverityWeighting,
    TemporalContextAnalyzer,
    TemporalNature,
    VerificationContext,
)

console = Console()


class ReportFormat(str, Enum):
    """Report output formats."""

    JSON = "json"
    MARKDOWN = "markdown"
    CONSOLE = "console"
    HTML = "html"


@dataclass
class ReportSummary:
    """Summary statistics for a verification report."""

    total_issues: int = 0
    verified_issues: int = 0
    false_positives: int = 0
    genuine_issues: int = 0

    # Severity breakdown
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    info: int = 0

    # Temporal breakdown
    temporary: int = 0
    permanent: int = 0
    unknown_temporal: int = 0

    # Documentation references
    documented_issues: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_issues": self.total_issues,
            "verified_issues": self.verified_issues,
            "false_positives": self.false_positives,
            "genuine_issues": self.genuine_issues,
            "severity_breakdown": {
                "critical": self.critical,
                "high": self.high,
                "medium": self.medium,
                "low": self.low,
                "info": self.info,
            },
            "temporal_breakdown": {
                "temporary": self.temporary,
                "permanent": self.permanent,
                "unknown": self.unknown_temporal,
            },
            "documented_issues": self.documented_issues,
        }


@dataclass
class VerificationReport:
    """Complete verification report for codebase analysis.

    Attributes:
        project_root: Root directory of the analyzed project
        timestamp: Report generation timestamp
        summary: Report summary statistics
        verified_issues: List of verified issues with full context
        false_positives: List of issues flagged as false positives
        recommendations: Generated recommendations
    """

    project_root: str
    timestamp: str = ""
    summary: ReportSummary = field(default_factory=ReportSummary)
    verified_issues: List[Dict[str, Any]] = field(default_factory=list)
    false_positives: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "project_root": self.project_root,
            "timestamp": self.timestamp,
            "summary": self.summary.to_dict(),
            "verified_issues": self.verified_issues,
            "false_positives": self.false_positives,
            "recommendations": self.recommendations,
        }


class VerificationReportGenerator:
    """Generates verification reports for codebase analysis.

    Usage:
        generator = VerificationReportGenerator(project_root=Path("."))
        report = await generator.generate_report(issues)
        generator.save_report(report, "verification_report.json")
    """

    def __init__(
        self,
        project_root: Path,
        enable_fp_detection: bool = True,
        enable_doc_crossref: bool = True,
        enable_temporal_analysis: bool = True,
        enable_severity_weighting: bool = True,
    ):
        """Initialize report generator.

        Args:
            project_root: Root directory of the project
            enable_fp_detection: Enable false positive detection
            enable_doc_crossref: Enable documentation cross-reference
            enable_temporal_analysis: Enable temporal analysis
            enable_severity_weighting: Enable severity weighting
        """
        self._project_root = Path(project_root)
        self._enable_fp_detection = enable_fp_detection
        self._enable_doc_crossref = enable_doc_crossref
        self._enable_temporal_analysis = enable_temporal_analysis
        self._enable_severity_weighting = enable_severity_weighting

        # Initialize components
        self._verifier = ClaimVerifier(project_root=self._project_root)
        self._fp_detector = FalsePositiveDetector() if enable_fp_detection else None
        self._crossref = (
            DocumentationCrossReference(project_root=self._project_root)
            if enable_doc_crossref
            else None
        )
        self._temporal = (
            TemporalContextAnalyzer(project_root=self._project_root)
            if enable_temporal_analysis
            else None
        )
        self._weighting = SeverityWeighting() if enable_severity_weighting else None

    async def generate_report(
        self,
        issues: List[Dict[str, Any] | ClaimIssue],
    ) -> VerificationReport:
        """Generate a comprehensive verification report.

        Args:
            issues: List of issues to verify

        Returns:
            VerificationReport with all analysis results
        """
        report = VerificationReport(project_root=str(self._project_root))
        summary = report.summary

        for issue_input in issues:
            # Normalize input
            if isinstance(issue_input, dict):
                issue = ClaimIssue(**issue_input)
            else:
                issue = issue_input

            summary.total_issues += 1

            # Core verification
            result = await self._verifier.verify_claim(issue)

            # False positive detection
            is_fp = False
            fp_confidence = 0.0
            fp_reason = ""
            if self._fp_detector:
                is_fp, fp_reason, fp_confidence = self._fp_detector.is_likely_false_positive(issue)

            # Skip false positives from main issue list
            if is_fp and fp_confidence > 0.7:
                summary.false_positives += 1
                report.false_positives.append(
                    {
                        **(issue_input if isinstance(issue_input, dict) else issue.model_dump()),
                        "fp_reason": fp_reason,
                        "fp_confidence": fp_confidence,
                    }
                )
                continue

            summary.genuine_issues += 1

            # Severity weighting
            severity = SeverityLevel.INFO
            severity_score = 0.0
            if self._weighting:
                severity_score, severity = self._weighting.score_and_classify(issue.model_dump())

            # Update severity counts
            if severity == SeverityLevel.CRITICAL:
                summary.critical += 1
            elif severity == SeverityLevel.HIGH:
                summary.high += 1
            elif severity == SeverityLevel.MEDIUM:
                summary.medium += 1
            elif severity == SeverityLevel.LOW:
                summary.low += 1
            else:
                summary.info += 1

            # Temporal analysis
            temporal_nature = TemporalNature.UNKNOWN
            if self._temporal and issue.file_path:
                temporal_ctx = self._temporal.analyze_issue_temporal_context(issue)
                temporal_nature = temporal_ctx.get("temporal_nature", TemporalNature.UNKNOWN)

                if temporal_nature == TemporalNature.TEMPORARY:
                    summary.temporary += 1
                elif temporal_nature == TemporalNature.PERMANENT:
                    summary.permanent += 1
                else:
                    summary.unknown_temporal += 1

            # Documentation cross-reference
            doc_refs = []
            if self._crossref:
                doc_refs = self._crossref.get_doc_references(issue.model_dump())
                if doc_refs:
                    summary.documented_issues += 1

            # Build issue result
            issue_result = {
                **(issue_input if isinstance(issue_input, dict) else issue.model_dump()),
                "verified": result.is_grounded,
                "confidence": result.confidence,
                "evidence_count": len(result.evidence.get("sources", [])),
                "severity": severity.value,
                "severity_score": severity_score,
                "temporal_nature": temporal_nature.value if temporal_nature else None,
                "doc_references": len(doc_refs),
                "reason": result.reason,
            }

            report.verified_issues.append(issue_result)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(summary)

        return report

    def _generate_recommendations(self, summary: ReportSummary) -> List[str]:
        """Generate recommendations based on report summary.

        Args:
            summary: Report summary statistics

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Priority recommendations based on severity
        if summary.critical > 0:
            recommendations.append(f"Address {summary.critical} critical issue(s) immediately")

        if summary.high > 5:
            recommendations.append(f"Consider prioritizing the {summary.high} high-severity issues")

        # False positive recommendations
        fp_rate = summary.false_positives / summary.total_issues if summary.total_issues > 0 else 0
        if fp_rate > 0.3:
            recommendations.append(
                f"High false positive rate ({fp_rate:.0%}) - review analysis filters"
            )

        # Temporal recommendations
        if summary.temporary > summary.permanent:
            recommendations.append(
                f"{summary.temporary} issues appear temporary - may resolve naturally"
            )

        # Documentation recommendations
        if summary.documented_issues < summary.genuine_issues * 0.5:
            recommendations.append(
                "Consider tracking more issues in TECHNICAL_DEBT.adoc for better traceability"
            )

        return recommendations

    def save_report(
        self,
        report: VerificationReport,
        output_path: Path,
        format: ReportFormat = ReportFormat.JSON,
    ) -> None:
        """Save report to file.

        Args:
            report: Report to save
            output_path: Output file path
            format: Output format
        """
        output_path = Path(output_path)

        if format == ReportFormat.JSON:
            output_path.write_text(json_dumps(report.to_dict(), indent=2))

        elif format == ReportFormat.MARKDOWN:
            md = self._format_markdown(report)
            output_path.write_text(md)

        elif format == ReportFormat.CONSOLE:
            self._print_console(report)

    def _format_markdown(self, report: VerificationReport) -> str:
        """Format report as Markdown.

        Args:
            report: Report to format

        Returns:
            Markdown string
        """
        lines = [
            "# Codebase Verification Report",
            "",
            f"**Project:** `{report.project_root}`",
            f"**Generated:** {report.timestamp}",
            "",
            "## Summary",
            "",
            f"- **Total Issues:** {report.summary.total_issues}",
            f"- **Verified:** {report.summary.verified_issues}",
            f"- **Genuine Issues:** {report.summary.genuine_issues}",
            f"- **False Positives Filtered:** {report.summary.false_positives}",
            "",
            "### Severity Distribution",
            "",
            "| Critical | High | Medium | Low | Info |",
            "|----------|------|--------|-----|-----|",
            f"| {report.summary.critical} | {report.summary.high} | {report.summary.medium} | {report.summary.low} | {report.summary.info} |",
            "",
            "### Temporal Distribution",
            "",
            f"- **Temporary:** {report.summary.temporary}",
            f"- **Permanent:** {report.summary.permanent}",
            f"- **Unknown:** {report.summary.unknown_temporal}",
            "",
        ]

        if report.recommendations:
            lines.extend(
                [
                    "## Recommendations",
                    "",
                ]
            )
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        if report.verified_issues:
            lines.extend(
                [
                    "## Verified Issues",
                    "",
                    "| Issue | File | Severity | Confidence | Temporal |",
                    "|-------|------|----------|------------|----------|",
                ]
            )
            for issue in report.verified_issues[:20]:  # Limit to 20
                issue_type = issue.get("issue_type", "unknown")
                # Truncate issue type if too long, add ellipsis
                if len(issue_type) > 18:
                    issue_type = issue_type[:15] + "..."
                file_path = issue.get("file_path", "N/A")
                # Show only filename for brevity
                if "/" in file_path:
                    file_path = file_path.split("/")[-1]
                if len(file_path) > 20:
                    file_path = file_path[:17] + "..."
                severity = issue.get("severity", "info")
                confidence = f"{issue.get('confidence', 0):.0%}"
                temporal = issue.get("temporal_nature", "unknown")
                lines.append(
                    f"| {issue_type} | {file_path} | {severity} | {confidence} | {temporal} |"
                )

            if len(report.verified_issues) > 20:
                lines.append(
                    f"| ... | ... | ... | ... | ... | ({len(report.verified_issues) - 20} more) |"
                )
            lines.append("")

        return "\n".join(lines)

    def _print_console(self, report: VerificationReport) -> None:
        """Print report to console with Rich formatting.

        Args:
            report: Report to print
        """
        console.print("\n[bold]Codebase Verification Report[/bold]")
        console.print(f"Project: {report.project_root}")
        console.print(f"Generated: {report.timestamp}")

        # Summary table
        console.print("\n[bold]Summary[/bold]")
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value")
        table.add_row("Total Issues", str(report.summary.total_issues))
        table.add_row("Genuine Issues", f"[green]{report.summary.genuine_issues}[/green]")
        table.add_row("False Positives", f"[yellow]{report.summary.false_positives}[/yellow]")
        console.print(table)

        # Severity table
        console.print("\n[bold]Severity Distribution[/bold]")
        sev_table = Table()
        sev_table.add_column("Severity")
        sev_table.add_column("Count", justify="right")
        sev_table.add_row("Critical", f"[red]{report.summary.critical}[/red]")
        sev_table.add_row("High", f"[orange3]{report.summary.high}[/orange3]")
        sev_table.add_row("Medium", f"[yellow]{report.summary.medium}[/yellow]")
        sev_table.add_row("Low", f"[blue]{report.summary.low}[/blue]")
        sev_table.add_row("Info", f"[dim]{report.summary.info}[/dim]")
        console.print(sev_table)

        # Recommendations
        if report.recommendations:
            console.print("\n[bold]Recommendations[/bold]")
            for i, rec in enumerate(report.recommendations, 1):
                console.print(f"  {i}. {rec}")
