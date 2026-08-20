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

"""Pipeline Analytics Tool - CI/CD and coverage analysis for Victor tools.

This tool provides integration with Victor's tool system for pipeline
analysis, coverage tracking, and optimization recommendations.
"""

import logging
from pathlib import Path
from typing import Any

from victor.observability.pipeline import (
    CoverageMetrics,
    PipelineAnalysisResult,
    PipelineManager,
    PipelinePlatform,
)
from victor.tools.base import (
    AccessMode,
    BaseTool,
    CostTier,
    DangerLevel,
    Priority,
    ToolMetadata,
    ToolResult,
)
from victor.tools.tool_names import ToolNames

logger = logging.getLogger(__name__)

# Lazy-loaded presentation adapter for icon rendering
_presentation = None


def _get_icon(name: str) -> str:
    """Get icon from presentation adapter."""
    global _presentation
    if _presentation is None:
        from victor.agent.presentation import create_presentation_adapter

        _presentation = create_presentation_adapter()
    return _presentation.icon(name, with_color=False)


class PipelineAnalyzerTool(BaseTool):
    """Tool for analyzing CI/CD pipelines and coverage."""

    name = ToolNames.PIPELINE
    description = """Analyze CI/CD pipelines (GitHub Actions, GitLab) and coverage.

    Actions: analyze, coverage, compare_coverage, summary, detect.
    Coverage formats: Cobertura, LCOV, JaCoCo."""

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "analyze",
                    "coverage",
                    "compare_coverage",
                    "summary",
                    "detect",
                ],
                "description": "Action to perform",
            },
            "platform": {
                "type": "string",
                "enum": ["github_actions", "gitlab_ci", "all"],
                "description": "CI/CD platform to analyze (default: all)",
            },
            "coverage_format": {
                "type": "string",
                "enum": ["cobertura", "lcov", "jacoco", "auto"],
                "description": "Coverage report format (default: auto-detect)",
            },
            "baseline_path": {
                "type": "string",
                "description": "Path to baseline coverage for comparison",
            },
            "include_coverage": {
                "type": "boolean",
                "description": "Include coverage analysis in pipeline analysis (default: true)",
            },
        },
        "required": ["action"],
    }

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.LOW

    @property
    def priority(self) -> Priority:
        """Tool priority for selection availability."""
        return Priority.MEDIUM  # Task-specific pipeline analysis

    @property
    def access_mode(self) -> AccessMode:
        """Tool access mode for approval tracking."""
        return AccessMode.READONLY  # Only reads pipeline configs and coverage

    @property
    def danger_level(self) -> DangerLevel:
        """Danger level for warning/confirmation logic."""
        return DangerLevel.SAFE  # No side effects

    @property
    def metadata(self) -> ToolMetadata:
        """Inline semantic metadata for dynamic tool selection."""
        return ToolMetadata(
            category="pipeline",
            keywords=[
                "pipeline",
                "ci/cd",
                "coverage",
                "github actions",
                "gitlab ci",
                "cobertura",
                "lcov",
                "jacoco",
                "test coverage",
                "pipeline health",
                "build analysis",
                "workflow analysis",
                "coverage trend",
            ],
            use_cases=[
                "analyzing CI/CD pipelines",
                "viewing code coverage metrics",
                "pipeline optimization",
                "coverage comparison",
                "GitHub Actions analysis",
                "GitLab CI analysis",
                "detecting pipeline issues",
            ],
            examples=[
                "analyzing GitHub Actions workflow",
                "checking code coverage percentage",
                "comparing coverage against baseline",
                "finding pipeline optimization opportunities",
            ],
            priority_hints=[
                "Use for CI/CD pipeline analysis and optimization",
                "Supports multiple CI platforms and coverage formats",
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute pipeline analysis action."""
        action = kwargs.get("action", "summary")
        platform_str = kwargs.get("platform", "all")
        coverage_format = kwargs.get("coverage_format", "auto")
        baseline_path = kwargs.get("baseline_path")
        include_coverage = kwargs.get("include_coverage", True)

        try:
            manager = PipelineManager()

            if action == "detect":
                platforms = await manager.detect_platforms()
                return ToolResult(
                    success=True,
                    output=self._format_platforms(platforms),
                    metadata={"platforms": [p.value for p in platforms]},
                )

            elif action == "summary":
                summary = await manager.get_pipeline_summary()
                return ToolResult(
                    success=True,
                    output=self._format_summary(summary),
                    metadata=summary,
                )

            elif action == "analyze":
                platforms = None
                if platform_str != "all":
                    platforms = [PipelinePlatform(platform_str)]

                result = await manager.analyze_pipelines(
                    platforms=platforms,
                    include_coverage=include_coverage,
                )
                return ToolResult(
                    success=True,
                    output=self._format_analysis(result),
                    metadata=result.to_dict(),
                )

            elif action == "coverage":
                format_hint = coverage_format if coverage_format != "auto" else None
                coverage = await manager.get_coverage(format_hint=format_hint)
                if not coverage:
                    return ToolResult(
                        success=False,
                        output="No coverage reports found in project",
                        error="No coverage data",
                    )
                return ToolResult(
                    success=True,
                    output=self._format_coverage(coverage),
                    metadata=coverage.to_dict(),
                )

            elif action == "compare_coverage":
                comparison = await manager.compare_coverage(
                    baseline_path=Path(baseline_path) if baseline_path else None
                )
                if "error" in comparison:
                    return ToolResult(
                        success=False,
                        output=comparison["error"],
                        error=comparison["error"],
                    )
                return ToolResult(
                    success=True,
                    output=self._format_comparison(comparison),
                    metadata=comparison,
                )

            else:
                return ToolResult(
                    success=False,
                    output=f"Unknown action: {action}",
                    error="Invalid action",
                )

        except Exception as e:
            logger.exception(f"Pipeline analysis failed: {e}")
            return ToolResult(
                success=False,
                output=f"Pipeline analysis failed: {e}",
                error=str(e),
            )

    def _format_platforms(self, platforms: list[PipelinePlatform]) -> str:
        """Format detected platforms."""
        if not platforms:
            return "No CI/CD platforms detected in this project."

        lines = ["**Detected CI/CD Platforms:**"]
        for p in platforms:
            lines.append(f"- {p.value}")
        return "\n".join(lines)

    def _format_summary(self, summary: dict[str, Any]) -> str:
        """Format pipeline summary."""
        lines = ["**Pipeline Health Summary**", ""]

        lines.append(f"**Platforms:** {summary['platforms_detected']} detected")
        lines.append(f"**Configs:** {summary['pipeline_configs']} pipeline configuration(s)")
        lines.append("")

        # Issues
        lines.append("**Issues:**")
        lines.append(f"- {_get_icon('alert')} Critical: {summary['critical_issues']}")
        lines.append(f"- {_get_icon('warning')} Warnings: {summary['warning_issues']}")
        lines.append(f"- Total: {summary['total_issues']}")
        lines.append("")

        # Success rate
        if summary["success_rate"] > 0:
            lines.append(f"**Success Rate:** {summary['success_rate']:.1f}%")
        if summary["avg_duration_seconds"]:
            lines.append(f"**Avg Duration:** {summary['avg_duration_seconds']:.0f}s")
        lines.append("")

        # Coverage
        if summary["coverage"]:
            cov = summary["coverage"]
            lines.append("**Coverage:**")
            lines.append(f"- Lines: {cov['line_coverage']:.1f}%")
            lines.append(f"- Branches: {cov['branch_coverage']:.1f}%")
            lines.append(f"- Uncovered files: {cov['uncovered_files']}")
            lines.append("")

        lines.append(f"**Recommendations:** {summary['recommendations_count']} available")

        return "\n".join(lines)

    def _format_analysis(self, result: "PipelineAnalysisResult") -> str:
        """Format full analysis result."""
        lines = ["**Pipeline Analysis Report**", ""]

        # Configs
        lines.append(f"**Configurations:** {len(result.configs)} found")
        for config in result.configs:
            lines.append(f"- {config.file_path.name} ({config.platform.value})")
            lines.append(f"  Steps: {len(config.steps)}, Triggers: {', '.join(config.triggers)}")
        lines.append("")

        # Issues by severity
        lines.append("**Issues:**")
        critical = [i for i in result.issues if i.severity == "critical"]
        warnings = [i for i in result.issues if i.severity == "warning"]
        info = [i for i in result.issues if i.severity == "info"]

        if critical:
            lines.append(f"\n{_get_icon('alert')} **Critical ({len(critical)}):**")
            for issue in critical[:5]:
                lines.append(f"  - [{issue.category}] {issue.message}")
                if issue.recommendation:
                    lines.append(f"    {_get_icon('arrow_right')} {issue.recommendation}")

        if warnings:
            lines.append(f"\n{_get_icon('warning')} **Warnings ({len(warnings)}):**")
            for issue in warnings[:5]:
                lines.append(f"  - [{issue.category}] {issue.message}")

        if info:
            lines.append(f"\n{_get_icon('info')} **Info ({len(info)}):**")
            for issue in info[:3]:
                lines.append(f"  - {issue.message}")

        lines.append("")

        # Recommendations
        if result.recommendations:
            lines.append("**Recommendations:**")
            for rec in result.recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)

    def _format_coverage(self, coverage: CoverageMetrics) -> str:
        """Format coverage metrics."""
        lines = ["**Code Coverage Report**", ""]

        lines.append(f"**Format:** {coverage.report_format}")
        if coverage.report_path:
            lines.append(f"**Source:** {coverage.report_path.name}")
        lines.append("")

        lines.append("**Metrics:**")
        lines.append(f"- Line Coverage: {coverage.line_coverage:.1f}%")
        lines.append(f"  ({coverage.covered_lines}/{coverage.total_lines} lines)")
        lines.append(f"- Branch Coverage: {coverage.branch_coverage:.1f}%")
        lines.append(f"  ({coverage.covered_branches}/{coverage.total_branches} branches)")
        if coverage.total_functions > 0:
            lines.append(f"- Function Coverage: {coverage.function_coverage:.1f}%")
            lines.append(f"  ({coverage.covered_functions}/{coverage.total_functions} functions)")
        lines.append("")

        if coverage.uncovered_files:
            lines.append(f"**Uncovered Files ({len(coverage.uncovered_files)}):**")
            for f in coverage.uncovered_files[:10]:
                lines.append(f"- {f}")
            if len(coverage.uncovered_files) > 10:
                lines.append(f"... and {len(coverage.uncovered_files) - 10} more")

        return "\n".join(lines)

    def _format_comparison(self, comparison: dict[str, Any]) -> str:
        """Format coverage comparison."""
        lines = ["**Coverage Comparison**", ""]

        # Delta summary
        line_delta = comparison["line_coverage_delta"]
        branch_delta = comparison["branch_coverage_delta"]

        status = (
            f"{_get_icon('trend_up')} Improved"
            if comparison["improved"]
            else f"{_get_icon('trend_down')} Regressed"
        )
        lines.append(f"**Status:** {status}")
        lines.append("")

        lines.append("**Changes:**")
        lines.append(f"- Line Coverage: {line_delta:+.1f}%")
        lines.append(f"- Branch Coverage: {branch_delta:+.1f}%")
        if "function_coverage_delta" in comparison:
            lines.append(f"- Function Coverage: {comparison['function_coverage_delta']:+.1f}%")
        lines.append("")

        # Changed files
        if comparison.get("improved_files"):
            lines.append(f"**Improved Files ({len(comparison['improved_files'])}):**")
            for item in comparison["improved_files"][:5]:
                lines.append(f"  {_get_icon('success')} {item['file']} (+{item['delta']:.1f}%)")

        if comparison.get("regressed_files"):
            lines.append(f"\n**Regressed Files ({len(comparison['regressed_files'])}):**")
            for item in comparison["regressed_files"][:5]:
                lines.append(f"  {_get_icon('error')} {item['file']} ({item['delta']:.1f}%)")

        if comparison.get("new_uncovered_files"):
            lines.append(f"\n**New Uncovered Files ({len(comparison['new_uncovered_files'])}):**")
            for f in comparison["new_uncovered_files"][:5]:
                lines.append(f"  {_get_icon('warning')} {f}")

        return "\n".join(lines)
