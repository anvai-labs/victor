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

"""Pipeline Analytics Manager - Orchestrates CI/CD and coverage analysis.

This module provides the PipelineManager class that coordinates multiple
analyzers to provide comprehensive pipeline analytics, coverage tracking,
and optimization recommendations.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from victor.config.settings import get_project_paths, load_settings

from .analyzers import (
    get_all_coverage_analyzers,
    get_all_pipeline_analyzers,
    get_coverage_analyzer,
    get_pipeline_analyzer,
)
from .protocol import (
    CoverageMetrics,
    PipelineAnalysisResult,
    PipelineConfig,
    PipelineIssue,
    PipelinePlatform,
    PipelineRun,
)

logger = logging.getLogger(__name__)


class PipelineManager:
    """Manager for pipeline analytics and coverage tracking.

    This class orchestrates multiple analyzers to provide:
    - CI/CD configuration analysis
    - Coverage report parsing and trending
    - Pipeline optimization recommendations
    - Historical run tracking

    Configuration is driven by settings.py for consistency with Victor.
    """

    def __init__(self, root_path: str | Path | None = None):
        """Initialize the pipeline manager.

        Args:
            root_path: Root directory of the project. Defaults to current directory.
        """
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self._settings = load_settings()
        self._paths = get_project_paths(self.root_path)
        self._history_file = self._paths.project_victor_dir / "pipeline_history.json"
        self._coverage_history_file = self._paths.project_victor_dir / "coverage_history.json"

    async def detect_platforms(self) -> list[PipelinePlatform]:
        """Detect which CI/CD platforms are configured in the project.

        Returns:
            List of detected CI/CD platforms
        """
        detected = []
        for analyzer in get_all_pipeline_analyzers():
            configs = await analyzer.detect_config_files(self.root_path)
            if configs:
                detected.append(analyzer.platform)
                logger.debug(f"Detected {analyzer.platform.value}: {len(configs)} config(s)")

        return detected

    async def analyze_pipelines(
        self,
        platforms: list[PipelinePlatform] | None = None,
        include_coverage: bool = True,
    ) -> PipelineAnalysisResult:
        """Perform comprehensive pipeline analysis.

        Args:
            platforms: Specific platforms to analyze. If None, auto-detects all.
            include_coverage: Whether to include coverage analysis.

        Returns:
            Complete analysis result with configs, issues, and recommendations
        """
        configs: list[PipelineConfig] = []
        issues: list[PipelineIssue] = []

        # Determine platforms to analyze
        if platforms is None:
            platforms = await self.detect_platforms()

        # Analyze each platform
        for platform in platforms:
            analyzer = get_pipeline_analyzer(platform)
            if not analyzer:
                logger.warning(f"No analyzer for platform: {platform}")
                continue

            config_files = await analyzer.detect_config_files(self.root_path)
            for config_file in config_files:
                try:
                    config = await analyzer.parse_config(config_file)
                    configs.append(config)

                    # Analyze for issues
                    config_issues = await analyzer.analyze(config)
                    issues.extend(config_issues)
                    logger.info(
                        f"Analyzed {config_file.name}: {len(config.steps)} steps, "
                        f"{len(config_issues)} issues"
                    )
                except Exception as e:
                    logger.error(f"Error analyzing {config_file}: {e}")
                    issues.append(
                        PipelineIssue(
                            severity="critical",
                            category="maintainability",
                            message=f"Failed to parse pipeline config: {e}",
                            file_path=config_file,
                        )
                    )

        # Analyze coverage
        coverage_trend = []
        if include_coverage:
            coverage_trend = await self._analyze_coverage()

        # Generate recommendations
        recommendations = self._generate_recommendations(configs, issues, coverage_trend)

        # Load historical runs
        recent_runs = await self._load_run_history()

        # Calculate success rate
        success_rate = 0.0
        avg_duration = None
        if recent_runs:
            successful = sum(1 for r in recent_runs if r.status.value == "success")
            success_rate = (successful / len(recent_runs)) * 100
            durations = [r.duration for r in recent_runs if r.duration]
            if durations:
                avg_duration = timedelta(
                    seconds=sum(d.total_seconds() for d in durations) / len(durations)
                )

        return PipelineAnalysisResult(
            configs=configs,
            issues=issues,
            recent_runs=recent_runs[:10],
            coverage_trend=coverage_trend,
            success_rate=success_rate,
            avg_duration=avg_duration,
            recommendations=recommendations,
        )

    async def _analyze_coverage(self) -> list[CoverageMetrics]:
        """Analyze coverage reports from all formats.

        Returns:
            List of coverage metrics from detected reports
        """
        metrics = []
        for analyzer in get_all_coverage_analyzers():
            try:
                reports = await analyzer.detect_reports(self.root_path)
                for report in reports:
                    coverage = await analyzer.parse_report(report)
                    if coverage.total_lines > 0:
                        metrics.append(coverage)
                        logger.info(
                            f"Parsed {analyzer.format_name} coverage from {report.name}: "
                            f"{coverage.line_coverage:.1f}% lines"
                        )
            except Exception as e:
                logger.warning(f"Error analyzing {analyzer.format_name} coverage: {e}")

        return metrics

    async def get_coverage(self, format_hint: str | None = None) -> CoverageMetrics | None:
        """Get current coverage metrics.

        Args:
            format_hint: Optional hint for which format to prefer

        Returns:
            Coverage metrics or None if no coverage found
        """
        if format_hint:
            analyzer = get_coverage_analyzer(format_hint)
            if analyzer:
                reports = await analyzer.detect_reports(self.root_path)
                if reports:
                    return await analyzer.parse_report(reports[0])

        # Auto-detect
        all_coverage = await self._analyze_coverage()
        if all_coverage:
            # Return the one with most detailed data
            return max(all_coverage, key=lambda c: c.total_lines)
        return None

    async def compare_coverage(self, baseline_path: str | Path | None = None) -> dict[str, Any]:
        """Compare current coverage against baseline.

        Args:
            baseline_path: Path to baseline coverage. If None, uses stored history.

        Returns:
            Comparison results with deltas and changed files
        """
        current = await self.get_coverage()
        if not current:
            return {"error": "No current coverage found"}

        # Load baseline
        baseline = None
        if baseline_path:
            path = Path(baseline_path)
            for analyzer in get_all_coverage_analyzers():
                if path.suffix in [".xml", ".info", ".json"]:
                    try:
                        baseline = await analyzer.parse_report(path)
                        if baseline.total_lines > 0:
                            break
                    except Exception:
                        continue
        else:
            # Load from history
            history = await self._load_coverage_history()
            if history:
                baseline = history[-1]

        if not baseline:
            return {"error": "No baseline coverage found", "current": current.to_dict()}

        # Compare
        comparison = {
            "current": current.to_dict(),
            "baseline": baseline.to_dict(),
            "line_coverage_delta": current.line_coverage - baseline.line_coverage,
            "branch_coverage_delta": current.branch_coverage - baseline.branch_coverage,
            "function_coverage_delta": current.function_coverage - baseline.function_coverage,
            "improved": current.line_coverage > baseline.line_coverage,
            "new_uncovered_files": [
                f for f in current.uncovered_files if f not in baseline.uncovered_files
            ],
            "improved_files": [],
            "regressed_files": [],
        }

        # Find improved/regressed files
        for file, coverage in current.coverage_by_file.items():
            if file in baseline.coverage_by_file:
                delta = coverage - baseline.coverage_by_file[file]
                if delta > 1:
                    comparison["improved_files"].append({"file": file, "delta": delta})
                elif delta < -1:
                    comparison["regressed_files"].append({"file": file, "delta": delta})

        return comparison

    async def record_run(self, run: PipelineRun) -> None:
        """Record a pipeline run to history.

        Args:
            run: The pipeline run to record
        """
        history = await self._load_run_history()
        history.insert(0, run)

        # Keep last 100 runs
        history = history[:100]

        # Ensure directory exists
        self._history_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self._history_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in history], f, indent=2)

        logger.info(f"Recorded pipeline run: {run.run_id} ({run.status.value})")

    async def record_coverage(self, metrics: CoverageMetrics) -> None:
        """Record coverage metrics to history.

        Args:
            metrics: The coverage metrics to record
        """
        history = await self._load_coverage_history()
        history.append(metrics)

        # Keep last 30 entries
        history = history[-30:]

        # Ensure directory exists
        self._coverage_history_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self._coverage_history_file, "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in history], f, indent=2)

        logger.info(f"Recorded coverage: {metrics.line_coverage:.1f}% lines")

    async def _load_run_history(self) -> list[PipelineRun]:
        """Load pipeline run history from storage."""
        if not self._history_file.exists():
            return []

        try:
            with open(self._history_file, encoding="utf-8") as f:
                data = json.load(f)
                runs = []
                for item in data:
                    runs.append(
                        PipelineRun(
                            run_id=item["run_id"],
                            pipeline_name=item["pipeline_name"],
                            platform=PipelinePlatform(item["platform"]),
                            status=item["status"],
                            started_at=datetime.fromisoformat(item["started_at"]),
                            finished_at=(
                                datetime.fromisoformat(item["finished_at"])
                                if item.get("finished_at")
                                else None
                            ),
                            duration=(
                                timedelta(seconds=item["duration"])
                                if item.get("duration")
                                else None
                            ),
                            trigger=item.get("trigger", "manual"),
                            branch=item.get("branch", "main"),
                            commit_sha=item.get("commit_sha"),
                            steps_completed=item.get("steps_completed", 0),
                            steps_total=item.get("steps_total", 0),
                        )
                    )
                return runs
        except Exception as e:
            logger.warning(f"Failed to load run history: {e}")
            return []

    async def _load_coverage_history(self) -> list[CoverageMetrics]:
        """Load coverage history from storage."""
        if not self._coverage_history_file.exists():
            return []

        try:
            with open(self._coverage_history_file, encoding="utf-8") as f:
                data = json.load(f)
                metrics = []
                for item in data:
                    metrics.append(
                        CoverageMetrics(
                            total_lines=item.get("total_lines", 0),
                            covered_lines=item.get("covered_lines", 0),
                            total_branches=item.get("total_branches", 0),
                            covered_branches=item.get("covered_branches", 0),
                            total_functions=item.get("total_functions", 0),
                            covered_functions=item.get("covered_functions", 0),
                            line_coverage=item.get("line_coverage", 0),
                            branch_coverage=item.get("branch_coverage", 0),
                            function_coverage=item.get("function_coverage", 0),
                            uncovered_files=item.get("uncovered_files", []),
                            coverage_by_file=item.get("coverage_by_file", {}),
                            report_format=item.get("report_format", "unknown"),
                        )
                    )
                return metrics
        except Exception as e:
            logger.warning(f"Failed to load coverage history: {e}")
            return []

    def _generate_recommendations(
        self,
        configs: list[PipelineConfig],
        issues: list[PipelineIssue],
        coverage: list[CoverageMetrics],
    ) -> list[str]:
        """Generate actionable recommendations based on analysis.

        Args:
            configs: Parsed pipeline configurations
            issues: Detected issues
            coverage: Coverage metrics

        Returns:
            List of recommendations
        """
        recommendations = []

        # Issue-based recommendations
        critical_issues = sum(1 for i in issues if i.severity == "critical")
        if critical_issues > 0:
            recommendations.append(
                f"🚨 Address {critical_issues} critical issue(s) immediately - "
                "security vulnerabilities or broken configs detected"
            )

        security_issues = sum(1 for i in issues if i.category == "security")
        if security_issues > 0:
            recommendations.append(
                f"🔒 Review {security_issues} security finding(s) - "
                "consider adding security scanning steps"
            )

        # Coverage recommendations
        if coverage:
            avg_coverage = sum(c.line_coverage for c in coverage) / len(coverage)
            if avg_coverage < 50:
                recommendations.append(
                    f"📊 Coverage is low ({avg_coverage:.1f}%) - "
                    "prioritize adding tests for critical paths"
                )
            elif avg_coverage < 80:
                recommendations.append(
                    f"📈 Coverage is moderate ({avg_coverage:.1f}%) - "
                    "consider increasing test coverage"
                )

            # Find uncovered files
            all_uncovered = set()
            for c in coverage:
                all_uncovered.update(c.uncovered_files)
            if all_uncovered:
                recommendations.append(f"📝 {len(all_uncovered)} file(s) have no test coverage")

        # Config-based recommendations
        has_matrix = any(c.matrix for c in configs)
        if not has_matrix and len(configs) > 0:
            recommendations.append(
                "🧪 Consider adding matrix builds to test across multiple versions/platforms"
            )

        # Check for test parallelization
        test_steps = sum(1 for c in configs for s in c.steps if s.step_type.value == "test")
        if test_steps > 3:
            recommendations.append(
                "⚡ Multiple test jobs detected - ensure parallel execution is configured"
            )

        # Check for caching
        has_cache = any(
            any(s.cache_paths for s in c.steps) or "cache" in c.raw_content.lower() for c in configs
        )
        if not has_cache and len(configs) > 0:
            recommendations.append("💨 Add dependency caching to speed up pipeline execution")

        return recommendations

    async def get_pipeline_summary(self) -> dict[str, Any]:
        """Get a high-level summary of pipeline health.

        Returns:
            Summary dictionary with key metrics
        """
        result = await self.analyze_pipelines()

        summary = {
            "platforms_detected": len({c.platform for c in result.configs}),
            "pipeline_configs": len(result.configs),
            "total_issues": len(result.issues),
            "critical_issues": sum(1 for i in result.issues if i.severity == "critical"),
            "warning_issues": sum(1 for i in result.issues if i.severity == "warning"),
            "success_rate": result.success_rate,
            "avg_duration_seconds": (
                result.avg_duration.total_seconds() if result.avg_duration else None
            ),
            "coverage": None,
            "recommendations_count": len(result.recommendations),
        }

        if result.coverage_trend:
            latest_coverage = result.coverage_trend[-1]
            summary["coverage"] = {
                "line_coverage": latest_coverage.line_coverage,
                "branch_coverage": latest_coverage.branch_coverage,
                "uncovered_files": len(latest_coverage.uncovered_files),
            }

        return summary
