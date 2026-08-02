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

"""Pipeline Analytics Protocol - Unified interface for CI/CD analysis.

This module defines the abstract interface and data structures for
pipeline analytics, supporting multiple CI/CD platforms:
- GitHub Actions
- GitLab CI
- Jenkins
- CircleCI
- Azure DevOps
- Bitbucket Pipelines
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class PipelinePlatform(str, Enum):
    """Supported CI/CD platforms."""

    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    CIRCLECI = "circleci"
    AZURE_DEVOPS = "azure_devops"
    BITBUCKET = "bitbucket"
    UNKNOWN = "unknown"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    RUNNING = "running"
    PENDING = "pending"
    SKIPPED = "skipped"


class StepType(str, Enum):
    """Pipeline step classification."""

    BUILD = "build"
    TEST = "test"
    LINT = "lint"
    SECURITY = "security"
    DEPLOY = "deploy"
    ARTIFACT = "artifact"
    NOTIFICATION = "notification"
    CACHE = "cache"
    CUSTOM = "custom"


@dataclass
class PipelineStep:
    """A single step/job in a pipeline."""

    name: str
    step_type: StepType
    command: str | None = None
    image: str | None = None
    timeout: int | None = None  # seconds
    retry_count: int = 0
    depends_on: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    cache_paths: list[str] = field(default_factory=list)
    estimated_duration: timedelta | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pipeline step to a dictionary.

        Returns:
            Dictionary with all step properties. The step_type is converted
            to its string value, and estimated_duration is converted to seconds.
        """
        return {
            "name": self.name,
            "step_type": self.step_type.value,
            "command": self.command,
            "image": self.image,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "depends_on": self.depends_on,
            "environment": self.environment,
            "artifacts": self.artifacts,
            "cache_paths": self.cache_paths,
            "estimated_duration": (
                self.estimated_duration.total_seconds() if self.estimated_duration else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineStep":
        """Deserialize a pipeline step from a dictionary.

        Args:
            data: Dictionary with step properties. Required keys: 'name', 'step_type'.
                  Optional keys match the dataclass fields.

        Returns:
            A new PipelineStep instance with values from the dictionary.
        """
        return cls(
            name=data["name"],
            step_type=StepType(data["step_type"]),
            command=data.get("command"),
            image=data.get("image"),
            timeout=data.get("timeout"),
            retry_count=data.get("retry_count", 0),
            depends_on=data.get("depends_on", []),
            environment=data.get("environment", {}),
            artifacts=data.get("artifacts", []),
            cache_paths=data.get("cache_paths", []),
            estimated_duration=(
                timedelta(seconds=data["estimated_duration"])
                if data.get("estimated_duration")
                else None
            ),
        )


@dataclass
class PipelineConfig:
    """Parsed pipeline configuration."""

    platform: PipelinePlatform
    file_path: Path
    name: str | None = None
    triggers: list[str] = field(default_factory=list)  # push, pull_request, schedule, etc.
    branches: list[str] = field(default_factory=list)
    steps: list[PipelineStep] = field(default_factory=list)
    global_environment: dict[str, str] = field(default_factory=dict)
    services: list[str] = field(default_factory=list)  # docker services
    matrix: dict[str, list[str]] = field(default_factory=dict)  # build matrix
    raw_content: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pipeline configuration to a dictionary.

        Returns:
            Dictionary with all configuration properties. Steps are recursively
            serialized, and Path objects are converted to strings.
        """
        return {
            "platform": self.platform.value,
            "file_path": str(self.file_path),
            "name": self.name,
            "triggers": self.triggers,
            "branches": self.branches,
            "steps": [s.to_dict() for s in self.steps],
            "global_environment": self.global_environment,
            "services": self.services,
            "matrix": self.matrix,
        }


@dataclass
class CoverageMetrics:
    """Code coverage metrics."""

    total_lines: int = 0
    covered_lines: int = 0
    total_branches: int = 0
    covered_branches: int = 0
    total_functions: int = 0
    covered_functions: int = 0
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    function_coverage: float = 0.0
    uncovered_files: list[str] = field(default_factory=list)
    coverage_by_file: dict[str, float] = field(default_factory=dict)
    report_path: Path | None = None
    report_format: str = "unknown"  # cobertura, lcov, jacoco, etc.

    def to_dict(self) -> dict[str, Any]:
        """Serialize the coverage metrics to a dictionary.

        Returns:
            Dictionary with all coverage metrics. Path objects are converted
            to strings. Suitable for JSON serialization or API responses.
        """
        return {
            "total_lines": self.total_lines,
            "covered_lines": self.covered_lines,
            "total_branches": self.total_branches,
            "covered_branches": self.covered_branches,
            "total_functions": self.total_functions,
            "covered_functions": self.covered_functions,
            "line_coverage": self.line_coverage,
            "branch_coverage": self.branch_coverage,
            "function_coverage": self.function_coverage,
            "uncovered_files": self.uncovered_files,
            "coverage_by_file": self.coverage_by_file,
            "report_path": str(self.report_path) if self.report_path else None,
            "report_format": self.report_format,
        }


@dataclass
class PipelineRun:
    """A single pipeline execution record."""

    run_id: str
    pipeline_name: str
    platform: PipelinePlatform
    status: PipelineStatus
    started_at: datetime
    finished_at: datetime | None = None
    duration: timedelta | None = None
    trigger: str = "manual"
    branch: str = "main"
    commit_sha: str | None = None
    steps_completed: int = 0
    steps_total: int = 0
    coverage: CoverageMetrics | None = None
    artifacts: list[str] = field(default_factory=list)
    logs_url: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pipeline run to a dictionary.

        Returns:
            Dictionary with all run properties. Datetime objects are converted
            to ISO format strings, timedelta to seconds, and enums to their
            string values. Coverage metrics are recursively serialized.
        """
        return {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "platform": self.platform.value,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration": self.duration.total_seconds() if self.duration else None,
            "trigger": self.trigger,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "coverage": self.coverage.to_dict() if self.coverage else None,
            "artifacts": self.artifacts,
            "logs_url": self.logs_url,
            "error_message": self.error_message,
        }


@dataclass
class PipelineIssue:
    """An issue or improvement opportunity in a pipeline."""

    severity: str  # critical, warning, info
    category: str  # security, performance, reliability, maintainability
    message: str
    file_path: Path | None = None
    line_number: int | None = None
    recommendation: str = ""
    auto_fixable: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pipeline issue to a dictionary.

        Returns:
            Dictionary with all issue properties. Path objects are converted
            to strings. Suitable for JSON serialization or reporting.
        """
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "file_path": str(self.file_path) if self.file_path else None,
            "line_number": self.line_number,
            "recommendation": self.recommendation,
            "auto_fixable": self.auto_fixable,
        }


@dataclass
class PipelineAnalysisResult:
    """Complete analysis result for a pipeline."""

    configs: list[PipelineConfig]
    issues: list[PipelineIssue]
    recent_runs: list[PipelineRun] = field(default_factory=list)
    coverage_trend: list[CoverageMetrics] = field(default_factory=list)
    success_rate: float = 0.0
    avg_duration: timedelta | None = None
    recommendations: list[str] = field(default_factory=list)
    analyzed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete analysis result to a dictionary.

        Returns:
            Dictionary with all analysis results. Nested objects (configs,
            issues, runs, coverage) are recursively serialized. Datetime
            objects are converted to ISO format strings.
        """
        return {
            "configs": [c.to_dict() for c in self.configs],
            "issues": [i.to_dict() for i in self.issues],
            "recent_runs": [r.to_dict() for r in self.recent_runs],
            "coverage_trend": [c.to_dict() for c in self.coverage_trend],
            "success_rate": self.success_rate,
            "avg_duration": (self.avg_duration.total_seconds() if self.avg_duration else None),
            "recommendations": self.recommendations,
            "analyzed_at": self.analyzed_at.isoformat(),
        }


class PipelineAnalyzerProtocol(ABC):
    """Abstract protocol for pipeline analysis.

    Implementations provide platform-specific parsing and analysis
    for different CI/CD systems.
    """

    @property
    @abstractmethod
    def platform(self) -> PipelinePlatform:
        """Return the platform this analyzer handles."""
        ...

    @abstractmethod
    async def detect_config_files(self, root_path: Path) -> list[Path]:
        """Find pipeline configuration files in a project.

        Args:
            root_path: Project root directory

        Returns:
            List of paths to pipeline config files
        """
        ...

    @abstractmethod
    async def parse_config(self, config_path: Path) -> PipelineConfig:
        """Parse a pipeline configuration file.

        Args:
            config_path: Path to the config file

        Returns:
            Parsed pipeline configuration
        """
        ...

    @abstractmethod
    async def analyze(self, config: PipelineConfig) -> list[PipelineIssue]:
        """Analyze a pipeline configuration for issues.

        Args:
            config: Parsed pipeline configuration

        Returns:
            List of issues and recommendations
        """
        ...


class CoverageAnalyzerProtocol(ABC):
    """Abstract protocol for coverage analysis.

    Implementations provide format-specific parsing for different
    coverage report formats.
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the coverage format this analyzer handles."""
        ...

    @abstractmethod
    async def detect_reports(self, root_path: Path) -> list[Path]:
        """Find coverage report files in a project.

        Args:
            root_path: Project root directory

        Returns:
            List of paths to coverage reports
        """
        ...

    @abstractmethod
    async def parse_report(self, report_path: Path) -> CoverageMetrics:
        """Parse a coverage report file.

        Args:
            report_path: Path to the coverage report

        Returns:
            Parsed coverage metrics
        """
        ...

    async def compare_coverage(
        self, baseline: CoverageMetrics, current: CoverageMetrics
    ) -> dict[str, Any]:
        """Compare two coverage reports.

        Args:
            baseline: Previous coverage metrics
            current: Current coverage metrics

        Returns:
            Dictionary with coverage changes
        """
        return {
            "line_coverage_delta": current.line_coverage - baseline.line_coverage,
            "branch_coverage_delta": current.branch_coverage - baseline.branch_coverage,
            "function_coverage_delta": current.function_coverage - baseline.function_coverage,
            "new_uncovered_files": [
                f for f in current.uncovered_files if f not in baseline.uncovered_files
            ],
            "improved_files": [
                f
                for f in current.coverage_by_file
                if f in baseline.coverage_by_file
                and current.coverage_by_file[f] > baseline.coverage_by_file[f]
            ],
            "regressed_files": [
                f
                for f in current.coverage_by_file
                if f in baseline.coverage_by_file
                and current.coverage_by_file[f] < baseline.coverage_by_file[f]
            ],
        }
