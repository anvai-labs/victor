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

"""Pipeline analyzers for different CI/CD platforms and coverage formats.

This module provides concrete implementations of the analyzer protocols
for various CI/CD systems and coverage report formats.
"""

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from .protocol import (
    CoverageAnalyzerProtocol,
    CoverageMetrics,
    PipelineAnalyzerProtocol,
    PipelineConfig,
    PipelineIssue,
    PipelinePlatform,
    PipelineStep,
    StepType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Step Type Detection
# =============================================================================


def detect_step_type(step_name: str, commands: str | list[str] | None = None) -> StepType:
    """Detect the type of a pipeline step from its name and commands.

    Args:
        step_name: Name of the step
        commands: Command(s) executed by the step

    Returns:
        Detected step type
    """
    name_lower = step_name.lower()
    cmd_str = ""
    if commands:
        if isinstance(commands, list):
            cmd_str = " ".join(commands).lower()
        else:
            cmd_str = commands.lower()

    combined = f"{name_lower} {cmd_str}"

    # Build patterns
    if any(kw in combined for kw in ["build", "compile", "make", "cargo build", "go build"]):
        return StepType.BUILD

    # Test patterns
    if any(kw in combined for kw in ["test", "pytest", "jest", "mocha", "rspec", "cargo test"]):
        return StepType.TEST

    # Lint patterns
    if any(
        kw in combined for kw in ["lint", "eslint", "flake8", "ruff", "black", "mypy", "clippy"]
    ):
        return StepType.LINT

    # Security patterns
    if any(
        kw in combined
        for kw in [
            "security",
            "scan",
            "trivy",
            "snyk",
            "bandit",
            "safety",
            "audit",
            "cve",
        ]
    ):
        return StepType.SECURITY

    # Deploy patterns
    if any(kw in combined for kw in ["deploy", "publish", "release", "push", "upload"]):
        return StepType.DEPLOY

    # Artifact patterns
    if any(kw in combined for kw in ["artifact", "archive", "package"]):
        return StepType.ARTIFACT

    # Notification patterns
    if any(kw in combined for kw in ["notify", "slack", "email", "webhook"]):
        return StepType.NOTIFICATION

    # Cache patterns
    if any(kw in combined for kw in ["cache", "restore", "save"]):
        return StepType.CACHE

    return StepType.CUSTOM


# =============================================================================
# GitHub Actions Analyzer
# =============================================================================


class GitHubActionsAnalyzer(PipelineAnalyzerProtocol):
    """Analyzer for GitHub Actions workflows."""

    @property
    def platform(self) -> PipelinePlatform:
        return PipelinePlatform.GITHUB_ACTIONS

    async def detect_config_files(self, root_path: Path) -> list[Path]:
        """Find GitHub Actions workflow files."""
        workflows_dir = root_path / ".github" / "workflows"
        if not workflows_dir.exists():
            return []

        return list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))

    async def parse_config(self, config_path: Path) -> PipelineConfig:
        """Parse a GitHub Actions workflow file."""
        content = config_path.read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse {config_path}: {e}")
            return PipelineConfig(
                platform=self.platform,
                file_path=config_path,
                raw_content=content,
            )

        if not data:
            return PipelineConfig(
                platform=self.platform,
                file_path=config_path,
                raw_content=content,
            )

        # Extract triggers
        triggers = []
        branches = []
        if "on" in data:
            on_config = data["on"]
            if isinstance(on_config, str):
                triggers = [on_config]
            elif isinstance(on_config, list):
                triggers = on_config
            elif isinstance(on_config, dict):
                triggers = list(on_config.keys())
                # Extract branches from push/pull_request
                for trigger in ["push", "pull_request"]:
                    if trigger in on_config and isinstance(on_config[trigger], dict):
                        branches.extend(on_config[trigger].get("branches", []))

        # Extract jobs as steps
        steps = []
        jobs = data.get("jobs", {})
        for job_name, job_config in jobs.items():
            if not isinstance(job_config, dict):
                continue

            # Extract job steps
            job_steps = job_config.get("steps", [])
            for step in job_steps:
                if not isinstance(step, dict):
                    continue

                step_name = step.get("name", step.get("run", step.get("uses", "unnamed")))
                command = step.get("run")
                image = job_config.get("runs-on")

                steps.append(
                    PipelineStep(
                        name=f"{job_name}/{step_name}",
                        step_type=detect_step_type(step_name, command),
                        command=command,
                        image=image,
                        depends_on=job_config.get("needs", []),
                        environment=step.get("env", {}),
                    )
                )

        # Extract matrix
        matrix = {}
        for job_name, job_config in jobs.items():
            if isinstance(job_config, dict):
                strategy = job_config.get("strategy", {})
                if "matrix" in strategy:
                    matrix[job_name] = strategy["matrix"]

        return PipelineConfig(
            platform=self.platform,
            file_path=config_path,
            name=data.get("name"),
            triggers=triggers,
            branches=list(set(branches)),
            steps=steps,
            global_environment=data.get("env", {}),
            matrix=matrix,
            raw_content=content,
        )

    async def analyze(self, config: PipelineConfig) -> list[PipelineIssue]:
        """Analyze a GitHub Actions workflow for issues."""
        issues = []

        # Check for hardcoded secrets
        if "GITHUB_TOKEN" not in str(config.raw_content):
            pass  # Using default token is fine
        secret_patterns = [
            r'password\s*[:=]\s*["\'][^"\']+["\']',
            r'api[_-]?key\s*[:=]\s*["\'][^"\']+["\']',
            r'secret\s*[:=]\s*["\'][^"\']+["\']',
        ]
        for pattern in secret_patterns:
            if re.search(pattern, config.raw_content, re.IGNORECASE):
                issues.append(
                    PipelineIssue(
                        severity="critical",
                        category="security",
                        message="Potential hardcoded secret detected in workflow",
                        file_path=config.file_path,
                        recommendation="Use GitHub Secrets instead of hardcoded values",
                        auto_fixable=False,
                    )
                )
                break

        # Check for unpinned actions
        uses_pattern = r"uses:\s*([^\s@]+)@([^\s]+)"
        for match in re.finditer(uses_pattern, config.raw_content):
            action, version = match.groups()
            if version in ("main", "master", "latest"):
                issues.append(
                    PipelineIssue(
                        severity="warning",
                        category="security",
                        message=f"Action {action} uses unpinned version: {version}",
                        file_path=config.file_path,
                        recommendation="Pin actions to specific commit SHA or version tag",
                        auto_fixable=True,
                    )
                )

        # Check for missing security scanning
        has_security_step = any(s.step_type == StepType.SECURITY for s in config.steps)
        if not has_security_step:
            issues.append(
                PipelineIssue(
                    severity="warning",
                    category="security",
                    message="No security scanning step detected in workflow",
                    file_path=config.file_path,
                    recommendation="Add security scanning (e.g., trivy, snyk, codeql)",
                    auto_fixable=True,
                )
            )

        # Check for missing cache
        has_cache = any(
            "cache" in s.name.lower() or s.step_type == StepType.CACHE for s in config.steps
        )
        if not has_cache and len(config.steps) > 3:
            issues.append(
                PipelineIssue(
                    severity="info",
                    category="performance",
                    message="No caching configured in workflow",
                    file_path=config.file_path,
                    recommendation="Add dependency caching to speed up builds",
                    auto_fixable=True,
                )
            )

        # Check for missing timeout
        if "timeout-minutes" not in config.raw_content:
            issues.append(
                PipelineIssue(
                    severity="info",
                    category="reliability",
                    message="No job timeout configured",
                    file_path=config.file_path,
                    recommendation="Add timeout-minutes to prevent runaway jobs",
                    auto_fixable=True,
                )
            )

        return issues


# =============================================================================
# GitLab CI Analyzer
# =============================================================================


class GitLabCIAnalyzer(PipelineAnalyzerProtocol):
    """Analyzer for GitLab CI pipelines."""

    @property
    def platform(self) -> PipelinePlatform:
        return PipelinePlatform.GITLAB_CI

    async def detect_config_files(self, root_path: Path) -> list[Path]:
        """Find GitLab CI configuration files."""
        main_config = root_path / ".gitlab-ci.yml"
        configs = []
        if main_config.exists():
            configs.append(main_config)

        # Also check for includes
        ci_dir = root_path / ".gitlab-ci"
        if ci_dir.exists():
            configs.extend(ci_dir.glob("*.yml"))
            configs.extend(ci_dir.glob("*.yaml"))

        return configs

    async def parse_config(self, config_path: Path) -> PipelineConfig:
        """Parse a GitLab CI configuration file."""
        content = config_path.read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse {config_path}: {e}")
            return PipelineConfig(
                platform=self.platform,
                file_path=config_path,
                raw_content=content,
            )

        if not data:
            return PipelineConfig(
                platform=self.platform,
                file_path=config_path,
                raw_content=content,
            )

        # Extract global settings
        global_env = data.get("variables", {})
        services = data.get("services", [])
        default_image = data.get("image")

        # Reserved keywords that are not jobs
        reserved_keys = {
            "image",
            "services",
            "stages",
            "variables",
            "before_script",
            "after_script",
            "cache",
            "include",
            "default",
            "workflow",
        }

        # Extract jobs as steps
        steps = []

        for key, value in data.items():
            if key in reserved_keys or not isinstance(value, dict):
                continue

            job_name = key
            job_config = value

            command = job_config.get("script", [])
            if isinstance(command, list):
                command = "\n".join(command)

            image = job_config.get("image", default_image)
            needs = job_config.get("needs", [])
            if isinstance(needs, list):
                needs = [n if isinstance(n, str) else n.get("job", "") for n in needs]

            steps.append(
                PipelineStep(
                    name=job_name,
                    step_type=detect_step_type(job_name, command),
                    command=command,
                    image=image,
                    depends_on=needs,
                    environment=job_config.get("variables", {}),
                    artifacts=job_config.get("artifacts", {}).get("paths", []),
                    cache_paths=job_config.get("cache", {}).get("paths", []),
                )
            )

        return PipelineConfig(
            platform=self.platform,
            file_path=config_path,
            name=config_path.stem,
            triggers=["push", "merge_request"] if "workflow" not in data else [],
            steps=steps,
            global_environment=global_env,
            services=services if isinstance(services, list) else [],
            raw_content=content,
        )

    async def analyze(self, config: PipelineConfig) -> list[PipelineIssue]:
        """Analyze a GitLab CI pipeline for issues."""
        issues = []

        # Check for hardcoded secrets
        secret_patterns = [
            r"\$\{[A-Z_]+_PASSWORD\}",  # This is OK - using variables
            r'password\s*[:=]\s*["\'][^$][^"\']+["\']',  # Hardcoded password
        ]
        for pattern in secret_patterns[1:]:  # Skip the OK pattern
            if re.search(pattern, config.raw_content, re.IGNORECASE):
                issues.append(
                    PipelineIssue(
                        severity="critical",
                        category="security",
                        message="Potential hardcoded secret detected in pipeline",
                        file_path=config.file_path,
                        recommendation="Use GitLab CI/CD variables for secrets",
                        auto_fixable=False,
                    )
                )
                break

        # Check for missing SAST
        has_sast = "sast" in config.raw_content.lower() or any(
            "sast" in s.name.lower() for s in config.steps
        )
        if not has_sast:
            issues.append(
                PipelineIssue(
                    severity="warning",
                    category="security",
                    message="GitLab SAST not enabled",
                    file_path=config.file_path,
                    recommendation="Include GitLab SAST template: include: - template: Security/SAST.gitlab-ci.yml",
                    auto_fixable=True,
                )
            )

        # Check for missing retry configuration
        if "retry:" not in config.raw_content:
            issues.append(
                PipelineIssue(
                    severity="info",
                    category="reliability",
                    message="No retry configuration for flaky jobs",
                    file_path=config.file_path,
                    recommendation="Add retry: 2 for jobs that may fail intermittently",
                    auto_fixable=True,
                )
            )

        return issues


# =============================================================================
# Coverage Analyzers
# =============================================================================


class CoberturaAnalyzer(CoverageAnalyzerProtocol):
    """Analyzer for Cobertura XML coverage reports."""

    @property
    def format_name(self) -> str:
        return "cobertura"

    async def detect_reports(self, root_path: Path) -> list[Path]:
        """Find Cobertura coverage reports."""
        patterns = [
            "coverage.xml",
            "cobertura.xml",
            "**/coverage.xml",
            "**/cobertura-coverage.xml",
            "coverage-reports/*.xml",
        ]
        reports = []
        for pattern in patterns:
            reports.extend(root_path.glob(pattern))
        return reports

    async def parse_report(self, report_path: Path) -> CoverageMetrics:
        """Parse a Cobertura XML coverage report."""
        try:
            tree = ET.parse(report_path)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.warning(f"Failed to parse Cobertura report {report_path}: {e}")
            return CoverageMetrics(report_path=report_path, report_format=self.format_name)

        # Extract metrics from coverage tag
        line_rate = float(root.get("line-rate", 0))
        branch_rate = float(root.get("branch-rate", 0))

        total_lines = 0
        covered_lines = 0
        total_branches = 0
        covered_branches = 0
        coverage_by_file: dict[str, float] = {}
        uncovered_files: list[str] = []

        # Parse packages and classes
        for package in root.findall(".//package"):
            for cls in package.findall(".//class"):
                filename = cls.get("filename", "")
                cls_line_rate = float(cls.get("line-rate", 0))

                # Count lines
                for line in cls.findall(".//line"):
                    total_lines += 1
                    if int(line.get("hits", 0)) > 0:
                        covered_lines += 1

                    # Count branches
                    if line.get("branch") == "true":
                        condition_coverage = line.get("condition-coverage", "")
                        match = re.search(r"\((\d+)/(\d+)\)", condition_coverage)
                        if match:
                            covered_branches += int(match.group(1))
                            total_branches += int(match.group(2))

                if filename:
                    coverage_by_file[filename] = cls_line_rate * 100
                    if cls_line_rate < 0.01:
                        uncovered_files.append(filename)

        return CoverageMetrics(
            total_lines=total_lines,
            covered_lines=covered_lines,
            total_branches=total_branches,
            covered_branches=covered_branches,
            line_coverage=line_rate * 100,
            branch_coverage=branch_rate * 100,
            uncovered_files=uncovered_files,
            coverage_by_file=coverage_by_file,
            report_path=report_path,
            report_format=self.format_name,
        )


class LCOVAnalyzer(CoverageAnalyzerProtocol):
    """Analyzer for LCOV coverage reports."""

    @property
    def format_name(self) -> str:
        return "lcov"

    async def detect_reports(self, root_path: Path) -> list[Path]:
        """Find LCOV coverage reports."""
        patterns = [
            "lcov.info",
            "coverage.lcov",
            "**/lcov.info",
            "coverage/lcov.info",
        ]
        reports = []
        for pattern in patterns:
            reports.extend(root_path.glob(pattern))
        return reports

    async def parse_report(self, report_path: Path) -> CoverageMetrics:
        """Parse an LCOV coverage report."""
        content = report_path.read_text(encoding="utf-8")

        total_lines = 0
        covered_lines = 0
        total_functions = 0
        covered_functions = 0
        total_branches = 0
        covered_branches = 0
        coverage_by_file: dict[str, float] = {}
        uncovered_files: list[str] = []

        current_file = ""
        file_lines = 0
        file_covered = 0

        for line in content.split("\n"):
            line = line.strip()

            if line.startswith("SF:"):
                current_file = line[3:]
                file_lines = 0
                file_covered = 0

            elif line.startswith("LF:"):
                file_lines = int(line[3:])
                total_lines += file_lines

            elif line.startswith("LH:"):
                file_covered = int(line[3:])
                covered_lines += file_covered

            elif line.startswith("FNF:"):
                total_functions += int(line[4:])

            elif line.startswith("FNH:"):
                covered_functions += int(line[4:])

            elif line.startswith("BRF:"):
                total_branches += int(line[4:])

            elif line.startswith("BRH:"):
                covered_branches += int(line[4:])

            elif line == "end_of_record":
                if current_file and file_lines > 0:
                    coverage = (file_covered / file_lines) * 100
                    coverage_by_file[current_file] = coverage
                    if coverage < 1:
                        uncovered_files.append(current_file)

        line_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
        branch_coverage = (covered_branches / total_branches * 100) if total_branches > 0 else 0
        function_coverage = (
            (covered_functions / total_functions * 100) if total_functions > 0 else 0
        )

        return CoverageMetrics(
            total_lines=total_lines,
            covered_lines=covered_lines,
            total_branches=total_branches,
            covered_branches=covered_branches,
            total_functions=total_functions,
            covered_functions=covered_functions,
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            function_coverage=function_coverage,
            uncovered_files=uncovered_files,
            coverage_by_file=coverage_by_file,
            report_path=report_path,
            report_format=self.format_name,
        )


class JaCoCoAnalyzer(CoverageAnalyzerProtocol):
    """Analyzer for JaCoCo XML coverage reports (Java)."""

    @property
    def format_name(self) -> str:
        return "jacoco"

    async def detect_reports(self, root_path: Path) -> list[Path]:
        """Find JaCoCo coverage reports."""
        patterns = [
            "jacoco.xml",
            "**/jacoco.xml",
            "target/site/jacoco/jacoco.xml",
            "build/reports/jacoco/test/jacocoTestReport.xml",
        ]
        reports = []
        for pattern in patterns:
            reports.extend(root_path.glob(pattern))
        return reports

    async def parse_report(self, report_path: Path) -> CoverageMetrics:
        """Parse a JaCoCo XML coverage report."""
        try:
            tree = ET.parse(report_path)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.warning(f"Failed to parse JaCoCo report {report_path}: {e}")
            return CoverageMetrics(report_path=report_path, report_format=self.format_name)

        total_lines = 0
        covered_lines = 0
        total_branches = 0
        covered_branches = 0
        total_methods = 0
        covered_methods = 0
        coverage_by_file: dict[str, float] = {}
        uncovered_files: list[str] = []

        # Parse packages and classes
        for package in root.findall(".//package"):
            for source_file in package.findall(".//sourcefile"):
                filename = source_file.get("name", "")
                file_covered = 0
                file_total = 0

                for counter in source_file.findall("counter"):
                    counter_type = counter.get("type")
                    missed = int(counter.get("missed", 0))
                    covered = int(counter.get("covered", 0))

                    if counter_type == "LINE":
                        file_total += missed + covered
                        file_covered += covered
                        total_lines += missed + covered
                        covered_lines += covered
                    elif counter_type == "BRANCH":
                        total_branches += missed + covered
                        covered_branches += covered
                    elif counter_type == "METHOD":
                        total_methods += missed + covered
                        covered_methods += covered

                if filename and file_total > 0:
                    coverage = (file_covered / file_total) * 100
                    coverage_by_file[filename] = coverage
                    if coverage < 1:
                        uncovered_files.append(filename)

        line_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
        branch_coverage = (covered_branches / total_branches * 100) if total_branches > 0 else 0
        function_coverage = (covered_methods / total_methods * 100) if total_methods > 0 else 0

        return CoverageMetrics(
            total_lines=total_lines,
            covered_lines=covered_lines,
            total_branches=total_branches,
            covered_branches=covered_branches,
            total_functions=total_methods,
            covered_functions=covered_methods,
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            function_coverage=function_coverage,
            uncovered_files=uncovered_files,
            coverage_by_file=coverage_by_file,
            report_path=report_path,
            report_format=self.format_name,
        )


# =============================================================================
# Analyzer Registry
# =============================================================================


# Global registry of analyzers
PIPELINE_ANALYZERS: dict[PipelinePlatform, PipelineAnalyzerProtocol] = {
    PipelinePlatform.GITHUB_ACTIONS: GitHubActionsAnalyzer(),
    PipelinePlatform.GITLAB_CI: GitLabCIAnalyzer(),
}

COVERAGE_ANALYZERS: dict[str, CoverageAnalyzerProtocol] = {
    "cobertura": CoberturaAnalyzer(),
    "lcov": LCOVAnalyzer(),
    "jacoco": JaCoCoAnalyzer(),
}


def get_pipeline_analyzer(
    platform: PipelinePlatform,
) -> PipelineAnalyzerProtocol | None:
    """Get an analyzer for a specific CI/CD platform.

    Args:
        platform: The CI/CD platform

    Returns:
        The analyzer or None if not supported
    """
    return PIPELINE_ANALYZERS.get(platform)


def get_coverage_analyzer(format_name: str) -> CoverageAnalyzerProtocol | None:
    """Get an analyzer for a specific coverage format.

    Args:
        format_name: The coverage report format

    Returns:
        The analyzer or None if not supported
    """
    return COVERAGE_ANALYZERS.get(format_name.lower())


def get_all_pipeline_analyzers() -> list[PipelineAnalyzerProtocol]:
    """Get all registered pipeline analyzers."""
    return list(PIPELINE_ANALYZERS.values())


def get_all_coverage_analyzers() -> list[CoverageAnalyzerProtocol]:
    """Get all registered coverage analyzers."""
    return list(COVERAGE_ANALYZERS.values())
