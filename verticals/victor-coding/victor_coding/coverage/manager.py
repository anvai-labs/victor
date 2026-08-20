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

"""Coverage manager for orchestrating coverage operations.

Provides a high-level API for collecting, analyzing, and visualizing
code coverage data.
"""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from victor_coding.coverage.parser import (
    COVERAGE_PARSERS,
    parse_coverage_file,
)
from victor_coding.coverage.protocol import (
    CoverageDiff,
    CoverageReport,
    CoverageThreshold,
)
from victor_coding.coverage.visualizer import CoverageVisualizer

logger = logging.getLogger(__name__)


class CoverageManager:
    """High-level manager for code coverage operations.

    Handles:
    - Running tests with coverage collection
    - Parsing coverage reports from various formats
    - Tracking coverage history
    - Generating visualizations and reports
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        history_dir: Optional[Path] = None,
        threshold: Optional[CoverageThreshold] = None,
    ):
        """Initialize the coverage manager.

        Args:
            project_root: Root directory of the project
            history_dir: Directory to store coverage history
            threshold: Coverage threshold configuration
        """
        self.project_root = project_root or Path.cwd()
        self.history_dir = history_dir or (self.project_root / ".coverage_history")
        self.threshold = threshold or CoverageThreshold()
        self.visualizer = CoverageVisualizer(threshold=self.threshold)
        self._current_report: Optional[CoverageReport] = None

    @property
    def current_report(self) -> Optional[CoverageReport]:
        """Get the current coverage report."""
        return self._current_report

    def collect_coverage(
        self,
        test_command: Optional[list[str]] = None,
        coverage_file: Optional[Path] = None,
    ) -> Optional[CoverageReport]:
        """Collect coverage by running tests or parsing existing file.

        Args:
            test_command: Command to run tests with coverage
            coverage_file: Path to existing coverage file

        Returns:
            CoverageReport or None if collection fails
        """
        if coverage_file:
            return self.parse_coverage(coverage_file)

        if test_command:
            return self.run_tests_with_coverage(test_command)

        # Auto-detect coverage files
        return self.discover_and_parse_coverage()

    def run_tests_with_coverage(
        self,
        command: list[str],
    ) -> Optional[CoverageReport]:
        """Run tests with coverage and parse results.

        Args:
            command: Test command to execute

        Returns:
            CoverageReport or None if collection fails
        """
        logger.info(f"Running tests: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning(
                    f"Tests exited with code {result.returncode}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            # Try to find and parse coverage file
            return self.discover_and_parse_coverage()

        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            return None

    def parse_coverage(self, file_path: Path) -> Optional[CoverageReport]:
        """Parse a coverage file.

        Args:
            file_path: Path to coverage file

        Returns:
            CoverageReport or None if parsing fails
        """
        report = parse_coverage_file(file_path)
        if report:
            self._current_report = report
            self._enrich_report(report)
        return report

    def discover_and_parse_coverage(self) -> Optional[CoverageReport]:
        """Discover and parse coverage files in the project.

        Returns:
            CoverageReport or None if no coverage found
        """
        # Common coverage file locations
        search_paths = [
            self.project_root,
            self.project_root / "coverage",
            self.project_root / ".coverage",
            self.project_root / "htmlcov",
            self.project_root / "target" / "coverage",  # Rust
        ]

        # Patterns from all parsers
        patterns = []
        for parser_class in COVERAGE_PARSERS:
            parser = parser_class()
            patterns.extend(parser.file_patterns)

        # Search for coverage files
        for search_path in search_paths:
            if not search_path.exists():
                continue

            for pattern in patterns:
                for file_path in search_path.glob(pattern):
                    if file_path.is_file():
                        logger.info(f"Found coverage file: {file_path}")
                        report = self.parse_coverage(file_path)
                        if report:
                            return report

        logger.warning("No coverage files found")
        return None

    def check_threshold(
        self,
        report: Optional[CoverageReport] = None,
    ) -> tuple[bool, list[str]]:
        """Check if coverage meets thresholds.

        Args:
            report: Report to check (uses current if not provided)

        Returns:
            Tuple of (passed, list of failure messages)
        """
        report = report or self._current_report
        if report is None:
            return False, ["No coverage report available"]

        return self.threshold.check(report)

    def save_report(
        self,
        report: Optional[CoverageReport] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save coverage report to history.

        Args:
            report: Report to save (uses current if not provided)
            output_path: Custom output path

        Returns:
            Path where report was saved
        """
        report = report or self._current_report
        if report is None:
            raise ValueError("No coverage report to save")

        self.history_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.history_dir / f"coverage_{timestamp}.json"

        # Serialize report
        data = self._serialize_report(report)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved coverage report: {output_path}")
        return output_path

    def load_report(self, file_path: Path) -> CoverageReport:
        """Load a coverage report from history.

        Args:
            file_path: Path to saved report

        Returns:
            CoverageReport
        """
        with open(file_path) as f:
            data = json.load(f)

        return self._deserialize_report(data)

    def get_coverage_history(
        self,
        limit: int = 10,
    ) -> list[tuple[Path, CoverageReport]]:
        """Get coverage history.

        Args:
            limit: Maximum number of reports to return

        Returns:
            List of (path, report) tuples, newest first
        """
        if not self.history_dir.exists():
            return []

        reports = []
        files = sorted(
            self.history_dir.glob("coverage_*.json"),
            reverse=True,
        )

        for file_path in files[:limit]:
            try:
                report = self.load_report(file_path)
                reports.append((file_path, report))
            except Exception as e:
                logger.warning(f"Failed to load report {file_path}: {e}")

        return reports

    def compare_reports(
        self,
        before: CoverageReport,
        after: CoverageReport,
    ) -> CoverageDiff:
        """Compare two coverage reports.

        Args:
            before: Earlier report
            after: Later report

        Returns:
            CoverageDiff with comparison data
        """
        diff = CoverageDiff(before=before, after=after)

        # Calculate line changes
        for file_path in after.files:
            if file_path not in before.files:
                # New file
                after_cov = after.files[file_path]
                diff.new_covered_lines += after_cov.covered_lines
                diff.new_uncovered_lines += after_cov.total_lines - after_cov.covered_lines
            else:
                # Compare lines
                before_cov = before.files[file_path]
                after_cov = after.files[file_path]

                for line_num, after_line in after_cov.lines.items():
                    before_line = before_cov.lines.get(line_num)
                    if before_line is None:
                        if after_line.hit_count > 0:
                            diff.new_covered_lines += 1
                        else:
                            diff.new_uncovered_lines += 1
                    elif before_line.hit_count == 0 and after_line.hit_count > 0:
                        diff.new_covered_lines += 1
                    elif before_line.hit_count > 0 and after_line.hit_count == 0:
                        diff.removed_covered_lines += 1

        # Handle removed files
        for file_path in before.files:
            if file_path not in after.files:
                before_cov = before.files[file_path]
                diff.removed_covered_lines += before_cov.covered_lines
                diff.removed_uncovered_lines += before_cov.total_lines - before_cov.covered_lines

        return diff

    def generate_text_report(
        self,
        report: Optional[CoverageReport] = None,
    ) -> str:
        """Generate a text coverage report.

        Args:
            report: Report to visualize (uses current if not provided)

        Returns:
            Text report string
        """
        report = report or self._current_report
        if report is None:
            return "No coverage report available"

        return self.visualizer.generate_text_report(report)

    def generate_html_report(
        self,
        report: Optional[CoverageReport] = None,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Generate an HTML coverage report.

        Args:
            report: Report to visualize (uses current if not provided)
            output_dir: Output directory

        Returns:
            Path to index.html
        """
        report = report or self._current_report
        if report is None:
            raise ValueError("No coverage report available")

        output_dir = output_dir or (self.project_root / "htmlcov")
        return self.visualizer.generate_html_report(report, output_dir)

    def generate_badge(
        self,
        report: Optional[CoverageReport] = None,
        format: str = "svg",
    ) -> str:
        """Generate a coverage badge.

        Args:
            report: Report to use (uses current if not provided)
            format: Badge format (svg, markdown, text)

        Returns:
            Badge content
        """
        report = report or self._current_report
        if report is None:
            return self.visualizer.generate_badge(
                CoverageReport(),  # Empty report for 0% badge
                format=format,
            )

        return self.visualizer.generate_badge(report, format=format)

    def get_uncovered_lines(
        self,
        file_path: Path,
        report: Optional[CoverageReport] = None,
    ) -> list[int]:
        """Get uncovered lines for a specific file.

        Args:
            file_path: Path to the file
            report: Report to use (uses current if not provided)

        Returns:
            List of uncovered line numbers
        """
        report = report or self._current_report
        if report is None:
            return []

        # Try exact match first
        if file_path in report.files:
            return report.files[file_path].get_uncovered_lines()

        # Try relative path matching
        for rpt_path, file_cov in report.files.items():
            if file_path.name == rpt_path.name:
                return file_cov.get_uncovered_lines()
            try:
                if file_path.resolve() == rpt_path.resolve():
                    return file_cov.get_uncovered_lines()
            except Exception:
                pass

        return []

    def _enrich_report(self, report: CoverageReport) -> None:
        """Enrich report with git information."""
        try:
            # Get current commit
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                report.commit_sha = result.stdout.strip()

            # Get current branch
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                report.branch_name = result.stdout.strip()

        except Exception:
            pass  # Git info is optional

    def _serialize_report(self, report: CoverageReport) -> dict:
        """Serialize a coverage report to JSON-compatible dict."""
        return {
            "timestamp": report.timestamp.isoformat(),
            "commit_sha": report.commit_sha,
            "branch_name": report.branch_name,
            "summary": {
                "total_lines": report.total_lines,
                "covered_lines": report.covered_lines,
                "line_coverage_percent": report.line_coverage_percent,
                "total_functions": report.total_functions,
                "covered_functions": report.covered_functions,
                "function_coverage_percent": report.function_coverage_percent,
                "total_branches": report.total_branches,
                "covered_branches": report.covered_branches,
                "branch_coverage_percent": report.branch_coverage_percent,
            },
            "files": {
                str(path): {
                    "lines": {
                        str(ln): {
                            "status": lc.status.value,
                            "hit_count": lc.hit_count,
                        }
                        for ln, lc in fc.lines.items()
                    },
                    "line_coverage_percent": fc.line_coverage_percent,
                }
                for path, fc in report.files.items()
            },
        }

    def _deserialize_report(self, data: dict) -> CoverageReport:
        """Deserialize a coverage report from JSON-compatible dict."""
        from victor_coding.coverage.protocol import LineCoverage, CoverageStatus

        report = CoverageReport(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            commit_sha=data.get("commit_sha"),
            branch_name=data.get("branch_name"),
        )

        for path_str, file_data in data.get("files", {}).items():
            from victor_coding.coverage.protocol import FileCoverage

            file_cov = FileCoverage(file_path=Path(path_str))

            for line_str, line_data in file_data.get("lines", {}).items():
                line_num = int(line_str)
                file_cov.lines[line_num] = LineCoverage(
                    line_number=line_num,
                    status=CoverageStatus(line_data["status"]),
                    hit_count=line_data.get("hit_count", 0),
                )

            report.files[file_cov.file_path] = file_cov

        return report


# Global manager singleton
_coverage_manager: Optional[CoverageManager] = None


def get_coverage_manager(
    project_root: Optional[Path] = None,
) -> CoverageManager:
    """Get the global coverage manager.

    Args:
        project_root: Project root directory

    Returns:
        CoverageManager instance
    """
    global _coverage_manager
    if _coverage_manager is None or (
        project_root and _coverage_manager.project_root != project_root
    ):
        _coverage_manager = CoverageManager(project_root=project_root)
    return _coverage_manager


def reset_coverage_manager() -> None:
    """Reset the global coverage manager."""
    global _coverage_manager
    _coverage_manager = None
