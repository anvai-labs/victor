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

"""Coverage data types and protocol definitions.

Defines the data structures for representing code coverage
from multiple sources and languages.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class CoverageType(Enum):
    """Types of code coverage."""

    LINE = "line"  # Line-level coverage
    BRANCH = "branch"  # Branch/decision coverage
    FUNCTION = "function"  # Function/method coverage
    STATEMENT = "statement"  # Statement coverage


class CoverageStatus(Enum):
    """Coverage status for a line or branch."""

    COVERED = "covered"  # Executed during tests
    NOT_COVERED = "not_covered"  # Not executed
    PARTIAL = "partial"  # Partially covered (branches)


@dataclass
class LineCoverage:
    """Coverage data for a single line."""

    line_number: int
    status: CoverageStatus
    hit_count: int = 0  # Number of times executed
    branch_coverage: Optional[tuple[int, int]] = None  # (covered, total) branches


@dataclass
class FunctionCoverage:
    """Coverage data for a function/method."""

    name: str
    start_line: int
    end_line: int
    hit_count: int = 0
    status: CoverageStatus = CoverageStatus.NOT_COVERED

    @property
    def is_covered(self) -> bool:
        return self.status == CoverageStatus.COVERED


@dataclass
class BranchCoverage:
    """Coverage data for a branch point."""

    line_number: int
    branch_id: str
    taken: bool = False
    hit_count: int = 0


@dataclass
class FileCoverage:
    """Coverage data for a single file."""

    file_path: Path
    lines: dict[int, LineCoverage] = field(default_factory=dict)
    functions: list[FunctionCoverage] = field(default_factory=list)
    branches: list[BranchCoverage] = field(default_factory=list)

    @property
    def total_lines(self) -> int:
        """Total executable lines."""
        return len(self.lines)

    @property
    def covered_lines(self) -> int:
        """Number of covered lines."""
        return sum(1 for line in self.lines.values() if line.status == CoverageStatus.COVERED)

    @property
    def line_coverage_percent(self) -> float:
        """Line coverage percentage."""
        if self.total_lines == 0:
            return 0.0
        return (self.covered_lines / self.total_lines) * 100

    @property
    def total_functions(self) -> int:
        """Total functions."""
        return len(self.functions)

    @property
    def covered_functions(self) -> int:
        """Number of covered functions."""
        return sum(1 for f in self.functions if f.is_covered)

    @property
    def function_coverage_percent(self) -> float:
        """Function coverage percentage."""
        if self.total_functions == 0:
            return 0.0
        return (self.covered_functions / self.total_functions) * 100

    @property
    def total_branches(self) -> int:
        """Total branches."""
        return len(self.branches)

    @property
    def covered_branches(self) -> int:
        """Number of covered branches."""
        return sum(1 for b in self.branches if b.taken)

    @property
    def branch_coverage_percent(self) -> float:
        """Branch coverage percentage."""
        if self.total_branches == 0:
            return 0.0
        return (self.covered_branches / self.total_branches) * 100

    def get_uncovered_lines(self) -> list[int]:
        """Get list of uncovered line numbers."""
        return [
            line_num
            for line_num, line in self.lines.items()
            if line.status == CoverageStatus.NOT_COVERED
        ]

    def get_uncovered_ranges(self) -> list[tuple[int, int]]:
        """Get ranges of consecutive uncovered lines."""
        uncovered = sorted(self.get_uncovered_lines())
        if not uncovered:
            return []

        ranges = []
        start = uncovered[0]
        end = start

        for line in uncovered[1:]:
            if line == end + 1:
                end = line
            else:
                ranges.append((start, end))
                start = line
                end = line

        ranges.append((start, end))
        return ranges


@dataclass
class CoverageReport:
    """Aggregated coverage report for a project."""

    files: dict[Path, FileCoverage] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    commit_sha: Optional[str] = None
    branch_name: Optional[str] = None

    @property
    def total_lines(self) -> int:
        """Total executable lines across all files."""
        return sum(fc.total_lines for fc in self.files.values())

    @property
    def covered_lines(self) -> int:
        """Total covered lines across all files."""
        return sum(fc.covered_lines for fc in self.files.values())

    @property
    def line_coverage_percent(self) -> float:
        """Overall line coverage percentage."""
        if self.total_lines == 0:
            return 0.0
        return (self.covered_lines / self.total_lines) * 100

    @property
    def total_functions(self) -> int:
        """Total functions across all files."""
        return sum(fc.total_functions for fc in self.files.values())

    @property
    def covered_functions(self) -> int:
        """Total covered functions across all files."""
        return sum(fc.covered_functions for fc in self.files.values())

    @property
    def function_coverage_percent(self) -> float:
        """Overall function coverage percentage."""
        if self.total_functions == 0:
            return 0.0
        return (self.covered_functions / self.total_functions) * 100

    @property
    def total_branches(self) -> int:
        """Total branches across all files."""
        return sum(fc.total_branches for fc in self.files.values())

    @property
    def covered_branches(self) -> int:
        """Total covered branches across all files."""
        return sum(fc.covered_branches for fc in self.files.values())

    @property
    def branch_coverage_percent(self) -> float:
        """Overall branch coverage percentage."""
        if self.total_branches == 0:
            return 0.0
        return (self.covered_branches / self.total_branches) * 100

    @property
    def file_count(self) -> int:
        """Number of files with coverage data."""
        return len(self.files)

    def get_files_below_threshold(self, threshold: float) -> list[FileCoverage]:
        """Get files with coverage below a threshold.

        Args:
            threshold: Coverage percentage threshold

        Returns:
            List of files below threshold
        """
        return [fc for fc in self.files.values() if fc.line_coverage_percent < threshold]

    def get_top_uncovered_files(self, n: int = 10) -> list[FileCoverage]:
        """Get files with lowest coverage.

        Args:
            n: Number of files to return

        Returns:
            List of files sorted by coverage (lowest first)
        """
        sorted_files = sorted(
            self.files.values(),
            key=lambda f: f.line_coverage_percent,
        )
        return sorted_files[:n]

    def merge(self, other: "CoverageReport") -> "CoverageReport":
        """Merge another coverage report into this one.

        Takes the maximum hit count for each line.

        Args:
            other: Report to merge

        Returns:
            Merged report
        """
        merged = CoverageReport(
            timestamp=max(self.timestamp, other.timestamp),
            commit_sha=self.commit_sha or other.commit_sha,
            branch_name=self.branch_name or other.branch_name,
        )

        all_files = set(self.files.keys()) | set(other.files.keys())

        for file_path in all_files:
            self_coverage = self.files.get(file_path)
            other_coverage = other.files.get(file_path)

            if self_coverage is None:
                merged.files[file_path] = other_coverage  # type: ignore
            elif other_coverage is None:
                merged.files[file_path] = self_coverage
            else:
                # Merge line coverage
                merged_lines = {}
                all_lines = set(self_coverage.lines.keys()) | set(other_coverage.lines.keys())

                for line_num in all_lines:
                    self_line = self_coverage.lines.get(line_num)
                    other_line = other_coverage.lines.get(line_num)

                    if self_line is None:
                        merged_lines[line_num] = other_line  # type: ignore
                    elif other_line is None:
                        merged_lines[line_num] = self_line
                    else:
                        # Take max hit count
                        hit_count = max(self_line.hit_count, other_line.hit_count)
                        status = (
                            CoverageStatus.COVERED if hit_count > 0 else CoverageStatus.NOT_COVERED
                        )
                        merged_lines[line_num] = LineCoverage(
                            line_number=line_num,
                            status=status,
                            hit_count=hit_count,
                        )

                merged.files[file_path] = FileCoverage(
                    file_path=file_path,
                    lines=merged_lines,
                    functions=self_coverage.functions + other_coverage.functions,
                    branches=self_coverage.branches + other_coverage.branches,
                )

        return merged


@dataclass
class CoverageDiff:
    """Difference between two coverage reports."""

    before: CoverageReport
    after: CoverageReport
    new_covered_lines: int = 0
    new_uncovered_lines: int = 0
    removed_covered_lines: int = 0
    removed_uncovered_lines: int = 0

    @property
    def coverage_delta(self) -> float:
        """Change in overall coverage percentage."""
        return self.after.line_coverage_percent - self.before.line_coverage_percent

    @property
    def is_improvement(self) -> bool:
        """Whether coverage improved."""
        return self.coverage_delta > 0


@dataclass
class CoverageThreshold:
    """Coverage threshold configuration."""

    line_coverage: float = 80.0
    branch_coverage: float = 70.0
    function_coverage: float = 80.0
    fail_under: bool = True  # Fail if thresholds not met

    def check(self, report: CoverageReport) -> tuple[bool, list[str]]:
        """Check if report meets thresholds.

        Args:
            report: Coverage report to check

        Returns:
            Tuple of (passed, list of failure messages)
        """
        failures = []

        if report.line_coverage_percent < self.line_coverage:
            failures.append(
                f"Line coverage {report.line_coverage_percent:.1f}% " f"< {self.line_coverage}%"
            )

        if report.branch_coverage_percent < self.branch_coverage:
            failures.append(
                f"Branch coverage {report.branch_coverage_percent:.1f}% "
                f"< {self.branch_coverage}%"
            )

        if report.function_coverage_percent < self.function_coverage:
            failures.append(
                f"Function coverage {report.function_coverage_percent:.1f}% "
                f"< {self.function_coverage}%"
            )

        passed = len(failures) == 0
        return passed, failures
