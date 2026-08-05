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

"""Tests for coverage protocol types and data structures."""

import pytest
from datetime import datetime
from pathlib import Path

pytest.importorskip("victor_coding")

from victor_coding.coverage.protocol import (
    BranchCoverage,
    CoverageDiff,
    CoverageReport,
    CoverageStatus,
    CoverageThreshold,
    CoverageType,
    FileCoverage,
    FunctionCoverage,
    LineCoverage,
)

# =============================================================================
# ENUM TESTS
# =============================================================================


class TestCoverageType:
    """Tests for CoverageType enum."""

    def test_line_type(self):
        """Test LINE coverage type."""
        assert CoverageType.LINE.value == "line"

    def test_branch_type(self):
        """Test BRANCH coverage type."""
        assert CoverageType.BRANCH.value == "branch"

    def test_function_type(self):
        """Test FUNCTION coverage type."""
        assert CoverageType.FUNCTION.value == "function"

    def test_statement_type(self):
        """Test STATEMENT coverage type."""
        assert CoverageType.STATEMENT.value == "statement"


class TestCoverageStatus:
    """Tests for CoverageStatus enum."""

    def test_covered_status(self):
        """Test COVERED status."""
        assert CoverageStatus.COVERED.value == "covered"

    def test_not_covered_status(self):
        """Test NOT_COVERED status."""
        assert CoverageStatus.NOT_COVERED.value == "not_covered"

    def test_partial_status(self):
        """Test PARTIAL status."""
        assert CoverageStatus.PARTIAL.value == "partial"


# =============================================================================
# LINE COVERAGE TESTS
# =============================================================================


class TestLineCoverage:
    """Tests for LineCoverage dataclass."""

    def test_creation_minimal(self):
        """Test minimal line coverage creation."""
        line = LineCoverage(
            line_number=10,
            status=CoverageStatus.COVERED,
        )
        assert line.line_number == 10
        assert line.status == CoverageStatus.COVERED
        assert line.hit_count == 0
        assert line.branch_coverage is None

    def test_creation_with_hit_count(self):
        """Test line coverage with hit count."""
        line = LineCoverage(
            line_number=20,
            status=CoverageStatus.COVERED,
            hit_count=5,
        )
        assert line.hit_count == 5

    def test_creation_with_branch(self):
        """Test line coverage with branch data."""
        line = LineCoverage(
            line_number=30,
            status=CoverageStatus.PARTIAL,
            hit_count=3,
            branch_coverage=(1, 2),  # 1 of 2 branches covered
        )
        assert line.branch_coverage == (1, 2)


# =============================================================================
# FUNCTION COVERAGE TESTS
# =============================================================================


class TestFunctionCoverage:
    """Tests for FunctionCoverage dataclass."""

    def test_creation_minimal(self):
        """Test minimal function coverage creation."""
        func = FunctionCoverage(
            name="process_data",
            start_line=10,
            end_line=25,
        )
        assert func.name == "process_data"
        assert func.start_line == 10
        assert func.hit_count == 0
        assert func.status == CoverageStatus.NOT_COVERED

    def test_is_covered_true(self):
        """Test is_covered when covered."""
        func = FunctionCoverage(
            name="func",
            start_line=1,
            end_line=10,
            hit_count=1,
            status=CoverageStatus.COVERED,
        )
        assert func.is_covered is True

    def test_is_covered_false(self):
        """Test is_covered when not covered."""
        func = FunctionCoverage(
            name="func",
            start_line=1,
            end_line=10,
        )
        assert func.is_covered is False


# =============================================================================
# BRANCH COVERAGE TESTS
# =============================================================================


class TestBranchCoverage:
    """Tests for BranchCoverage dataclass."""

    def test_creation_minimal(self):
        """Test minimal branch coverage creation."""
        branch = BranchCoverage(
            line_number=15,
            branch_id="if_true",
        )
        assert branch.line_number == 15
        assert branch.branch_id == "if_true"
        assert branch.taken is False
        assert branch.hit_count == 0

    def test_creation_taken(self):
        """Test branch coverage when taken."""
        branch = BranchCoverage(
            line_number=15,
            branch_id="if_true",
            taken=True,
            hit_count=10,
        )
        assert branch.taken is True
        assert branch.hit_count == 10


# =============================================================================
# FILE COVERAGE TESTS
# =============================================================================


class TestFileCoverage:
    """Tests for FileCoverage dataclass."""

    @pytest.fixture
    def sample_file_coverage(self):
        """Create sample file coverage."""
        lines = {
            1: LineCoverage(1, CoverageStatus.COVERED, 5),
            2: LineCoverage(2, CoverageStatus.COVERED, 3),
            3: LineCoverage(3, CoverageStatus.NOT_COVERED),
            4: LineCoverage(4, CoverageStatus.COVERED, 1),
            5: LineCoverage(5, CoverageStatus.NOT_COVERED),
            6: LineCoverage(6, CoverageStatus.NOT_COVERED),
        }
        functions = [
            FunctionCoverage("func1", 1, 2, 5, CoverageStatus.COVERED),
            FunctionCoverage("func2", 3, 6, 0, CoverageStatus.NOT_COVERED),
        ]
        branches = [
            BranchCoverage(3, "if_true", True, 2),
            BranchCoverage(3, "if_false", False, 0),
        ]
        return FileCoverage(
            file_path=Path("test.py"),
            lines=lines,
            functions=functions,
            branches=branches,
        )

    def test_creation_minimal(self):
        """Test minimal file coverage creation."""
        fc = FileCoverage(file_path=Path("test.py"))
        assert fc.file_path == Path("test.py")
        assert fc.lines == {}
        assert fc.functions == []
        assert fc.branches == []

    def test_total_lines(self, sample_file_coverage):
        """Test total_lines property."""
        assert sample_file_coverage.total_lines == 6

    def test_covered_lines(self, sample_file_coverage):
        """Test covered_lines property."""
        assert sample_file_coverage.covered_lines == 3

    def test_line_coverage_percent(self, sample_file_coverage):
        """Test line_coverage_percent property."""
        assert sample_file_coverage.line_coverage_percent == 50.0

    def test_line_coverage_percent_empty(self):
        """Test line_coverage_percent with no lines."""
        fc = FileCoverage(file_path=Path("test.py"))
        assert fc.line_coverage_percent == 0.0

    def test_total_functions(self, sample_file_coverage):
        """Test total_functions property."""
        assert sample_file_coverage.total_functions == 2

    def test_covered_functions(self, sample_file_coverage):
        """Test covered_functions property."""
        assert sample_file_coverage.covered_functions == 1

    def test_function_coverage_percent(self, sample_file_coverage):
        """Test function_coverage_percent property."""
        assert sample_file_coverage.function_coverage_percent == 50.0

    def test_function_coverage_percent_empty(self):
        """Test function_coverage_percent with no functions."""
        fc = FileCoverage(file_path=Path("test.py"))
        assert fc.function_coverage_percent == 0.0

    def test_total_branches(self, sample_file_coverage):
        """Test total_branches property."""
        assert sample_file_coverage.total_branches == 2

    def test_covered_branches(self, sample_file_coverage):
        """Test covered_branches property."""
        assert sample_file_coverage.covered_branches == 1

    def test_branch_coverage_percent(self, sample_file_coverage):
        """Test branch_coverage_percent property."""
        assert sample_file_coverage.branch_coverage_percent == 50.0

    def test_branch_coverage_percent_empty(self):
        """Test branch_coverage_percent with no branches."""
        fc = FileCoverage(file_path=Path("test.py"))
        assert fc.branch_coverage_percent == 0.0

    def test_get_uncovered_lines(self, sample_file_coverage):
        """Test get_uncovered_lines method."""
        uncovered = sample_file_coverage.get_uncovered_lines()
        assert sorted(uncovered) == [3, 5, 6]

    def test_get_uncovered_ranges(self, sample_file_coverage):
        """Test get_uncovered_ranges method."""
        ranges = sample_file_coverage.get_uncovered_ranges()
        assert ranges == [(3, 3), (5, 6)]

    def test_get_uncovered_ranges_empty(self):
        """Test get_uncovered_ranges with no uncovered lines."""
        fc = FileCoverage(
            file_path=Path("test.py"),
            lines={
                1: LineCoverage(1, CoverageStatus.COVERED, 1),
            },
        )
        assert fc.get_uncovered_ranges() == []


# =============================================================================
# COVERAGE REPORT TESTS
# =============================================================================


class TestCoverageReport:
    """Tests for CoverageReport dataclass."""

    @pytest.fixture
    def sample_report(self):
        """Create sample coverage report."""
        file1 = FileCoverage(
            file_path=Path("a.py"),
            lines={
                1: LineCoverage(1, CoverageStatus.COVERED, 1),
                2: LineCoverage(2, CoverageStatus.COVERED, 1),
            },
            functions=[FunctionCoverage("f1", 1, 2, 1, CoverageStatus.COVERED)],
            branches=[BranchCoverage(1, "b1", True, 1)],
        )
        file2 = FileCoverage(
            file_path=Path("b.py"),
            lines={
                1: LineCoverage(1, CoverageStatus.NOT_COVERED),
                2: LineCoverage(2, CoverageStatus.NOT_COVERED),
            },
            functions=[FunctionCoverage("f2", 1, 2, 0, CoverageStatus.NOT_COVERED)],
            branches=[BranchCoverage(1, "b2", False, 0)],
        )
        return CoverageReport(
            files={Path("a.py"): file1, Path("b.py"): file2},
        )

    def test_creation_empty(self):
        """Test empty report creation."""
        report = CoverageReport()
        assert report.files == {}
        assert isinstance(report.timestamp, datetime)
        assert report.commit_sha is None

    def test_total_lines(self, sample_report):
        """Test total_lines property."""
        assert sample_report.total_lines == 4

    def test_covered_lines(self, sample_report):
        """Test covered_lines property."""
        assert sample_report.covered_lines == 2

    def test_line_coverage_percent(self, sample_report):
        """Test line_coverage_percent property."""
        assert sample_report.line_coverage_percent == 50.0

    def test_line_coverage_percent_empty(self):
        """Test line_coverage_percent with no files."""
        report = CoverageReport()
        assert report.line_coverage_percent == 0.0

    def test_total_functions(self, sample_report):
        """Test total_functions property."""
        assert sample_report.total_functions == 2

    def test_covered_functions(self, sample_report):
        """Test covered_functions property."""
        assert sample_report.covered_functions == 1

    def test_function_coverage_percent(self, sample_report):
        """Test function_coverage_percent property."""
        assert sample_report.function_coverage_percent == 50.0

    def test_function_coverage_percent_empty(self):
        """Test function_coverage_percent with no files."""
        report = CoverageReport()
        assert report.function_coverage_percent == 0.0

    def test_total_branches(self, sample_report):
        """Test total_branches property."""
        assert sample_report.total_branches == 2

    def test_covered_branches(self, sample_report):
        """Test covered_branches property."""
        assert sample_report.covered_branches == 1

    def test_branch_coverage_percent(self, sample_report):
        """Test branch_coverage_percent property."""
        assert sample_report.branch_coverage_percent == 50.0

    def test_branch_coverage_percent_empty(self):
        """Test branch_coverage_percent with no files."""
        report = CoverageReport()
        assert report.branch_coverage_percent == 0.0

    def test_file_count(self, sample_report):
        """Test file_count property."""
        assert sample_report.file_count == 2

    def test_get_files_below_threshold(self, sample_report):
        """Test get_files_below_threshold method."""
        below = sample_report.get_files_below_threshold(50.0)
        assert len(below) == 1
        assert below[0].file_path == Path("b.py")

    def test_get_top_uncovered_files(self, sample_report):
        """Test get_top_uncovered_files method."""
        top = sample_report.get_top_uncovered_files(1)
        assert len(top) == 1
        assert top[0].file_path == Path("b.py")

    def test_merge_no_overlap(self):
        """Test merge with no overlapping files."""
        report1 = CoverageReport(
            files={
                Path("a.py"): FileCoverage(
                    file_path=Path("a.py"),
                    lines={1: LineCoverage(1, CoverageStatus.COVERED, 1)},
                )
            },
            timestamp=datetime(2025, 1, 1),
        )
        report2 = CoverageReport(
            files={
                Path("b.py"): FileCoverage(
                    file_path=Path("b.py"),
                    lines={1: LineCoverage(1, CoverageStatus.COVERED, 1)},
                )
            },
            timestamp=datetime(2025, 1, 2),
        )
        merged = report1.merge(report2)
        assert len(merged.files) == 2
        assert merged.timestamp == datetime(2025, 1, 2)

    def test_merge_with_overlap(self):
        """Test merge with overlapping files."""
        report1 = CoverageReport(
            files={
                Path("a.py"): FileCoverage(
                    file_path=Path("a.py"),
                    lines={
                        1: LineCoverage(1, CoverageStatus.COVERED, 2),
                        2: LineCoverage(2, CoverageStatus.NOT_COVERED, 0),
                    },
                )
            },
        )
        report2 = CoverageReport(
            files={
                Path("a.py"): FileCoverage(
                    file_path=Path("a.py"),
                    lines={
                        1: LineCoverage(1, CoverageStatus.COVERED, 1),
                        2: LineCoverage(2, CoverageStatus.COVERED, 3),
                    },
                )
            },
        )
        merged = report1.merge(report2)
        assert merged.files[Path("a.py")].lines[1].hit_count == 2  # max
        assert merged.files[Path("a.py")].lines[2].hit_count == 3  # max
        assert merged.files[Path("a.py")].lines[2].status == CoverageStatus.COVERED


# =============================================================================
# COVERAGE DIFF TESTS
# =============================================================================


class TestCoverageDiff:
    """Tests for CoverageDiff dataclass."""

    def test_creation(self):
        """Test coverage diff creation."""
        before = CoverageReport(
            files={
                Path("a.py"): FileCoverage(
                    file_path=Path("a.py"),
                    lines={1: LineCoverage(1, CoverageStatus.COVERED, 1)},
                )
            }
        )
        after = CoverageReport(
            files={
                Path("a.py"): FileCoverage(
                    file_path=Path("a.py"),
                    lines={
                        1: LineCoverage(1, CoverageStatus.COVERED, 1),
                        2: LineCoverage(2, CoverageStatus.COVERED, 1),
                    },
                )
            }
        )
        diff = CoverageDiff(before=before, after=after)
        assert diff.before == before
        assert diff.after == after

    def test_coverage_delta_positive(self):
        """Test coverage_delta with improvement."""
        before = CoverageReport(
            files={
                Path("a.py"): FileCoverage(
                    file_path=Path("a.py"),
                    lines={
                        1: LineCoverage(1, CoverageStatus.COVERED, 1),
                        2: LineCoverage(2, CoverageStatus.NOT_COVERED),
                    },
                )
            }
        )
        after = CoverageReport(
            files={
                Path("a.py"): FileCoverage(
                    file_path=Path("a.py"),
                    lines={
                        1: LineCoverage(1, CoverageStatus.COVERED, 1),
                        2: LineCoverage(2, CoverageStatus.COVERED, 1),
                    },
                )
            }
        )
        diff = CoverageDiff(before=before, after=after)
        assert diff.coverage_delta == 50.0  # 50% -> 100%
        assert diff.is_improvement is True

    def test_coverage_delta_negative(self):
        """Test coverage_delta with regression."""
        before = CoverageReport(
            files={
                Path("a.py"): FileCoverage(
                    file_path=Path("a.py"),
                    lines={
                        1: LineCoverage(1, CoverageStatus.COVERED, 1),
                    },
                )
            }
        )
        after = CoverageReport(
            files={
                Path("a.py"): FileCoverage(
                    file_path=Path("a.py"),
                    lines={
                        1: LineCoverage(1, CoverageStatus.NOT_COVERED),
                    },
                )
            }
        )
        diff = CoverageDiff(before=before, after=after)
        assert diff.coverage_delta == -100.0
        assert diff.is_improvement is False


# =============================================================================
# COVERAGE THRESHOLD TESTS
# =============================================================================


class TestCoverageThreshold:
    """Tests for CoverageThreshold dataclass."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        threshold = CoverageThreshold()
        assert threshold.line_coverage == 80.0
        assert threshold.branch_coverage == 70.0
        assert threshold.function_coverage == 80.0
        assert threshold.fail_under is True

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        threshold = CoverageThreshold(
            line_coverage=90.0,
            branch_coverage=85.0,
            function_coverage=90.0,
            fail_under=False,
        )
        assert threshold.line_coverage == 90.0
        assert threshold.fail_under is False

    def test_check_passing(self):
        """Test check with passing report."""
        threshold = CoverageThreshold(
            line_coverage=50.0,
            branch_coverage=50.0,
            function_coverage=50.0,
        )
        report = CoverageReport(
            files={
                Path("a.py"): FileCoverage(
                    file_path=Path("a.py"),
                    lines={
                        1: LineCoverage(1, CoverageStatus.COVERED, 1),
                        2: LineCoverage(2, CoverageStatus.COVERED, 1),
                    },
                    functions=[FunctionCoverage("f1", 1, 2, 1, CoverageStatus.COVERED)],
                    branches=[BranchCoverage(1, "b1", True, 1)],
                )
            }
        )
        passed, failures = threshold.check(report)
        assert passed is True
        assert failures == []

    def test_check_failing_line(self):
        """Test check with failing line coverage."""
        threshold = CoverageThreshold(line_coverage=80.0)
        report = CoverageReport(
            files={
                Path("a.py"): FileCoverage(
                    file_path=Path("a.py"),
                    lines={
                        1: LineCoverage(1, CoverageStatus.COVERED, 1),
                        2: LineCoverage(2, CoverageStatus.NOT_COVERED),
                    },
                )
            }
        )
        passed, failures = threshold.check(report)
        assert passed is False
        assert len(failures) >= 1
        assert "Line coverage" in failures[0]

    def test_check_failing_branch(self):
        """Test check with failing branch coverage."""
        threshold = CoverageThreshold(branch_coverage=80.0)
        report = CoverageReport(
            files={
                Path("a.py"): FileCoverage(
                    file_path=Path("a.py"),
                    lines={1: LineCoverage(1, CoverageStatus.COVERED, 1)},
                    branches=[
                        BranchCoverage(1, "b1", True, 1),
                        BranchCoverage(1, "b2", False, 0),
                    ],
                )
            }
        )
        passed, failures = threshold.check(report)
        assert passed is False
        assert any("Branch coverage" in f for f in failures)

    def test_check_failing_function(self):
        """Test check with failing function coverage."""
        threshold = CoverageThreshold(function_coverage=80.0)
        report = CoverageReport(
            files={
                Path("a.py"): FileCoverage(
                    file_path=Path("a.py"),
                    lines={1: LineCoverage(1, CoverageStatus.COVERED, 1)},
                    functions=[
                        FunctionCoverage("f1", 1, 2, 1, CoverageStatus.COVERED),
                        FunctionCoverage("f2", 3, 4, 0, CoverageStatus.NOT_COVERED),
                    ],
                )
            }
        )
        passed, failures = threshold.check(report)
        assert passed is False
        assert any("Function coverage" in f for f in failures)
