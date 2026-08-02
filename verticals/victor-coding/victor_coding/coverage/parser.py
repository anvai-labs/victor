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

"""Coverage report parsers.

Defines the abstract parser interface and implementations for
different coverage formats.
"""

import json
import logging
import xml.etree.ElementTree as ET  # noqa: S405 -- ET.Element used only as a type; parsing goes through defusedxml
from abc import ABC, abstractmethod

from defusedxml.ElementTree import parse as _safe_xml_parse
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from victor_coding.coverage.protocol import (
    BranchCoverage,
    CoverageReport,
    CoverageStatus,
    FileCoverage,
    FunctionCoverage,
    LineCoverage,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class CoverageParser(Protocol):
    """Protocol for coverage report parsers."""

    @property
    def name(self) -> str:
        """Parser name/format identifier."""
        ...

    @property
    def file_patterns(self) -> list[str]:
        """Glob patterns for files this parser handles."""
        ...

    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle a file."""
        ...

    def parse(self, file_path: Path) -> CoverageReport:
        """Parse a coverage file into a report."""
        ...


class BaseCoverageParser(ABC):
    """Abstract base class for coverage parsers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Parser name/format identifier."""
        ...

    @property
    @abstractmethod
    def file_patterns(self) -> list[str]:
        """Glob patterns for files this parser handles."""
        ...

    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle a file.

        Default implementation checks file patterns.
        """
        import fnmatch

        name = file_path.name
        return any(fnmatch.fnmatch(name, pattern) for pattern in self.file_patterns)

    @abstractmethod
    def parse(self, file_path: Path) -> CoverageReport:
        """Parse a coverage file into a report.

        Args:
            file_path: Path to coverage file

        Returns:
            CoverageReport with parsed data
        """
        ...


class CoberturaParser(BaseCoverageParser):
    """Parser for Cobertura XML coverage format.

    Used by pytest-cov, coverage.py, Java, etc.
    """

    @property
    def name(self) -> str:
        return "cobertura"

    @property
    def file_patterns(self) -> list[str]:
        return ["coverage.xml", "cobertura.xml", "cobertura-coverage.xml"]

    def parse(self, file_path: Path) -> CoverageReport:
        """Parse Cobertura XML coverage report."""
        report = CoverageReport()

        try:
            tree = _safe_xml_parse(file_path)
            root = tree.getroot()

            # Parse packages/classes
            for package in root.findall(".//package"):
                for class_elem in package.findall("classes/class"):
                    file_coverage = self._parse_class(class_elem)
                    if file_coverage:
                        report.files[file_coverage.file_path] = file_coverage

        except Exception as e:
            logger.error(f"Failed to parse Cobertura XML: {e}")

        return report

    def _parse_class(self, class_elem: ET.Element) -> Optional[FileCoverage]:
        """Parse a class element from Cobertura XML."""
        filename = class_elem.get("filename")
        if not filename:
            return None

        file_path = Path(filename)
        coverage = FileCoverage(file_path=file_path)

        # Parse lines
        for line_elem in class_elem.findall(".//line"):
            line_num = int(line_elem.get("number", 0))
            hits = int(line_elem.get("hits", 0))
            branch = line_elem.get("branch", "false") == "true"

            status = CoverageStatus.COVERED if hits > 0 else CoverageStatus.NOT_COVERED

            branch_coverage = None
            if branch:
                condition_coverage = line_elem.get("condition-coverage", "")
                if condition_coverage:
                    # Parse "50% (1/2)" format
                    try:
                        parts = condition_coverage.split("(")[1].rstrip(")")
                        covered, total = map(int, parts.split("/"))
                        branch_coverage = (covered, total)
                        if covered < total:
                            status = CoverageStatus.PARTIAL
                    except (IndexError, ValueError):
                        pass

            coverage.lines[line_num] = LineCoverage(
                line_number=line_num,
                status=status,
                hit_count=hits,
                branch_coverage=branch_coverage,
            )

        # Parse methods as functions
        for method_elem in class_elem.findall(".//method"):
            method_name = method_elem.get("name", "")
            lines = method_elem.findall(".//line")
            if lines:
                start_line = min(int(ln.get("number", 0)) for ln in lines)
                end_line = max(int(ln.get("number", 0)) for ln in lines)
                total_hits = sum(int(ln.get("hits", 0)) for ln in lines)

                coverage.functions.append(
                    FunctionCoverage(
                        name=method_name,
                        start_line=start_line,
                        end_line=end_line,
                        hit_count=total_hits,
                        status=(
                            CoverageStatus.COVERED if total_hits > 0 else CoverageStatus.NOT_COVERED
                        ),
                    )
                )

        return coverage


class LcovParser(BaseCoverageParser):
    """Parser for LCOV/gcov coverage format.

    Used by many tools including Istanbul (JavaScript).
    """

    @property
    def name(self) -> str:
        return "lcov"

    @property
    def file_patterns(self) -> list[str]:
        return ["lcov.info", "*.lcov", "coverage.lcov"]

    def parse(self, file_path: Path) -> CoverageReport:
        """Parse LCOV coverage report."""
        report = CoverageReport()
        current_file: Optional[FileCoverage] = None

        try:
            with open(file_path) as f:
                for line in f:
                    line = line.strip()

                    if line.startswith("SF:"):
                        # Source file
                        source_file = Path(line[3:])
                        current_file = FileCoverage(file_path=source_file)

                    elif line.startswith("DA:"):
                        # Line data: DA:line,hit_count
                        if current_file:
                            parts = line[3:].split(",")
                            line_num = int(parts[0])
                            hits = int(parts[1]) if len(parts) > 1 else 0

                            current_file.lines[line_num] = LineCoverage(
                                line_number=line_num,
                                status=(
                                    CoverageStatus.COVERED
                                    if hits > 0
                                    else CoverageStatus.NOT_COVERED
                                ),
                                hit_count=hits,
                            )

                    elif line.startswith("FN:"):
                        # Function: FN:start_line,function_name
                        if current_file:
                            parts = line[3:].split(",")
                            if len(parts) >= 2:
                                start_line = int(parts[0])
                                func_name = parts[1]
                                current_file.functions.append(
                                    FunctionCoverage(
                                        name=func_name,
                                        start_line=start_line,
                                        end_line=start_line,  # Updated later
                                        hit_count=0,
                                    )
                                )

                    elif line.startswith("FNDA:"):
                        # Function hit data: FNDA:hit_count,function_name
                        if current_file:
                            parts = line[5:].split(",")
                            if len(parts) >= 2:
                                hits = int(parts[0])
                                func_name = parts[1]
                                for func in current_file.functions:
                                    if func.name == func_name:
                                        func.hit_count = hits
                                        func.status = (
                                            CoverageStatus.COVERED
                                            if hits > 0
                                            else CoverageStatus.NOT_COVERED
                                        )
                                        break

                    elif line.startswith("BRDA:"):
                        # Branch data: BRDA:line,block,branch,taken
                        if current_file:
                            parts = line[5:].split(",")
                            if len(parts) >= 4:
                                line_num = int(parts[0])
                                block_id = parts[1]
                                branch_id = parts[2]
                                taken = parts[3] != "-" and int(parts[3]) > 0

                                current_file.branches.append(
                                    BranchCoverage(
                                        line_number=line_num,
                                        branch_id=f"{block_id}:{branch_id}",
                                        taken=taken,
                                        hit_count=(int(parts[3]) if parts[3] != "-" else 0),
                                    )
                                )

                    elif line == "end_of_record":
                        # End of file record
                        if current_file:
                            report.files[current_file.file_path] = current_file
                            current_file = None

        except Exception as e:
            logger.error(f"Failed to parse LCOV: {e}")

        return report


class GoCoverParser(BaseCoverageParser):
    """Parser for Go coverage profile format."""

    @property
    def name(self) -> str:
        return "go"

    @property
    def file_patterns(self) -> list[str]:
        return ["coverage.out", "*.coverprofile", "cover.out"]

    def parse(self, file_path: Path) -> CoverageReport:
        """Parse Go coverage profile."""
        report = CoverageReport()
        files: dict[str, FileCoverage] = {}

        try:
            with open(file_path) as f:
                for line in f:
                    line = line.strip()

                    # Skip mode line
                    if line.startswith("mode:"):
                        continue

                    # Format: file:start.col,end.col statements count
                    parts = line.split()
                    if len(parts) != 3:
                        continue

                    location = parts[0]
                    int(parts[1])
                    count = int(parts[2])

                    # Parse location
                    file_part, range_part = location.rsplit(":", 1)
                    start_str, end_str = range_part.split(",")
                    start_line = int(start_str.split(".")[0])
                    end_line = int(end_str.split(".")[0])

                    # Get or create file coverage
                    file_path_obj = Path(file_part)
                    if file_part not in files:
                        files[file_part] = FileCoverage(file_path=file_path_obj)

                    coverage = files[file_part]

                    # Mark lines as covered/uncovered
                    for line_num in range(start_line, end_line + 1):
                        existing = coverage.lines.get(line_num)
                        if existing:
                            # Max of existing and new count
                            count = max(existing.hit_count, count)

                        coverage.lines[line_num] = LineCoverage(
                            line_number=line_num,
                            status=(
                                CoverageStatus.COVERED if count > 0 else CoverageStatus.NOT_COVERED
                            ),
                            hit_count=count,
                        )

            for _file_key, file_coverage in files.items():
                report.files[file_coverage.file_path] = file_coverage

        except Exception as e:
            logger.error(f"Failed to parse Go coverage: {e}")

        return report


class JestCoverageParser(BaseCoverageParser):
    """Parser for Jest JSON coverage format."""

    @property
    def name(self) -> str:
        return "jest"

    @property
    def file_patterns(self) -> list[str]:
        return ["coverage-final.json", "jest-coverage.json"]

    def parse(self, file_path: Path) -> CoverageReport:
        """Parse Jest JSON coverage report."""
        report = CoverageReport()

        try:
            with open(file_path) as f:
                data = json.load(f)

            for file_key, file_data in data.items():
                coverage = FileCoverage(file_path=Path(file_key))

                # Parse statement coverage
                statement_map = file_data.get("statementMap", {})
                statement_hits = file_data.get("s", {})

                for stmt_id, stmt_info in statement_map.items():
                    start_line = stmt_info.get("start", {}).get("line", 0)
                    hits = statement_hits.get(stmt_id, 0)

                    coverage.lines[start_line] = LineCoverage(
                        line_number=start_line,
                        status=(CoverageStatus.COVERED if hits > 0 else CoverageStatus.NOT_COVERED),
                        hit_count=hits,
                    )

                # Parse function coverage
                fn_map = file_data.get("fnMap", {})
                fn_hits = file_data.get("f", {})

                for fn_id, fn_info in fn_map.items():
                    name = fn_info.get("name", "")
                    start_line = fn_info.get("loc", {}).get("start", {}).get("line", 0)
                    end_line = fn_info.get("loc", {}).get("end", {}).get("line", 0)
                    hits = fn_hits.get(fn_id, 0)

                    coverage.functions.append(
                        FunctionCoverage(
                            name=name,
                            start_line=start_line,
                            end_line=end_line,
                            hit_count=hits,
                            status=(
                                CoverageStatus.COVERED if hits > 0 else CoverageStatus.NOT_COVERED
                            ),
                        )
                    )

                # Parse branch coverage
                branch_map = file_data.get("branchMap", {})
                branch_hits = file_data.get("b", {})

                for branch_id, branch_info in branch_map.items():
                    line = branch_info.get("loc", {}).get("start", {}).get("line", 0)
                    hits_list = branch_hits.get(branch_id, [])

                    for i, hits in enumerate(hits_list):
                        coverage.branches.append(
                            BranchCoverage(
                                line_number=line,
                                branch_id=f"{branch_id}:{i}",
                                taken=hits > 0,
                                hit_count=hits,
                            )
                        )

                report.files[coverage.file_path] = coverage

        except Exception as e:
            logger.error(f"Failed to parse Jest coverage: {e}")

        return report


class CloverParser(BaseCoverageParser):
    """Parser for Clover XML coverage format.

    Used by PHP, Java (Clover plugin), etc.
    """

    @property
    def name(self) -> str:
        return "clover"

    @property
    def file_patterns(self) -> list[str]:
        return ["clover.xml", "clover-coverage.xml"]

    def parse(self, file_path: Path) -> CoverageReport:
        """Parse Clover XML coverage report."""
        report = CoverageReport()

        try:
            tree = _safe_xml_parse(file_path)
            root = tree.getroot()

            for file_elem in root.findall(".//file"):
                filename = file_elem.get("name") or file_elem.get("path")
                if not filename:
                    continue

                coverage = FileCoverage(file_path=Path(filename))

                for line_elem in file_elem.findall("line"):
                    line_num = int(line_elem.get("num", 0))
                    line_type = line_elem.get("type", "")
                    hits = int(line_elem.get("count", 0))

                    if line_type == "method":
                        # Function/method
                        name = line_elem.get("name", "")
                        coverage.functions.append(
                            FunctionCoverage(
                                name=name,
                                start_line=line_num,
                                end_line=line_num,
                                hit_count=hits,
                                status=(
                                    CoverageStatus.COVERED
                                    if hits > 0
                                    else CoverageStatus.NOT_COVERED
                                ),
                            )
                        )
                    else:
                        # Regular line
                        coverage.lines[line_num] = LineCoverage(
                            line_number=line_num,
                            status=(
                                CoverageStatus.COVERED if hits > 0 else CoverageStatus.NOT_COVERED
                            ),
                            hit_count=hits,
                        )

                report.files[coverage.file_path] = coverage

        except Exception as e:
            logger.error(f"Failed to parse Clover XML: {e}")

        return report


# Registry of available parsers
COVERAGE_PARSERS: list[type[BaseCoverageParser]] = [
    CoberturaParser,
    LcovParser,
    GoCoverParser,
    JestCoverageParser,
    CloverParser,
]


def get_parser_for_file(file_path: Path) -> Optional[BaseCoverageParser]:
    """Get an appropriate parser for a coverage file.

    Args:
        file_path: Path to coverage file

    Returns:
        Parser instance or None if no parser matches
    """
    for parser_class in COVERAGE_PARSERS:
        parser = parser_class()
        if parser.can_parse(file_path):
            return parser
    return None


def parse_coverage_file(file_path: Path) -> Optional[CoverageReport]:
    """Parse a coverage file using an appropriate parser.

    Args:
        file_path: Path to coverage file

    Returns:
        CoverageReport or None if parsing fails
    """
    parser = get_parser_for_file(file_path)
    if parser is None:
        logger.warning(f"No parser found for: {file_path}")
        return None

    return parser.parse(file_path)
