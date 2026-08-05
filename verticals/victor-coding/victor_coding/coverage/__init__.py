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

"""Coverage visualization and analysis system.

Provides comprehensive coverage parsing, visualization, and management
for multiple coverage formats (Cobertura, LCOV, Go, Jest, Clover).
"""

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
from victor_coding.coverage.parser import (
    BaseCoverageParser,
    CloverParser,
    CoberturaParser,
    CoverageParser,
    GoCoverParser,
    JestCoverageParser,
    LcovParser,
    get_parser_for_file,
    parse_coverage_file,
)
from victor_coding.coverage.visualizer import CoverageVisualizer
from victor_coding.coverage.manager import (
    CoverageManager,
    get_coverage_manager,
    reset_coverage_manager,
)

__all__ = [
    # Protocol types
    "BranchCoverage",
    "CoverageDiff",
    "CoverageReport",
    "CoverageStatus",
    "CoverageThreshold",
    "CoverageType",
    "FileCoverage",
    "FunctionCoverage",
    "LineCoverage",
    # Parsers
    "BaseCoverageParser",
    "CloverParser",
    "CoberturaParser",
    "CoverageParser",
    "GoCoverParser",
    "JestCoverageParser",
    "LcovParser",
    "get_parser_for_file",
    "parse_coverage_file",
    # Visualizer
    "CoverageVisualizer",
    # Manager
    "CoverageManager",
    "get_coverage_manager",
    "reset_coverage_manager",
]
